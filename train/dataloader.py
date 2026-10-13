"""
HCPE / PSV 学習データローダー（447bit 特徴量版）。

explore_experiment/util/dataloader.py を基に、本リポジトリの
HybridAlphaZeroNet（入力 shape: (B, 447)、Value は Tanh → [-1, 1]）向けに移植した。
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Union

import numpy as np
import torch
from cshogi import BLACK, Board, HuffmanCodedPosAndEval, PackedSfenValue, move16_from_psv

from game.board import TOTAL_BITS, bit447_to_numpy, make_move_label, make_result

logger = logging.getLogger(__name__)


def _to_tanh_value(result_01: float) -> float:
    """勝敗ラベル [0, 1] を Value head（Tanh）向けの [-1, 1] に変換する。"""
    return float(result_01) * 2.0 - 1.0


def _psv_score_to_value(score: object, turn: int, scale: float) -> float:
    """PSV の score（先手視点 cp 想定）を Value 教師 [-1, 1] に変換する。"""
    s = float(score)
    v = 1.0 / (1.0 + math.exp(-s / scale))
    if turn != BLACK:
        v = 1.0 - v
    return _to_tanh_value(v)


def _normalize_files(files: Union[list[str], tuple[str], str]) -> list[str]:
    if isinstance(files, str):
        return [files]
    return list(files)


class HcpeDataLoader:
    """HuffmanCodedPosAndEval (.hcpe) を読み、447bit 特徴量バッチを返す。"""

    def __init__(
        self,
        files: Union[list[str], tuple[str], str],
        batch_size: int,
        device: torch.device,
        shuffle: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.limit = limit
        self.load(files)

        pin = device.type != "cpu"
        self.torch_features = torch.empty(
            (batch_size, TOTAL_BITS),
            dtype=torch.float32,
            pin_memory=pin,
        )
        self.torch_move_label = torch.empty((batch_size,), dtype=torch.int64, pin_memory=pin)
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=pin)

        self.features = self.torch_features.numpy()
        self.move_label = self.torch_move_label.numpy()
        self.result = self.torch_result.numpy().reshape(-1)

        self.i = 0
        self.l = 0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.f: Optional[Future] = None
        self.board = Board()

    def load(self, files: Union[list[str], tuple[str], str]) -> None:
        data: list[np.ndarray] = []
        for path in _normalize_files(files):
            if os.path.exists(path):
                logger.info("Loading HCPE: %s", path)
                data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval))
            else:
                logger.warning("%s not found, skipping", path)

        if not data:
            raise FileNotFoundError(f"HCPE file(s) not found or empty: {files}")

        self.data = np.concatenate(data)
        if self.limit is not None:
            if self.shuffle:
                np.random.shuffle(self.data)
            self.data = self.data[: self.limit]

    def mini_batch(self, hcpevec: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for i, hcpe in enumerate(hcpevec):
            self.board.set_hcp(hcpe["hcp"])
            self.features[i] = bit447_to_numpy(self.board)
            self.move_label[i] = make_move_label(int(hcpe["bestMove16"]), self.board.turn)
            self.result[i] = _to_tanh_value(make_result(int(hcpe["gameResult"]), self.board.turn))

        if self.device.type == "cpu":
            return (
                self.torch_features.clone(),
                self.torch_move_label.clone(),
                self.torch_result.clone(),
            )
        return (
            self.torch_features.to(self.device),
            self.torch_move_label.to(self.device),
            self.torch_result.to(self.device),
        )

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    def pre_fetch(self) -> None:
        hcpevec = self.data[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            return
        self.f = self.executor.submit(self.mini_batch, hcpevec)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> "HcpeDataLoader":
        self.i = 0
        self.l = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.l > len(self.data) or self.f is None:
            raise StopIteration()

        result = self.f.result()
        self.l = self.i
        self.pre_fetch()
        return result


class PsvDataLoader:
    """PackedSfenValue (.bin) を読み、HcpeDataLoader と同じインターフェースで返す。"""

    def __init__(
        self,
        files: Union[list[str], tuple[str], str],
        batch_size: int,
        device: torch.device,
        shuffle: bool = False,
        limit: Optional[int] = None,
        score_scale: float = 600.0,
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.limit = limit
        self.score_scale = score_scale
        self.load(files)

        pin = device.type != "cpu"
        self.torch_features = torch.empty(
            (batch_size, TOTAL_BITS),
            dtype=torch.float32,
            pin_memory=pin,
        )
        self.torch_move_label = torch.empty((batch_size,), dtype=torch.int64, pin_memory=pin)
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=pin)

        self.features = self.torch_features.numpy()
        self.move_label = self.torch_move_label.numpy()
        self.result = self.torch_result.numpy().reshape(-1)

        self.i = 0
        self.l = 0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.f: Optional[Future] = None
        self.board = Board()

    def load(self, files: Union[list[str], tuple[str], str]) -> None:
        data: list[np.ndarray] = []
        for path in _normalize_files(files):
            if os.path.exists(path):
                logger.info("Loading PSV: %s", path)
                data.append(np.fromfile(path, dtype=PackedSfenValue))
            else:
                logger.warning("%s not found, skipping", path)

        if not data:
            raise FileNotFoundError(f"PSV file(s) not found or empty: {files}")

        self.data = np.concatenate(data)
        if self.limit is not None:
            if self.shuffle:
                np.random.shuffle(self.data)
            self.data = self.data[: self.limit]

    def mini_batch(self, psvvec: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for i, psv in enumerate(psvvec):
            self.board.set_psfen(psv["sfen"])
            self.features[i] = bit447_to_numpy(self.board)
            move16 = move16_from_psv(int(psv["move"]))
            self.move_label[i] = make_move_label(move16, self.board.turn)
            self.result[i] = _psv_score_to_value(psv["score"], self.board.turn, self.score_scale)

        if self.device.type == "cpu":
            return (
                self.torch_features.clone(),
                self.torch_move_label.clone(),
                self.torch_result.clone(),
            )
        return (
            self.torch_features.to(self.device),
            self.torch_move_label.to(self.device),
            self.torch_result.to(self.device),
        )

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    def pre_fetch(self) -> None:
        psvvec = self.data[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        if len(psvvec) < self.batch_size:
            return
        self.f = self.executor.submit(self.mini_batch, psvvec)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> "PsvDataLoader":
        self.i = 0
        self.l = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.l > len(self.data) or self.f is None:
            raise StopIteration()

        result = self.f.result()
        self.l = self.i
        self.pre_fetch()
        return result
