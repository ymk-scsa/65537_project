"""
教師あり学習エントリポイント。

HCPE / PSV から 447bit 特徴量を読み、HybridAlphaZeroNet を学習する。
既存 weights からの再開 (--resume) に対応。
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.model import create_model
from train.dataloader import HcpeDataLoader, PsvDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_weights(model: nn.Module, path: str, device: torch.device) -> bool:
    if not path or not os.path.exists(path):
        return False
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    logger.info("Resumed weights from %s", path)
    return True


def _wdl_target(values: torch.Tensor, data_format: str) -> torch.Tensor:
    """
    価値ラベル（-1〜+1）から 勝ち/引き分け/負け の目標分布を作る（apple 系モデル用）。
      hcpe: 対局結果 (+1/0/-1) → one-hot（0 は引き分け）
      psv : 評価値由来の連続値 → 勝ち=(1+v)/2, 負け=(1-v)/2, 引き分け=0
    """
    v = values.view(-1)
    if data_format == "hcpe":
        win = (v > 0.5).float()
        loss = (v < -0.5).float()
        draw = 1.0 - win - loss
    else:
        win = (1.0 + v) * 0.5
        loss = (1.0 - v) * 0.5
        draw = torch.zeros_like(v)
    return torch.stack([win, draw, loss], dim=1)


def _compute_loss(model, features, policies, values, criterion_policy, criterion_value, data_format):
    if getattr(model, "uses_turn", False):
        # 手番別ヘッド（WDL）: 特徴量の 448 列目が手番
        policy_out, wdl_logits = model.forward_wdl(features)
        target = _wdl_target(values, data_format)
        value_loss = -(target * torch.log_softmax(wdl_logits, dim=1)).sum(dim=1).mean()
        return criterion_policy(policy_out, policies) + value_loss
    policy_out, value_out = model(features, return_aux=False)
    return criterion_policy(policy_out, policies) + criterion_value(value_out, values)


def _save_weights(model: nn.Module, path: str, backup: bool = True) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if backup and os.path.exists(path):
        bak = path + ".bak"
        shutil.copy2(path, bak)
    torch.save(model.state_dict(), path)
    logger.info("Model saved to %s", path)


def train(
    train_data: list[str],
    test_data: list[str] | None = None,
    data_format: str = "hcpe",
    mode: str = "modelC",
    epochs: int = 1,
    batch_size: int = 256,
    test_batch_size: int = 256,
    lr: float = 0.001,
    limit: int | None = None,
    weights_path: str = "weights/checkpoint.pth",
    resume: bool = True,
    save_every: int = 1,
    psv_score_scale: float = 600.0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s mode=%s format=%s resume=%s", device, mode, data_format, resume)

    model = create_model(num_actions=2187, mode=mode)
    model.to(device)
    with_turn = bool(getattr(model, "uses_turn", False))
    if resume:
        _load_weights(model, weights_path, device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    fmt = data_format.strip().lower()
    if fmt == "hcpe":
        train_loader = HcpeDataLoader(
            train_data, batch_size, device, shuffle=True, limit=limit, with_turn=with_turn
        )
        test_loader = (
            HcpeDataLoader(
                test_data, test_batch_size, device, shuffle=False, limit=limit, with_turn=with_turn
            )
            if test_data
            else None
        )
    elif fmt == "psv":
        train_loader = PsvDataLoader(
            train_data,
            batch_size,
            device,
            shuffle=True,
            limit=limit,
            score_scale=psv_score_scale,
            with_turn=with_turn,
        )
        test_loader = (
            PsvDataLoader(
                test_data,
                test_batch_size,
                device,
                shuffle=False,
                limit=limit,
                score_scale=psv_score_scale,
                with_turn=with_turn,
            )
            if test_data
            else None
        )
    else:
        raise ValueError(f"data_format must be 'hcpe' or 'psv', got {data_format!r}")

    logger.info("train positions = %s", len(train_loader))
    if test_loader is not None:
        logger.info("test positions = %s", len(test_loader))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for features, policies, values in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _compute_loss(
                model, features, policies, values, criterion_policy, criterion_value, fmt
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            if n_batches % 100 == 0:
                logger.info(
                    "  epoch %s batch %s/%s loss=%.4f",
                    epoch,
                    n_batches,
                    max(len(train_loader) // batch_size, 1),
                    loss.item(),
                )

        avg = epoch_loss / max(n_batches, 1)
        msg = f"Epoch {epoch}/{epochs} - train_loss: {avg:.4f} - time: {time.time() - t0:.1f}s"

        if test_loader is not None:
            model.eval()
            test_loss = 0.0
            test_batches = 0
            with torch.no_grad():
                for features, policies, values in test_loader:
                    loss = _compute_loss(
                        model, features, policies, values, criterion_policy, criterion_value, fmt
                    )
                    test_loss += loss.item()
                    test_batches += 1
            msg += f" - test_loss: {test_loss / max(test_batches, 1):.4f}"

        logger.info(msg)

        if save_every > 0 and epoch % save_every == 0:
            _save_weights(model, weights_path, backup=True)
            snap = weights_path.replace(".pth", f"_ep{epoch}.pth")
            if snap != weights_path:
                torch.save(model.state_dict(), snap)
                logger.info("Snapshot saved to %s", snap)

    _save_weights(model, weights_path, backup=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised training for GNN-Shogi (447bit)")
    p.add_argument(
        "--train_data",
        nargs="+",
        default=["train_data/2024_train.hcpe"],
        help="Training HCPE/PSV file(s)",
    )
    p.add_argument(
        "--test_data",
        nargs="*",
        default=["train_data/2024_test.hcpe"],
        help="Optional test HCPE/PSV file(s)",
    )
    p.add_argument("--data_format", choices=["hcpe", "psv"], default="hcpe")
    p.add_argument("--mode", default="modelC", help="Model architecture mode")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--test_batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--limit", type=int, default=None, help="Optional cap on positions")
    p.add_argument("--weights", default="weights/checkpoint.pth")
    p.add_argument("--resume", type=int, default=1, choices=[0, 1], help="1=load existing weights")
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--psv_score_scale", type=float, default=600.0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    test = args.test_data if args.test_data else None
    train(
        train_data=args.train_data,
        test_data=test,
        data_format=args.data_format,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        lr=args.lr,
        limit=args.limit,
        weights_path=args.weights,
        resume=bool(args.resume),
        save_every=args.save_every,
        psv_score_scale=args.psv_score_scale,
    )
