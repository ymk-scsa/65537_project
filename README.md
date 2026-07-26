# GNN-Shogi

CNN / GNN / Transformer などを組み合わせたハイブリッド評価関数と、独自探索を備えた将棋 AI です。将棋所・ShogiGUI などと **USI** で通信します。

盤面は **447bit**（盤面 405bit + 持ち駒 42bit）に圧縮し、埋め込み後に Policy / Value を出力します。

## 必要条件

- WSL2 + **Ubuntu-22.04**
- プロジェクト直下の **`.venv`**（bat はこれを使う）
- CUDA 対応 GPU（任意。なければ CPU）

```bash
# Ubuntu-22.04 上で
cd /mnt/c/Users/kotet/Documents/program/65537_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# GPU 版 torch 例:
# pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Windows の bat（`bat/*.bat`）は内部で `wsl -d Ubuntu-22.04` → `.venv` → `python -u` を呼び出します。
共通ランチャは `bat/wsl_python.bat` / `bat/run_wsl.sh` です。venv の場所を変える場合は環境変数 `WSL_VENV` に WSL 絶対パスを指定してください。

主な依存: `torch`, `numpy`, `cshogi`

Docker を使う場合は `settings/Dockerfile` を参照してください。

## ディレクトリ構成

```
.
├── main.py                 # usi.py への互換ラッパ
├── usi.py                  # 将棋GUI向け USI エンジン入口
├── requirements.txt
├── bat/
│   ├── wsl_python.bat      # WSL+venv 共通ランチャ
│   ├── run_wsl.sh          # venv activate + python -u
│   ├── usi.bat             # 将棋GUI登録用 USI 起動
│   ├── start_engine.bat    # 大会向けパラメータ例付き USI
│   ├── train.bat           # 教師あり学習
│   └── match.bat           # エンジン同士の対局
├── game/
│   └── board.py            # 447bit 特徴量・指し手ラベル
├── model/
│   ├── model.py            # HybridAlphaZeroNet / create_model
│   ├── cnn_block.py        # CNN 系ブロック
│   ├── gnn_block.py        # GNN 系ブロック
│   ├── transformer_block.py
│   ├── others_block.py     # 軽量・特殊ブロック
│   └── model_README.md
├── search/
│   ├── mcts.py             # モンテカルロ木探索
│   ├── npls.py             # UW-NPLS（推奨）
│   ├── mc_rzf.py           # MC-RZF（実験）
│   └── spsa.py             # 探索ハイパーパラメータ最適化
├── play/
│   └── match.py            # エンジン同士の対局（学習とは独立）
├── train/
│   ├── dataloader.py       # HCPE/PSV → 447bit バッチ
│   ├── supervised_train.py
│   ├── reinforcement_train.py
│   ├── parallel_train.py
│   └── train_README.md
├── train_data/             # 学習データ（ジャンクション可）
├── settings/
│   ├── Dockerfile
│   └── requirements.txt
└── weights/                # 学習済み重み（要配置）
```

## USI エンジンの起動（将棋所 / ShogiGUI）

GUI には **`bat/usi.bat`** をエンジンとして登録してください（WSL + venv 経由）。

対応コマンド: `usi` / `isready` / `setoption` / `usinewgame` / `position` / `go`（`movetime` 含む） / `stop` / `ponderhit` / `gameover` / `quit`

プロジェクトルートで直接動かす場合:

```bash
python usi.py --search_type npls --mode modelC --weights weights/checkpoint.pth
```

Windows では次の bat も利用できます。

- `bat/usi.bat` … **将棋GUI登録用（推奨）**
- `bat/start_engine.bat` … 大会向けパラメータ例付き USI 起動
- `bat/train.bat` … 教師あり学習
- `bat/train_continue.bat` … 既存重みから継続学習
- `bat/match.bat` … エンジン同士の対局（`play/match.py`）

### 主な引数

| 引数 | 既定 | 説明 |
|------|------|------|
| `--search_type` | `npls` | `mcts` / `npls` / `mc_rzf` |
| `--mode` | `modelC` | モデル構成（下記） |
| `--weights` | `weights/checkpoint.pth` | 重みファイル |
| `--name` | `GNN-Shogi` | USI エンジン名 |
| `--time_rate` | `0.5` | 持ち時間使用率（0.01〜1.0） |
| `--draw_value` | `0` | 千日手評価値（正=歓迎 / 負=回避） |
| `--impasse_score` | `28` | 入玉宣言に必要な最低点数 |
| `--impasse_value` | `0` | 持将棋局面の評価値 |
| `--max_moves` | `320` | 最大手数（大会規定に合わせる） |
| `--use_search` | `1` | `1`=探索あり / `0`=評価関数のみ |

GUI 側の `setoption` でも TimeRate / DrawValue / ImpasseScore などを変更できます。

### モデルモード (`--mode`)

| mode | 概要 |
|------|------|
| `modelA` | ResBlock ×10（軽量ベースライン） |
| `modelB` | ResBlock + SE |
| `modelC` | 多様な CNN ブロック ×30（既定） |
| `modelD` | MobileNetV3 + SlotAttention |
| `modelE` | InceptionV4 ×30 |
| `modelF` | ResBlock + SGC 交互 |
| `modelG` | CNN → GCN → GIN |
| `modelH` | CNN + Graph + LocalAttention |
| `fastA` | MLPMixer + SE（CPU 向け軽量） |

詳細は `model/model_README.md` を参照してください。

## 探索

- **MCTS** — PUCT ベースの標準木探索
- **NPLS (UW-NPLS)** — Policy / Value / 不確実性を組み合わせた優先度付き探索（既定・推奨）
- **MC-RZF** — 短いロールアウトの実験実装

NPLS の重みなどは `search/spsa.py` の SPSA で調整できます。

## 学習

詳細は `train/train_README.md` を参照。`train_data/` に HCPE / PSV を置く（またはジャンクションする）。

```bash
# 教師あり（HCPE）
python train/supervised_train.py \
  --train_data train_data/2024_train.hcpe \
  --test_data train_data/2024_test.hcpe \
  --epochs 1 --batch_size 256

# 動作確認用に件数制限
python train/supervised_train.py --limit 4096

python train/reinforcement_train.py   # 自己対局強化学習
python train/parallel_train.py        # 並列自己対局
```

学習結果の重みは `weights/` に保存する想定です。対局前に `weights/checkpoint.pth` などを配置してください。

## 447bit 盤面表現

| 領域 | ビット数 | 内容 |
|------|----------|------|
| 盤面 | 405 | 81マス × 5bit（手番側視点の駒種） |
| 持ち駒 | 42 | 自分 21 + 相手 21 |
| **合計** | **447** | `game/board.py` で生成 |

指し手ラベル数は 2187（移動方向・打ち駒 × 81マス）です。
