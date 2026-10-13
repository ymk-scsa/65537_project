# 学習スクリプトの説明

本ディレクトリには、将棋AIを学習させるための各種スクリプトが含まれています。
すべてのスクリプトは 447bit の軽量盤面表現およびベクトル埋め込みモデルに対応しています。

## データ形式

- **HCPE** (`.hcpe`): `cshogi.HuffmanCodedPosAndEval`。floodgate / 自己対局など。
- **PSV** (`.bin`): `cshogi.PackedSfenValue`。水匠系の教師データなど。

ローダーは `train/dataloader.py`（`explore_experiment` 由来、447bit 向け）。
学習データはリポジトリ直下の `train_data/` を参照する想定です。

## 各スクリプトの役割

1. **supervised_train.py**
   - HCPE/PSV から教師あり学習する。
   - Windows: `bat/train.bat` を実行（中身の引数を編集して使う）。
   - 例:
     ```bash
     python train/supervised_train.py \
       --train_data train_data/2024_train.hcpe \
       --test_data train_data/2024_test.hcpe \
       --epochs 1 --batch_size 256
     ```
   - 動作確認だけなら `--limit 4096` を付けると速い。

2. **reinforcement_train.py**
   - 自己対局による強化学習。
   - 実行: `python train/reinforcement_train.py`

3. **parallel_train.py**
   - 自己対局を並列化した強化学習。
   - 実行: `python train/parallel_train.py`
