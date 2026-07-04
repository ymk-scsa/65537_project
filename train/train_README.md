# 学習スクリプトの説明

本ディレクトリには、将棋AIを学習させるための各種スクリプトが含まれています。
すべてのスクリプトは 447bit の軽量盤面表現およびベクトル埋め込みモデルに対応しています。

## 各スクリプトの役割

1. **supervised_train.py**
   - **概要**: 教師あり学習用のスクリプトです。棋譜や局面データから447bit表現の特徴量、指し手ラベル、勝敗結果を学習します。
   - **実行方法**:
     ```bash
     python train/supervised_train.py
     ```

2. **reinforcement_train.py**
   - **概要**: 自己対局を用いた強化学習（RL）スクリプトです。最新のモデルどうしで自己対局を行い、UW-NPLS探索アルゴリズムを用いて評価データを収集し、モデルを繰り返し学習します。
   - **実行方法**:
     ```bash
     python train/reinforcement_train.py
     ```

3. **parallel_train.py**
   - **概要**: `ProcessPoolExecutor` を用いて、複数の自己対局プロセスを並列（例えば8プロセス）に実行し、データ収集の効率を高めた強化学習スクリプトです。
   - **実行方法**:
     ```bash
     python train/parallel_train.py
     ```
