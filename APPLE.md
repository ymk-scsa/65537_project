# Apple 版（apple_branch）の追加内容

`apple_branch` で追加した、次の 4 つについてまとめます。

1. 新しいニューラルネットワーク **RayGraph ハイブリッド**（`model/apple_net.py`）
2. **先手・後手で評価を変える**しくみ（共通の胴体 + 手番別の頭）
3. 探索の **C++ 版**（`search/cpp/`）。Python 版はそのまま残してあります。
4. **高速化・軽量化**（CPU だけで動かす前提）

---

## 1. RayGraph ハイブリッド（`appleXS` / `appleS` / `appleM`）

これまでのモデルは「CNN・GNN・Transformer のブロックを順番に積む」形でした。
RayGraph では、**将棋の駒の動き方そのものを、ネットワークの形として組み込みました。**

| 部品 | 中身 | 将棋での意味 |
|---|---|---|
| Local（CNN） | 3x3 の深さ方向畳み込み + 点ごとの MLP | 金・銀・玉など「近くの形」 |
| **RayCast**（新しく考えた部分） | 8 方向の「光線」に沿った伝達。マスごとに**透明度**を学習し、駒があると光がさえぎられる | 飛・角・香など**走り駒の利き**。駒にぶつかると利きが止まることを、微分できる形で表す |
| **MoveGraph**（GNN） | 玉の 8 近傍 + 桂の跳びでつながるグラフ。送り元の駒の種類からゲートを学習 | 「この駒はこの方向に利きがあるか」を学ぶ**利きグラフ** |
| Global（Transformer） | 81 マスの自己注意を最後に 1 層 | 盤全体のつながり |

この Local → RayCast → MoveGraph を 1 ブロックとして繰り返し、最後に Global を通します。

RayCast は、線に沿った漸化式 `m(j) = v(j+1) + γ・τ(j+1)・m(j+1)` で計算します。
4 系統（筋・段・斜め・逆斜め）× 前後 2 方向 = 104 本の線をまとめて処理するので、軽く動きます。
総当たりで計算した値と一致することをテストで確かめています（誤差 1e-7 程度）。

| モデル | チャンネル | ブロック | 注意層 | パラメータ数 |
|---|---|---|---|---|
| appleXS | 48 | 3 | なし | 約 0.09M |
| appleS（標準） | 64 | 4 | あり | 約 0.23M |
| appleM | 96 | 6 | あり | 約 0.63M |

参考: 今までの modelC は約 18.4M、fastA は約 6.7M です。

## 2. 先手・後手で評価を変える

- **胴体は共通**です。入力に手番の埋め込みを足しています。
- **方策**は、共通の出力に手番別の補正を足します。
- **価値**は、手番ごとに「勝ち / 引き分け / 負け（WDL）」の 3 つを出します。
- 探索に渡す値は「勝ち点の期待値」から作ります。
  `E = P(勝) + d × P(引き分け)`、`value = 2E − 1`
  `d` は電竜戦のルールに合わせて**先手 0.4・後手 0.6**です（変更できます）。
- 探索の中の千日手・最大手数（512 手）による引き分けも、同じ手番別の値で扱います。

学習（`train/supervised_train.py`）は、apple 系のモデルなら自動で WDL の学習に切り替わります。
HCPE では対局結果をそのまま WDL の目標にし、PSV では評価値から勝ち・負けの確率を作ります。

## 3. 探索の C++ 版

```
search/
├── mcts.py / npls.py / mc_rzf.py   ← Python 版（残してあります）
├── cpp_player.py                   ← C++ 探索 + PyTorch 推論のプレイヤー（usi.py から使う）
└── cpp/
    ├── src/apple_core.hpp          ← 447bit 特徴量・指し手ラベル・終局判定
    ├── src/apple_search.hpp        ← MCTS / NPLS 本体
    ├── src/time_manager.hpp        ← 電竜戦向けの時間の決め方
    ├── src/py_apple_search.cpp     ← Python 用モジュール（pybind11）
    ├── src/engine_main.cpp         ← 全部 C++ の USI エンジン（LibTorch）
    ├── build.sh / build_engine.sh  ← ビルド
    ├── export_torchscript.py       ← .pth → TorchScript(.ts)
    └── third_party/cshogi/         ← 盤面・指し手生成・詰み探索（cshogi の C++ ソース、GPL-3.0）
```

### 2 つの動かし方

| 方式 | 探索 | 推論 | 使い方 |
|---|---|---|---|
| **A（おすすめ）** | C++ | PyTorch（Python） | `usi.py --search_type mcts_cpp`（`bat/usi_apple.bat`） |
| B（試作） | C++ | LibTorch（C++） | `search/cpp/apple_engine`（`bat/apple_engine.bat`） |

### Python 版から変えたところ

- 盤面を毎回 sfen 文字列から作り直すのをやめ、指し手を進める・戻す方式にしました。
- 葉の局面を `batch_size` 個ためてから、NN にまとめて渡します（バーチャルロス方式）。
- 終局判定を足しました: 詰み、千日手（手番別の引き分けの値、連続王手）、最大手数、入玉宣言（27 点法）、葉での 1 手詰め。
- ルートで df-pn の詰み探索をし、入玉宣言できるときは `win` を返します。
- **MCTS の符号を直しました。** Python 版 `mcts.py` は、相手から見た値で手を選んでいました。`mcts.py` 側も 1 行直しています。
- 特徴量と指し手ラベルは、Python 版（`game/board.py`）と**完全に同じ**です。24 万手以上で一致を確認しました。
- NPLS は Python 版の計算方法をそのまま移植しました。最後の手の選び方は「訪問回数だけ」で、価値を使いません。このため、駒をタダで取れる局面でも取らないことがあります（材料の損得だけで評価する簡単なテストで確認）。強さを求めるなら MCTS をおすすめします。

### 電竜戦向けの時間の決め方（`--time_mode denryu`）

電竜戦では、1 手の消費時間は **1 秒未満切り捨て** で、そのあと 2 秒加算されます。
そこで、思考時間を「k 秒 + (1000 − margin) ミリ秒」にそろえます。
たとえば 2.6 秒考えても消費は 2 秒で、すぐ 2 秒戻るので、持ち時間は減りません。

| 状況 | 思考時間 |
|---|---|
| 先手・残り 180 秒 | 6.6 秒 |
| 後手・残り 600 秒 | 17.6 秒 |
| 残り 20 秒 | 2.6 秒（持ち時間が減らない） |

`margin`（既定 400ms）は通信の遅れの分の余裕です。サーバーとの通信で一度テストしてから決めてください。

## 4. 速さ（CPU 2 スレッドでの測定値）

`python play/bench_search.py --seconds 2 --threads 2` で測りました（1 秒あたりに NN で評価できた局面数）。

| 探索 : モデル | 局面/秒 | 元との比較 |
|---|---|---|
| Python NPLS : modelC（元の構成） | 33 | 1 倍 |
| Python NPLS : appleS | 150 | 約 4.5 倍 |
| C++ NPLS : appleS | 1,264 | 約 38 倍 |
| **C++ MCTS : appleS** | **1,346** | **約 41 倍** |
| C++ MCTS : appleXS | 2,484 | 約 75 倍 |
| C++ MCTS : appleM | 638 | 約 19 倍 |

自分の PC（22 スレッド）では、もっと速くなるはずです。`--threads` と `--batch_size` を変えて、いちばん速い組み合わせを探してください。

そのほかの高速化:

- 推論のときは BatchNorm を畳み込みに吸収させています（`fuse_for_inference`）。
- TorchScript にして、バッチの大きさを 2 のべき乗にそろえています。
- 学習データの読み込みも C++ で特徴量を作るようにしました（約 12 倍速く、Python 版と同じ結果）。

---

## 使い方（WSL の Ubuntu で）

```bash
cd <プロジェクトのフォルダ>
source .venv/bin/activate
sudo apt install -y build-essential python3-dev   # 初回だけ
pip install pybind11                              # 初回だけ

# C++ のビルド（Windows なら bat\build_cpp.bat でも可）
bash search/cpp/build.sh            # 方式A 用の Python モジュール
bash search/cpp/build_engine.sh     # 方式B 用の C++ エンジン（使う場合だけ）

# 学習（bat\train_apple.bat）
python train/supervised_train.py --mode appleS --weights weights/apple_s.pth \
  --train_data train_data/2024_train.hcpe --test_data train_data/2024_test.hcpe

# 対局（GUI には bat\usi_apple.bat を登録）
python usi.py --search_type mcts_cpp --mode appleS --weights weights/apple_s.pth \
  --time_mode denryu --max_moves 512 --draw_black 0.4 --draw_white 0.6

# 速さの比較
python play/bench_search.py --seconds 2

# 方式B を使う場合: モデルを書き出して、GUI の ModelPath に指定
python search/cpp/export_torchscript.py --mode appleS --weights weights/apple_s.pth --out weights/apple_s.ts
```

AVX2 のない CPU では、`NO_AVX2=1 bash search/cpp/build.sh` でビルドしてください。

## 注意

- **学習はまだです。** ここにあるのは仕組みと速さの改良です。強さは、学習してから先輩の modelC などと対局させて確かめる必要があります。
- `third_party/cshogi` は GPL-3.0 です。プログラムを外部に配布するときは、GPL の条件に従ってください。
- 新しい PyTorch では `torch.jit` に「非推奨」の警告が出ますが、動作に問題はありません。警告は表示しないようにしてあります。
- 電竜戦は CSA プロトコルでの接続が必須です。USI エンジンを CSA サーバーにつなぐ中継ソフトが別に必要です（まだ対応していません）。
