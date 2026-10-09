# 同梱している cshogi の C++ ソース

`search/cpp` の C++ 探索は、盤面・指し手生成・詰み探索に
[cshogi](https://github.com/TadaoYamaoka/cshogi)（Apery 由来）の C++ ソースを使っています。

- 取得元: https://github.com/TadaoYamaoka/cshogi `src/`（コミット 73f4a0da5f7e）
- 同梱しているもの: 盤面・指し手生成・df-pn 詰み探索に必要なファイルだけ
  （`cshogi.h`・OSL 版 df-pn・定跡関係は除外）
- ライセンス: GPL-3.0（`LICENSE` を参照）

**注意:** GPL のコードを含むため、このリポジトリのプログラムを外部に配布する場合は
GPL-3.0 の条件（ソースコードの公開など）に従う必要があります。
電竜戦の規定（第28条）によるプログラムの保存とは別の話です。
