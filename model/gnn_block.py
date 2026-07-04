import torch
import torch.nn as nn
import torch.nn.functional as F

#1. GDS = DynamicGNNBlock (動的グラフ構築)
# 自動グラフ構築型GNNブロック (Vision GNN Style)　外部からの隣接行列を必要とせず、テンソルから動的にグラフを生成する
class DynamicGNNBlock(nn.Module): #動的グラフニューラルネットワーク（Dynamic GNN）という手法を用いたネットワーク層を定義
    def __init__(self, channels, k=9): #このブロックを初期化する関数。各マスが持つ情報の深さ（ベクトルサイズ）をchannelsで指定。kは各マスが接続する近傍ノード数
        super(DynamicGNNBlock, self).__init__() #PyTorchのベースクラス（nn.Module）を初期化し、このクラスをPyTorchのモジュールとして扱えるようにする
        self.k = k  # 各マスが接続する近傍ノード数 引数で受け取った k の値を、クラス内の変数として保存

        # 特徴変換用の層
        self.fc_in = nn.Sequential( ## 入力データを変換するためのシーケンシャル（一連の処理）を定義
            nn.Conv2d(channels, channels, 1), # 1x1の畳み込み層：マスの位置関係は変えず、各マスの「情報の質」を線形変換する
            nn.BatchNorm2d(channels), ## バッチ正規化：学習を安定させるために、データの分布を平均0、分散1に近い状態に整える
            nn.GELU() # GELU活性化関数：ReLUをより滑らかにした関数で、近年の高性能なAIモデル（BERT等）でよく使われる
        )
        self.fc_out = nn.Sequential( # グラフ処理を終えた後のデータを最終調整するためのシーシャルを定義
            nn.Conv2d(channels, channels, 1), # 1x1の畳み込み層：集約された近傍ノードの情報と自分自身を混ぜ合わせ、最終的な特徴量を抽出する
            nn.BatchNorm2d(channels) # バッチ正規化：出力前のデータの偏りを取り除き、次のレイヤーへ渡しやすくする
        )

        # メッセージパッシング用の重み
        self.nn = nn.Sequential( # 2つのノード間の「関係性」を学習するための小さなニューラルネットワークを定義
            nn.Linear(channels * 2, channels), # 全結合層：自分と相手の情報を合わせた「2倍のチャンネル数」を入力し、元のサイズに圧縮する
            nn.LeakyReLU(0.2), ## LeakyReLU活性化関数：入力がマイナスのときもわずかに値を通す（0.2倍）ことで、学習が止まるのを防ぐ
            nn.Linear(channels, channels) # 全結合層：最終的な「関係性の深さ（エッジの特徴）」を出力する
        )

    def forward(self, x):
        # x: (B, C, 9, 9) 
        B, C, H, W = x.shape # 入力の形状を取得 (B:バッチサイズ, C:チャンネル, H:高さ9, W:幅9)
        residual = x # あとで足し戻すために元の入力を保存（スキップ接続用）

        # 1. 各マスをノードとして展開 (B, 81, C)
        x_in = self.fc_in(x) # 前処理用の1x1畳み込みを通し、各マスの特徴を洗練させる
        features = x_in.view(B, C, -1).permute(0, 2, 1) # (B, 81, C)

        # 2. 動的グラフ構築 (k-Nearest Neighbors)
        # 各ノード間の距離を計算
        dist = torch.cdist(features, features) # (B, 81, 81)
        # 距離が近い上位k個のインデックスを取得  _ は距離の値そのもの（今回は不要）、nn_idx はそのマスの番号（インデックス）
        _, nn_idx = torch.topk(dist, self.k, dim=-1, largest=False) # (B, 81, k)

        # 3. メッセージパッシング
        # (B, 81, k, C) の形に周辺ノードの特徴を並べる
        batch_idx = torch.arange(B, device=x.device).view(-1, 1, 1) # バッチごとのインデックス
        node_idx = torch.arange(81, device=x.device).view(1, -1, 1) # 各マス(0-80)のインデックス
        neighbor_features = features[batch_idx, nn_idx] # (B, 81, k, C)

        # 中心ノードの特徴を拡張して結合
        center_features = features.unsqueeze(2).repeat(1, 1, self.k, 1) # (B, 81, k, C)
        edge_features = torch.cat([center_features, neighbor_features - center_features], dim=-1) # neighbor_features - center_features は自分から見た相手との違いを意味する

        # メッセージの集約 (Max Pooling)
        messages = self.nn(edge_features) # (B, 81, k, C)
        new_features, _ = torch.max(messages, dim=2) # (B, 81, C)

        # 4. 元のテンソル形状に戻す
        out = new_features.permute(0, 2, 1).view(B, C, H, W) # 処理のために並べ替えていた軸を (B, C, 81) に戻し、さらに (B, C, 9, 9) の盤面形状に復元する
        out = self.fc_out(out) # 後処理用の1x1畳み込みを通し、最終的な特徴を調整する

        return F.relu(out + residual) # 最初の入力を足し合わせ（残差接続）、ReLUで活性化して出力する
    

#2. GDG = DynamicGraphBlock（EdgeConv系・shallow）
# 動的にグラフを構築し、浅いネットワークで効果的に特徴を抽出するGNNブロック
class DynamicGraphBlock(nn.Module): # 動的なグラフ構造を用いて盤面の特徴を抽出するブロック
    def __init__(self, channels, k=9): # 自分に関連の深い上位k個（9個）のマスを選択する設定
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential( # エッジ（マスの繋がり）の情報を計算するための多層パーセプトロン(MLP)
            nn.Linear(channels * 2, channels), # 自分と相手の情報を結合した(channels * 2)を入力し、圧縮する
            nn.LeakyReLU(0.2), # 負の値をわずかに通すことで学習の停滞を防ぐ活性化関数
            nn.Linear(channels, channels) # 最終的なエッジの特徴量を出力する
        )
        self.bn = nn.BatchNorm2d(channels) # バッチ正規化：計算結果を整えて学習効率を上げる

    def forward(self, x):
        B, C, H, W = x.shape # 入力の形状を取得 (B=バッチ, C=チャンネル, H=9, W=9)
        r = x # 残差接続（後で足すため）に元のデータを保存

        feats = x.view(B, C, -1).permute(0, 2, 1)  # (B,81,C) 盤面の9x9を81個の独立したノードに並べ替える
        dist = torch.cdist(feats, feats) # 全てのマス同士の「特徴の距離」を計算する
        _, idx = torch.topk(dist, self.k, largest=False) # 距離が近い（＝今の戦況で関係が深い）上位k個のインデックスを取得

        batch = torch.arange(B, device=x.device).view(B,1,1) # 各バッチの計算用インデックスを作成
        neighbors = feats[batch, idx]  # (B,81,k,C)
        center = feats.unsqueeze(2).expand_as(neighbors) # 自分自身の情報を、近傍ノードと同じ数だけコピーして並べる

        edge = torch.cat([center, neighbors - center], dim=-1) # 「自分」と「自分と相手の差分」を合体させて、エッジの特徴を作る
        msg = self.edge_mlp(edge).max(dim=2)[0] # エッジ情報をMLPで計算し、k個の中から最大の値（最も強い特徴）を抽出する(Max Pooling)

        out = msg.permute(0,2,1).view(B,C,H,W) #列に並んだデータを元の (B, C, 9, 9) の形状に戻す
        out = self.bn(out) # バッチ正規化を適用
        return F.relu(out + r) # 元の入力(r)を足し合わせ（スキップ接続）、ReLUで仕上げて出力

#3. GCN = GCNBlock (DeepSets / Global Context Injection)
# 事前定義された近傍グラフに基づいて情報を集約するGNNブロック
class GCNBlock(nn.Module): # グラフ畳み込み（GCN）の概念を簡略化した軽量な情報共有ブロック
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(channels, channels) # 盤面全体の平均情報を、各マスに適合するように調整するための全結合層

    def forward(self, x):
        B, C, H, W = x.shape # 入力の形状（バッチ, チャンネル, 高さ9, 幅9）を取得
        r = x # 残差接続（あとで足し合わせる）ために元のデータを保存

        # 盤面の9x9を81個の「ノード（頂点）」として1列に展開
        tokens = x.view(B, C, -1).permute(0,2,1) # 形状を (B, 81, C) に変換する
        # 1. 盤面全体の要約（平均）を計算
        # 81マスすべての情報を平均し、現在の局面の「全体的な特徴」を1つのベクトルにする
        mean = tokens.mean(dim=1, keepdim=True) # 形状: (B, 1, C)
        tokens = tokens + self.linear(mean) # これにより、個々の駒が「盤面全体のムード」を考慮した状態に更新される

        out = tokens.permute(0,2,1).view(B,C,H,W) # 処理のために並べ替えていた軸を戻し、(B, C, 9, 9) の盤面形状に復元
        return F.relu(out + r) # 元の入力(r)を足し、ReLUで活性化して出力
    
#4. GSG = GraphSAGE ブロック
# 近傍の情報を集約（今回はMean集約）し、自分自身の情報と結合して更新する
class GraphSAGEBlock(nn.Module):
    def __init__(self, channels, k=9):
        super().__init__()
        self.k = k
        self.proj_neighbor = nn.Linear(channels, channels)
        self.proj_self = nn.Linear(channels, channels)
        self.combine = nn.Linear(channels * 2, channels)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        tokens = x.view(B, C, -1).permute(0, 2, 1) # (B, 81, C)

        # 動的グラフ構築（距離ベースで近傍k個を選択）
        dist = torch.cdist(tokens, tokens)
        _, idx = torch.topk(dist, self.k, largest=False)
        
        batch = torch.arange(B, device=x.device).view(B, 1, 1)
        neighbors = tokens[batch, idx] # (B, 81, k, C)

        # 近傍の平均を集約
        neighbor_mean = neighbors.mean(dim=2) # (B, 81, C)
        
        # 自分自身と結合 (SAGEの特徴)
        combined = torch.cat([self.proj_self(tokens), self.proj_neighbor(neighbor_mean)], dim=-1)
        out = self.combine(combined) # (B, 81, C)

        out = out.permute(0, 2, 1).view(B, C, H, W)
        return F.relu(self.bn(out) + r)

#5. GIN = Graph Isomorphism Network ブロック
# グラフの同型性判定において最強の理論的表現力を持つ構造
class GINBlock(nn.Module):
    def __init__(self, channels, k=9):
        super().__init__()
        self.k = k
        self.eps = nn.Parameter(torch.zeros(1))
        # GINの核：集約後に強力なMLPで変換する
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        tokens = x.view(B, C, -1).permute(0, 2, 1)

        dist = torch.cdist(tokens, tokens)
        _, idx = torch.topk(dist, self.k, largest=False)
        
        batch = torch.arange(B, device=x.device).view(B, 1, 1)
        neighbors = tokens[batch, idx]

        # GINの公式: (1 + eps) * self + sum(neighbors)
        neighbor_sum = neighbors.sum(dim=2)
        out = self.mlp((1 + self.eps) * tokens + neighbor_sum)

        out = out.permute(0, 2, 1).view(B, C, H, W)
        return F.relu(self.bn(out) + r)

#6. G2I = GCNII ブロック
# 深い層でも「初期状態の残差(Initial Residual)」を混ぜることで過平滑化を防ぐ
class GCNIIBlock(nn.Module):
    def __init__(self, channels, alpha=0.1, theta=0.5, layer_idx=1, k=9):
        super().__init__()
        self.alpha = alpha
        self.beta = theta / layer_idx # 深くなるほど調整を小さくする
        self.k = k
        self.weight = nn.Linear(channels, channels)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x, x_0=None):
        # x_0 はネットワークの最初の層の出力（入力チャンネルから変換直後のもの）
        if x_0 is None: x_0 = x
        
        B, C, H, W = x.shape
        r = x
        tokens = x.view(B, C, -1).permute(0, 2, 1)
        tokens_0 = x_0.view(B, C, -1).permute(0, 2, 1)

        dist = torch.cdist(tokens, tokens)
        _, idx = torch.topk(dist, self.k, largest=False)
        
        batch = torch.arange(B, device=x.device).view(B, 1, 1)
        neighbor_mean = tokens[batch, idx].mean(dim=2)

        # Initial Residual (alpha) と Identity Mapping (beta)
        h = (1 - self.alpha) * neighbor_mean + self.alpha * tokens_0
        out = (1 - self.beta) * h + self.beta * self.weight(h)

        out = out.permute(0, 2, 1).view(B, C, H, W)
        return F.relu(self.bn(out) + r)

#7. GSG = Simple Graph Convolution ブロック
# 活性化関数を介さず、近傍情報の平滑化のみを高速に行う
class SGCBlock(nn.Module):
    def __init__(self, channels, k=9):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(channels, channels)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        tokens = x.view(B, C, -1).permute(0, 2, 1)

        dist = torch.cdist(tokens, tokens)
        _, idx = torch.topk(dist, self.k, largest=False)
        
        batch = torch.arange(B, device=x.device).view(B, 1, 1)
        # SGCは単なる情報の拡散（平均化）に特化
        out = tokens[batch, idx].mean(dim=2)
        out = self.linear(out)

        out = out.permute(0, 2, 1).view(B, C, H, W)
        return F.relu(self.bn(out) + r)

#8. GA2 = GATv2 ブロック
# 従来のアテンションよりも「動的な注目」が可能な改良版GAT
class GATv2Block(nn.Module):
    def __init__(self, channels, heads=4, k=9):
        super().__init__()
        self.k = k
        self.heads = heads
        d_k = channels // heads
        
        self.w_q = nn.Linear(channels, channels)
        self.w_k = nn.Linear(channels, channels)
        self.w_v = nn.Linear(channels, channels)
        # GATv2: QueryとKeyを足してから非線形変換し、Attentionスコアを出す
        self.a = nn.Parameter(torch.randn(heads, d_k))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        tokens = x.view(B, C, -1).permute(0, 2, 1) # (B, 81, C)

        dist = torch.cdist(tokens, tokens)
        _, idx = torch.topk(dist, self.k, largest=False)
        
        batch = torch.arange(B, device=x.device).view(B, 1, 1)
        neighbor_feats = tokens[batch, idx] # (B, 81, k, C)
        
        # アテンションスコア計算
        q = self.w_q(tokens).unsqueeze(2) # (B, 81, 1, C)
        k = self.w_k(neighbor_feats)      # (B, 81, k, C)
        
        # GATv2の特徴: スコア = a * leaky_relu(Wq + Wk)
        combined = F.leaky_relu(q + k, 0.2)
        # (B, 81, k, heads, d_k) に変形してアテンション
        combined = combined.view(B, 81, self.k, self.heads, -1)
        scores = (combined * self.a).sum(dim=-1) # (B, 81, k, heads)
        attn = F.softmax(scores, dim=2).unsqueeze(-1) # (B, 81, k, heads, 1)
        
        v = self.w_v(neighbor_feats).view(B, 81, self.k, self.heads, -1)
        out = (attn * v).sum(dim=2).reshape(B, 81, C)

        out = out.permute(0, 2, 1).view(B, C, H, W)
        return F.relu(self.bn(out) + r)