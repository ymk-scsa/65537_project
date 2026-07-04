import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """
    MobileNetV3Block・EfficientNetBlock 内部で使うヘルパークラス。
    チャンネルをいったん圧縮（Squeeze）してから
    各チャンネルの重要度を学習し、元に戻す（Excitation）。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)          # 9×9 → 1×1 に平均圧縮
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),  # 256 → 16
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),  # 16 → 256
            nn.Sigmoid()                                  # 0〜1 の重みに変換
        )

    def forward(self, x):
        B, C = x.shape[:2]
        w = self.pool(x).view(B, C)          # (B, C)
        w = self.fc(w).view(B, C, 1, 1)      # (B, C, 1, 1)
        return x * w                          # チャンネルごとに重みを掛ける

#1. OST = SetBlock (Set Transformer / Deep Sets ブロック# 元の入力(r)を足し（スキップ接続）、ReLUで活性化して次の層へ渡す)
# 集合データを扱うためのTransformerベースのブロック
class SetBlock(nn.Module): # 盤面全体の「集合的（Set）」な情報を抽出し、各マスに共有するブロック
    def __init__(self, channels):
        super().__init__()
        self.phi = nn.Sequential( # 各マスの情報を「全体としての特徴」に変換するためのネットワーク
            nn.Linear(channels, channels),
            nn.ReLU()
        )
        self.rho = nn.Sequential( # 全体の特徴（平均）を、再び各マスの情報へ統合するためのネットワーク
            nn.Linear(channels, channels),
            nn.ReLU()
        )

    def forward(self, x): 
        B, C, H, W = x.shape # 入力の形状（バッチ, チャンネル, 高さ9, 幅9）を取得
        r = x # あとで足し合わせる（残差接続）ために元のデータを保存

        tokens = x.view(B, C, -1).permute(0,2,1)  # (B,81,C) 盤面を81個の「要素（トークン）」として1列に展開
        # 1. 盤面全体の情報を集約
        # 各マスの情報をphiで加工してから、全81マスの「平均（mean）」を計算する
        # これにより、盤面全体の「厚み」や「戦況の要約」が1つのベクトルに凝縮される
        pooled = self.phi(tokens).mean(dim=1, keepdim=True) # 形状: (B, 1, C)
        # 2. 全体情報を各マスに配分
        # 得られた全体の要約（pooled）をrhoで加工し、元の各マス（tokens）に足し合わせる
        # これにより、各マスの駒の情報に「盤面全体の状況」という追加情報が加わる
        tokens = tokens + self.rho(pooled) 

        out = tokens.permute(0,2,1).view(B,C,H,W) # 処理のために並べ替えていた軸を戻し、(B, C, 9, 9) の形状に復元
        return F.relu(out + r) # 元の入力(r)を足し、ReLUで活性化して出力

#2. OSB = SlotBlock (Object-centric / Slot Attention（軽量版）ブロック)
# オブジェクト中心の注意機構を用いて、重要な特徴を抽出するブロック
class SlotBlock(nn.Module): # 特定の「役割（スロット）」に情報を集約して盤面を解析するブロック
    def __init__(self, channels, slots=16): # 16個の「情報の受け皿（スロット）」を学習可能なパラメータとして定義
        super().__init__()
        self.slots = nn.Parameter( # 初期値は小さな乱数(0.02倍)で設定し、学習を通じて「注目すべきパターン」を覚える
            torch.randn(slots, channels)*0.02 #係数は修正の余地あり
        )
        self.attn = nn.MultiheadAttention(channels, 4, batch_first=True) # スロットが盤面のどのマスに注目するかを計算するためのアテンション機構（4ヘッド）

    def forward(self, x):
        B, C, H, W = x.shape # 入力の形状（バッチ, チャンネル, 高さ9, 幅9）を取得
        r = x # あとで足し合わせる（残差接続）ために元のデータを保存

        # 盤面を81個の「トークン（マスの情報）」に展開
        tokens = x.view(B, C, -1).permute(0,2,1) # 形状: (B, 81, C)
        slots = self.slots.unsqueeze(0).expand(B, -1, -1) # 形状: (B, 16, C)

        # アテンション実行：スロット(Q)が盤面の各マス(K, V)から情報を吸い上げる
        # これにより、16個のスロットそれぞれが「王様の周り」「攻めの拠点」などの特徴を掴む
        slot_out, _ = self.attn(slots, tokens, tokens)
        pooled = slot_out.mean(dim=1, keepdim=True) # 16個のスロットが得た情報を平均し、盤面全体の「要約」を1つ作る 形状: (B, 1, C)

        tokens = tokens + pooled # 抽出された重要な要約情報を、元の81マスの情報に足し合わせる
        out = tokens.permute(0,2,1).view(B,C,H,W) # 1列に並んだデータを元の (B, C, 9, 9) の盤面形状に戻す

        return F.relu(out + r) # 元の入力(r)を足し、ReLUで活性化して出力
    
#3. OM1 = MobileNetV1Block (Depthwise Separable Conv)
class MobileNetV1Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))) + x)

#4. OM2 = MobileNetV2Block (Inverted Residual)
class MobileNetV2Block(nn.Module):
    def __init__(self, channels, expand_ratio=6):
        super().__init__()
        hidden_dim = channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1), nn.BatchNorm2d(hidden_dim), nn.ReLU6(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim), nn.BatchNorm2d(hidden_dim), nn.ReLU6(),
            nn.Conv2d(hidden_dim, channels, 1), nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.conv(x)

#5. OM3 = MobileNetV3Block (h-swish + SE)
class MobileNetV3Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels*2, 1), nn.BatchNorm2d(channels*2), nn.Hardswish(),
            nn.Conv2d(channels*2, channels, 1), nn.BatchNorm2d(channels)
        )
        self.se = SqueezeExcitation(channels)

    def forward(self, x):
        return x + self.se(self.conv(x))

#6. OS1 = ShuffleNetV1Block (Channel Shuffle + Depthwise Separable Conv)
class ShuffleNetV1Block(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=groups), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels), nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # Channel Shuffle処理
        B, C, H, W = x.shape
        x = x.view(B, self.groups, C // self.groups, H, W).transpose(1, 2).reshape(B, C, H, W)
        return F.relu(x + self.conv(x))

#7. OS2 = ShuffleNetV2Block (ShuffleNet v2)
class ShuffleNetV2Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c = channels // 2
        self.branch = nn.Sequential(
            nn.Conv2d(self.c, self.c, 1), nn.BatchNorm2d(self.c), nn.ReLU(),
            nn.Conv2d(self.c, self.c, 3, padding=1, groups=self.c), nn.BatchNorm2d(self.c),
            nn.Conv2d(self.c, self.c, 1), nn.BatchNorm2d(self.c), nn.ReLU()
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.cat([x1, self.branch(x2)], dim=1)

#8. OSQ = SqueezeNetBlock (Fire Module)
class SqueezeNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        s = channels // 8
        self.squeeze = nn.Conv2d(channels, s, 1)
        self.expand1x1 = nn.Conv2d(s, channels//2, 1)
        self.expand3x3 = nn.Conv2d(s, channels//2, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.squeeze(x))
        return torch.cat([F.relu(self.expand1x1(out)), F.relu(self.expand3x3(out))], dim=1)

#9. OMN = MNasNetBlock (MNasNet ブロック)
class MNasNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 探索された効率的なカーネルサイズ5x5を採用
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

#10. OEF = EfficientNetBlock (MBConv + Swish)
class EfficientNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels*4, 1), nn.BatchNorm2d(channels*4), nn.SiLU(),
            nn.Conv2d(channels*4, channels*4, 3, padding=1, groups=channels*4), nn.BatchNorm2d(channels*4), nn.SiLU(),
            nn.Conv2d(channels*4, channels, 1), nn.BatchNorm2d(channels)
        )
        self.se = SqueezeExcitation(channels)

    def forward(self, x):
        return x + self.se(self.conv(x))

#11. OCB = CapsuleBlock (Simplified Layer)
class CapsuleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        # 将棋の「駒の向き」や「存在」をベクトルの長さで表現するSquash関数
    def squash(self, x):
        norm = torch.norm(x, dim=1, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (x / (norm + 1e-8))

    def forward(self, x):
        return self.squash(self.conv(x)) + x

#12. OMM = MLPMixerBlock
class MLPMixerBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.token_mix = nn.Linear(81, 81) # 空間方向の混合
        self.norm2 = nn.LayerNorm(channels)
        self.channel_mix = nn.Linear(channels, channels) # チャンネル方向の混合

    def forward(self, x):
        B, C, H, W = x.shape
        y = x.view(B, C, -1) # (B, C, 81)
        # Token Mixing
        y = y + self.token_mix(self.norm1(y.transpose(1, 2)).transpose(1, 2))
        # Channel Mixing
        y = y + self.channel_mix(self.norm2(y.transpose(1, 2))).transpose(1, 2)
        return y.view(B, C, H, W)

#13. OSE = SqueezeExcitationBlock (Squeeze-and-Excitation)
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        y = x.view(B, C, -1).mean(dim=2)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y

#14. OUN = UNetBlock (Encoder-Decoder)
class UNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 9x9を解像度維持したままボトルネックを作る
        self.down = nn.Conv2d(channels, channels*2, 3, stride=1, padding=1)
        self.up = nn.Conv2d(channels*2, channels, 1)

    def forward(self, x):
        # 簡易的なSkip-connection構造
        return F.relu(self.up(F.relu(self.down(x)))) + x