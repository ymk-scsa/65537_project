import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. CRE = Residual ブロック(ノーマルResNet)
# 盤面の局所的なパターン（3x3）を抽出する
class ResBlock(nn.Module): #ResBlockという名前の新しいネットワークのクラスを定義
    def __init__(self, channels): #初期化メソッド、channelsは入力チャネル数
        super(ResBlock, self).__init__() #PyTorchの基本クラスを初期化
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1) #3x3の畳み込み層を定義。padding=1 を指定することで、画像のサイズ（盤面のサイズ）を変えずに処理。
        self.bn1 = nn.BatchNorm2d(channels) #バッチ正規化層を定義
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1) #もう一つ3x3の畳み込み層を定義
        self.bn2 = nn.BatchNorm2d(channels) #もう一つバッチ正規化層を定義

    def forward(self, x): #順伝播の定義
        residual = x #入力をresidualとして保存
        out = F.relu(self.bn1(self.conv1(x))) #1つ目の畳み込み、バッチ正規化、ReLU活性化関数を適用
        out = self.bn2(self.conv2(out)) #2つ目の畳み込みとバッチ正規化を適用
        out += residual #residual接続を追加。加工したデータ out に、最初に保存しておいた「生の入力 residual」を足し合わせる。
        return F.relu(out) #最後にReLUで仕上げて、結果を返す。
    
# 2. CDN = DenseNet ブロック
class DenseBlock(nn.Module):
    def __init__(self, channels, growth_rate=32):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1)
        # DenseNetは入力を「捨てる」のではなく「後ろに足していく(Concat)」
        self.fc = nn.Conv2d(channels + growth_rate, channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat([x, out], dim=1) # チャンネル方向に結合
        return self.fc(out)
    
# 3. CNX = ResNeXt ブロック
class ResNeXtBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        # チャンネルをグループに分けて並列処理することで表現力を高める
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        r = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + r)

# 4. CNT = ConvNeXt ブロック
class ConvNeXtBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Transformerの思想：大きなカーネル、LayerNorm、GELUを採用
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm([channels, 9, 9])
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * channels, channels)

    def forward(self, x):
        r = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)
        return F.relu(x + r)

# 5. CXC = Xception (Depthwise Separable Conv) ブロック
class XceptionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # DeepとPointwiseを分離して計算効率を最大化
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))) + x)

# 6. CGL = GoogLeNet (Inception) ブロック
class InceptionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 1x1, 3x3, 5x5 の異なる視点を同時に持つ
        c = channels // 4
        self.b1 = nn.Conv2d(channels, c, kernel_size=1)
        self.b2 = nn.Sequential(nn.Conv2d(channels, c, 1), nn.Conv2d(c, c, 3, padding=1))
        self.b3 = nn.Sequential(nn.Conv2d(channels, c, 1), nn.Conv2d(c, c, 5, padding=2))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(channels, c, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

# 7. CI3 = Inception-v3 ブロック
class InceptionV3Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 3x3を1x3と3x1に分解して軽量化・高性能化
        c = channels // 2
        self.conv1 = nn.Conv2d(channels, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, c, 1),
            nn.Conv2d(c, c, (1, 3), padding=(0, 1)),
            nn.Conv2d(c, c, (3, 1), padding=(1, 0))
        )

    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x)], dim=1)

# 8. CI4 = Inception-v4 ブロック
class InceptionV4Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # より深く、並列パスを増やした構造
        c = channels // 4
        self.path1 = nn.Conv2d(channels, c, 1)
        self.path2 = nn.Sequential(nn.Conv2d(channels, c, 1), nn.Conv2d(c, c, 3, padding=1))
        self.path3 = nn.Sequential(nn.Conv2d(channels, c, 1), nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1))
        self.path4 = nn.Sequential(nn.AvgPool2d(3, 1, 1), nn.Conv2d(channels, c, 1))

    def forward(self, x):
        return torch.cat([self.path1(x), self.path2(x), self.path3(x), self.path4(x)], dim=1)

# 9. CIR = Inception-ResNet ブロック
class InceptionResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Inceptionの複雑さとResNetの安定性を融合
        c = channels // 2
        self.path1 = nn.Conv2d(channels, c, 1)
        self.path2 = nn.Sequential(nn.Conv2d(channels, c, 1), nn.Conv2d(c, c, 3, padding=1))
        self.reduction = nn.Conv2d(channels, channels, 1) # 結合後の次元調整

    def forward(self, x):
        r = x
        out = torch.cat([self.path1(x), self.path2(x)], dim=1)
        out = self.reduction(out)
        return F.relu(out + r)