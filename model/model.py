import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_block import (
    ResBlock, DenseBlock, ResNeXtBlock, ConvNeXtBlock, XceptionBlock,
    InceptionBlock, InceptionV3Block, InceptionV4Block, InceptionResNetBlock,
)
from .gnn_block import (
    DynamicGNNBlock, DynamicGraphBlock, GCNBlock, GraphSAGEBlock,
    GINBlock, GCNIIBlock, SGCBlock, GATv2Block,
)
from .transformer_block import (
    RTGNNBlock, GATBlock, ViTBlock, SwinBlock, DeiTBlock,
    BEiTBlock, MAEBlock, DINOBlock, LocalAttentionBlock,
)
from .others_block import (
    SetBlock, SlotBlock, MobileNetV1Block, MobileNetV2Block, MobileNetV3Block,
    ShuffleNetV1Block, ShuffleNetV2Block, SqueezeNetBlock, MNasNetBlock,
    EfficientNetBlock, CapsuleBlock, MLPMixerBlock, SqueezeExcitationBlock, UNetBlock,
)


class HybridAlphaZeroNet(nn.Module):
    def __init__(self, num_actions, blocks_config):
        """
        447bit 埋め込み専用ネットワーク。

        入力: float32 テンソル shape (B, 447)
          - board_bits : x[:, :405]  81マス × 5bit
          - hand_bits  : x[:, 405:]  持ち駒 42bit (自分21 + 相手21)

        blocks_config: ['CRE', 'GDS', 'CNX', ...] のリスト
        """
        super().__init__()

        # ── 入力埋め込み ──────────────────────────────────────────
        # 盤面: 81マスの 5bit パターン (32 状態) → 256 次元
        self.board_embedding = nn.Embedding(32, 256)
        # 持ち駒: 42bit の連続ベクトル → 256 次元
        self.hand_mlp = nn.Sequential(
            nn.Linear(42, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # ── バックボーンブロック ──────────────────────────────────
        self.block_map = {
            'CRE': lambda: ResBlock(256),
            'CDN': lambda: DenseBlock(256),
            'CNX': lambda: ResNeXtBlock(256),
            'CNT': lambda: ConvNeXtBlock(256),
            'CXC': lambda: XceptionBlock(256),
            'CGL': lambda: InceptionBlock(256),
            'CI3': lambda: InceptionV3Block(256),
            'CI4': lambda: InceptionV4Block(256),
            'CIR': lambda: InceptionResNetBlock(256),
            'GDS': lambda: DynamicGNNBlock(256),
            'GDG': lambda: DynamicGraphBlock(256),
            'GCN': lambda: GCNBlock(256),
            'GSG': lambda: GraphSAGEBlock(256),
            'GIN': lambda: GINBlock(256),
            'G2I': lambda: GCNIIBlock(256),
            'GSC': lambda: SGCBlock(256),
            'GA2': lambda: GATv2Block(256),
            'TRT': lambda: RTGNNBlock(256),
            'TGA': lambda: GATBlock(256),
            'TVT': lambda: ViTBlock(256),
            'TSW': lambda: SwinBlock(256),
            'TDE': lambda: DeiTBlock(256),
            'TBT': lambda: BEiTBlock(256),
            'TMA': lambda: MAEBlock(256),
            'TDN': lambda: DINOBlock(256),
            'TLA': lambda: LocalAttentionBlock(256),
            'OST': lambda: SetBlock(256),
            'OSB': lambda: SlotBlock(256),
            'OM1': lambda: MobileNetV1Block(256),
            'OM2': lambda: MobileNetV2Block(256),
            'OM3': lambda: MobileNetV3Block(256),
            'OS1': lambda: ShuffleNetV1Block(256),
            'OS2': lambda: ShuffleNetV2Block(256),
            'OSQ': lambda: SqueezeNetBlock(256),
            'OMN': lambda: MNasNetBlock(256),
            'OEF': lambda: EfficientNetBlock(256),
            'OCB': lambda: CapsuleBlock(256),
            'OMM': lambda: MLPMixerBlock(256),
            'OSE': lambda: SqueezeExcitationBlock(256),
            'OUN': lambda: UNetBlock(256),
        }

        self.backbone = nn.ModuleList()
        gcnii_count = 0
        for b in blocks_config:
            if b == 'G2I':
                gcnii_count += 1
                self.backbone.append(GCNIIBlock(256, layer_idx=gcnii_count))
            else:
                self.backbone.append(self.block_map[b]())

        # ── Policy / Value head ──────────────────────────────────
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * 9 * 9, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        # ── Auxiliary task heads ──────────────────────────────────
        # 1. King safety: 自玉・敵玉の安全度 (2,)
        self.head_king_safety = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 2),
        )
        # 2. Material: 駒得スコア (1,)
        self.head_material = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
        )
        # 3. Mobility: 合法手数 (1,)
        self.head_mobility = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
        )
        # 4. Attack map: 自分の利き (1,9,9) — Sigmoid なし (BCEWithLogitsLoss 前提)
        self.head_attack = nn.Conv2d(256, 1, kernel_size=1)
        # 5. Threat map: 相手の利き (1,9,9)
        self.head_threat = nn.Conv2d(256, 1, kernel_size=1)
        # 6. Damage map: 危険にさらされた自駒 (1,9,9)
        self.head_damage = nn.Conv2d(256, 1, kernel_size=1)

        # ── Self-Supervised heads (Phase 2 用) ───────────────────
        self.num_piece_types = 31
        self.head_masked_board   = nn.Conv2d(256, self.num_piece_types, kernel_size=1)
        self.head_attack_map_ssl = nn.Sequential(nn.Conv2d(256, 8, kernel_size=1), nn.Sigmoid())
        self.head_move_feature   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 128))
        self.head_future         = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 256))
        self.head_search         = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 256))

    def forward(self, x, return_aux=True):
        """
        x: (B, 447)  float32
          x[:, :405] — 盤面 81マス × 5bit
          x[:, 405:] — 持ち駒 42bit
        """
        # ── 埋め込み ──────────────────────────────────────────────
        board_bits = x[:, :405].view(-1, 81, 5)
        device = x.device
        weights = torch.tensor([16, 8, 4, 2, 1], dtype=x.dtype, device=device)
        board_indices = (board_bits * weights).sum(dim=-1).long()  # (B, 81)

        board_embed = self.board_embedding(board_indices)            # (B, 81, 256)
        board_embed = board_embed.permute(0, 2, 1).view(-1, 256, 9, 9)  # (B, 256, 9, 9)

        hand_bits  = x[:, 405:]                                     # (B, 42)
        hand_embed = self.hand_mlp(hand_bits).view(-1, 256, 1, 1)  # (B, 256, 1, 1)

        h = board_embed + hand_embed  # (B, 256, 9, 9)

        # ── バックボーン ──────────────────────────────────────────
        for block in self.backbone:
            h = block(h)

        policy = self.policy_head(h)
        value  = self.value_head(h)

        if not return_aux:
            return policy, value

        aux = {
            "king_safety":    self.head_king_safety(h),
            "material":       self.head_material(h),
            "mobility":       self.head_mobility(h),
            "attack":         self.head_attack(h),          # raw logits
            "threat":         self.head_threat(h),          # raw logits
            "damage":         self.head_damage(h),          # raw logits
            "masked_board":   self.head_masked_board(h),    # raw logits
            "attack_map_ssl": self.head_attack_map_ssl(h),
            "move_feature":   self.head_move_feature(h),
            "future_rep":     self.head_future(h),
            "search_rep":     self.head_search(h),
        }
        return policy, value, aux


def create_model(num_actions, mode="modelA"):
    """
    mode に応じて異なる構成の HybridAlphaZeroNet (447bit専用) を返す。

    対応モード: modelA / modelB / modelC / modelD / modelE /
                modelF / modelG / modelH / fastA
    """
    if mode == "modelA":
        # 軽量ベースライン: ResBlock 10ブロック
        config = ['CRE'] * 10

    elif mode == "modelB":
        # CNN主体 + SE: ResBlock 20 + SqueezeExcitation 10
        config = ['CRE'] * 20 + ['OSE'] * 10

    elif mode == "modelC":
        # 多様な CNN ブロックを横断的に使う 30ブロック
        config = (
            ['CRE'] * 5 + ['CDN'] * 5 + ['CNX'] * 5
            + ['CNT'] * 5 + ['CGL'] * 5 + ['CI3'] * 5
        )

    elif mode == "modelD":
        # MobileNetV3 + SlotAttention の交互積み上げ 30ブロック
        config = ['OM3'] * 15 + ['OSB'] * 15

    elif mode == "modelE":
        # InceptionV4 純粋積み上げ 30ブロック
        config = ['CI4'] * 30

    elif mode == "modelF":
        # ResBlock + SGCGraph の交互 6ペア 30ブロック
        config = (['CRE'] * 5 + ['GSC'] * 5) * 3

    elif mode == "modelG":
        # CNN → GCN → GIN の3段階遷移 30ブロック
        config = ['CRE'] * 10 + ['GSC'] * 10 + ['GIN'] * 10

    elif mode == "modelH":
        # CNN + Graph + LocalAttention の混合 5セット 30ブロック
        config = (['CRE'] * 3 + ['GSC'] * 2 + ['TLA'] * 1) * 5

    elif mode == "fastA":
        # CPU対局用の軽量10ブロック
        config = ['OMM'] * 5 + ['OSE'] * 5

    else:
        config = ['CRE'] * 10
        print(f"[create_model] Unknown mode: '{mode}'. Falling back to CRE×10.")

    return HybridAlphaZeroNet(num_actions, config)
