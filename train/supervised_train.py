import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cshogi
from model.model import create_model
from game.board import bit447_to_numpy, make_move_label, make_result

class ShogiDataset(Dataset):
    """
    教師あり学習用のダミー/実データセット。
    """
    def __init__(self, num_samples=1000):
        # 実際にはファイルから読み込むが、ここでは動作確認用にダミーデータを自動生成する
        self.num_samples = num_samples
        self.features = []
        self.policy_labels = []
        self.value_labels = []
        
        # ダミーの局面を作成
        board = cshogi.Board()
        for _ in range(num_samples):
            # ランダムに数手進める
            board.set_startpos()
            for _ in range(np.random.randint(0, 40)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(np.random.choice(moves))
            
            # 特徴量抽出 (447bit)
            feat = bit447_to_numpy(board)
            self.features.append(feat)
            
            # ダミーの最善手ラベル (ランダム)
            legal_moves = list(board.legal_moves)
            if legal_moves:
                chosen_move = np.random.choice(legal_moves)
                label = make_move_label(chosen_move, board.turn)
            else:
                label = 0
            self.policy_labels.append(label)
            
            # ダミーの勝敗結果 (-1.0 ~ 1.0)
            self.value_labels.append(np.random.uniform(-1.0, 1.0))
            
        self.features = np.array(self.features, dtype=np.float32)
        self.policy_labels = np.array(self.policy_labels, dtype=np.int64)
        self.value_labels = np.array(self.value_labels, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.policy_labels[idx]),
            torch.tensor(self.value_labels[idx])
        )

def train(epochs=5, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 447bit入力モードでモデルを作成
    model = create_model(num_actions=2187, mode="modelC")
    model.to(device)
    
    dataset = ShogiDataset(num_samples=200)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    os.makedirs("weights", exist_ok=True)
    
    print("Starting Supervised Training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (features, policies, values) in enumerate(dataloader):
            features = features.to(device)
            policies = policies.to(device)
            values = values.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # 補助損失を無視するために return_aux=False にする
            policy_out, value_out = model(features, return_aux=False)
            
            loss_p = criterion_policy(policy_out, policies)
            loss_v = criterion_value(value_out, values)
            loss = loss_p + loss_v
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")
        
    # 重みを保存
    torch.save(model.state_dict(), "weights/checkpoint.pth")
    print("Model saved to weights/checkpoint.pth")

if __name__ == "__main__":
    train(epochs=2, batch_size=16)
