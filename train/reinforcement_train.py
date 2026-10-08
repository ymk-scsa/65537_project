import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cshogi
from model.model import create_model
from game.board import bit447_to_numpy, make_move_label, make_result
from search.npls import NPLSPlayer

class SelfPlayDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        feat, policy_dist, value = self.data_list[idx]
        return (
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(policy_dist, dtype=torch.float32),
            torch.tensor(value, dtype=torch.float32)
        )

def self_play_game(player_black, player_white, max_moves=200):
    """
    1対局の自己対局を実行し、学習用データを収集する。
    """
    board = cshogi.Board()
    game_history = []  # (sfen, chosen_move_label, current_player_turn)
    
    move_count = 0
    best_move_usi = None
    while not board.is_game_over() and move_count < max_moves:
        current_player = player_black if board.turn == cshogi.BLACK else player_white
        
        # プレイヤーの局面情報を更新
        # position コマンドの動きをシミュレート
        sfen = board.sfen()
        current_player.position(sfen, [])
        
        # 思考実行
        current_player.set_limits(think_time_ms=50) # 高速動作のために思考時間を短く設定
        best_move_usi, _ = current_player.go()
        
        if best_move_usi in ("resign", "resnet"):
            break
            
        move = board.move_from_usi(best_move_usi)
        
        # 履歴保存
        feat = bit447_to_numpy(board)
        move_label = make_move_label(move, board.turn)
        
        # 簡易的に、選択された手に確率 1.0 のOne-hotを方策ターゲットとする
        policy_target = np.zeros(2187, dtype=np.float32)
        policy_target[move_label] = 1.0
        
        game_history.append((feat, policy_target, board.turn))
        
        board.push(move)
        move_count += 1

    # 勝敗結果の反映
    # 手番から見た報酬を算出（Board.result() は cshogi に存在しないため自前で判定）
    reward_black = 0.5
    reward_white = 0.5
    if board.is_game_over() or best_move_usi in ("resign", "resnet"):
        # 詰み・投了: その時点の手番側の負け
        if board.turn == cshogi.BLACK:
            reward_black, reward_white = 0.0, 1.0
        else:
            reward_black, reward_white = 1.0, 0.0
        
    training_data = []
    for feat, policy_target, turn in game_history:
        reward = reward_black if turn == cshogi.BLACK else reward_white
        training_data.append((feat, policy_target, reward))
        
    return training_data

def reinforcement_learning_loop(iterations=2, games_per_iter=2, epochs_per_iter=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RL training on device: {device}")
    
    weights_path = "weights/checkpoint.pth"
    # 教師あり学習の重みがあればそれをロード、なければ初期モデル
    model = create_model(num_actions=2187, mode="modelC")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded initial weights from checkpoint.pth")
    else:
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        print("Initialized new model weights")
        
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    # 強化学習サイクル
    for iter_idx in range(iterations):
        print(f"\n--- Iteration {iter_idx+1}/{iterations} ---")
        
        # 自己対局用プレイヤーの初期化
        player_black = NPLSPlayer(blocks_config_mode="modelC", modelfile=weights_path, name="Player-Black")
        player_white = NPLSPlayer(blocks_config_mode="modelC", modelfile=weights_path, name="Player-White")
        
        # 1. 自己対局によるデータ収集
        all_game_data = []
        print(f"Generating data with {games_per_iter} self-play games...")
        for g in range(games_per_iter):
            t0 = time.time()
            game_data = self_play_game(player_black, player_white)
            all_game_data.extend(game_data)
            print(f"  Game {g+1} completed, steps: {len(game_data)}, time: {time.time()-t0:.2f}s")
            
        if not all_game_data:
            print("No data collected, skipping training step.")
            continue
            
        # 2. モデルの学習
        print(f"Training on {len(all_game_data)} states...")
        dataset = SelfPlayDataset(all_game_data)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        model.train()
        for epoch in range(epochs_per_iter):
            epoch_loss = 0.0
            for feats, policies, values in dataloader:
                feats = feats.to(device)
                policies = policies.to(device)
                values = values.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                policy_out, value_out = model(feats, return_aux=False)
                
                # 方策ターゲットは確率分布。KLDivergence または CrossEntropy を計算。
                # PyTorchのCrossEntropyLossはターゲットが確率分布の場合もサポート（PyTorch 1.10以降）
                loss_p = criterion_policy(policy_out, policies)
                loss_v = criterion_value(value_out, values)
                loss = loss_p + loss_v
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            print(f"  Epoch {epoch+1}/{epochs_per_iter} - Loss: {epoch_loss/len(dataloader):.4f}")
            
        # 最新のモデルの重みを保存
        torch.save(model.state_dict(), weights_path)
        print("Updated checkpoint.pth with RL progress")

if __name__ == "__main__":
    reinforcement_learning_loop(iterations=2, games_per_iter=2, epochs_per_iter=2)
