import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cshogi
from model.model import create_model
from game.board import bit447_to_numpy, make_move_label
from search.npls import NPLSPlayer
from train.reinforcement_train import SelfPlayDataset, self_play_game

def run_single_self_play(weights_path, game_id, mode="modelC"):
    """
    並列実行される個別の自己対局タスク。
    """
    # 各プロセス内でPlayerインスタンスを初期化（CUDA共有問題を回避するため、必要に応じてCPUを指定する等の処理）
    # ここではプロセスごとに独立したPlayerインスタンスを作成する
    p_black = NPLSPlayer(blocks_config_mode=mode, modelfile=weights_path, name=f"P-Black-{game_id}")
    p_white = NPLSPlayer(blocks_config_mode=mode, modelfile=weights_path, name=f"P-White-{game_id}")
    
    # 探索のオーバーヘッドを下げるために、思考時間を短めに
    p_black.think_time_ms = 30
    p_white.think_time_ms = 30
    
    t0 = time.time()
    data = self_play_game(p_black, p_white)
    # print は並列実行時に出力が混ざる可能性があるが、進捗確認用
    print(f"[Parallel] Game {game_id} finished. Collected {len(data)} states in {time.time()-t0:.2f}s")
    return data

def parallel_reinforcement_training(iterations=2, num_parallel_games=8, epochs_per_iter=2, model_mode="modelC"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Parallel RL training starting on main device: {device}")
    
    weights_path = "weights/checkpoint.pth"
    model = create_model(num_actions=2187, mode=model_mode)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded initial weights from checkpoint.pth")
    else:
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        print("Initialized weights/checkpoint.pth")
        
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    for iter_idx in range(iterations):
        print(f"\n--- Parallel Iteration {iter_idx+1}/{iterations} ---")
        
        # 1. 8プロセスの並列自己対局データ収集
        all_game_data = []
        print(f"Spawning {num_parallel_games} parallel games using ProcessPoolExecutor...")
        
        # Windowsのマルチプロセッシング時の制約を考慮し、 weights_path を渡してロードさせる
        # cuda テンソルの別プロセス転送エラーを避けるため、子プロセスはCPU側で動かすことが一般的
        with ProcessPoolExecutor(max_workers=num_parallel_games) as executor:
            # プロセス数分のタスクをサブミット
            futures = [
                executor.submit(run_single_self_play, weights_path, i, model_mode)
                for i in range(num_parallel_games)
            ]
            
            for future in futures:
                try:
                    game_data = future.result()
                    all_game_data.extend(game_data)
                except Exception as e:
                    print(f"Error in parallel self-play game: {e}")
                    
        print(f"Collected total of {len(all_game_data)} states from parallel games.")
        
        if not all_game_data:
            print("No data collected, skipping training.")
            continue
            
        # 2. モデル更新
        print("Training model with aggregated self-play data...")
        dataset = SelfPlayDataset(all_game_data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(epochs_per_iter):
            epoch_loss = 0.0
            for feats, policies, values in dataloader:
                feats = feats.to(device)
                policies = policies.to(device)
                values = values.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                policy_out, value_out = model(feats, return_aux=False)
                
                loss_p = criterion_policy(policy_out, policies)
                loss_v = criterion_value(value_out, values)
                loss = loss_p + loss_v
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            print(f"  Epoch {epoch+1}/{epochs_per_iter} - Loss: {epoch_loss/len(dataloader):.4f}")
            
        # 重みを更新保存
        torch.save(model.state_dict(), weights_path)
        print("Updated checkpoint.pth with parallel RL progress")

if __name__ == "__main__":
    # WindowsでProcessPoolExecutorを使うために必要
    # （これがないと並列起動時に無限再帰インポートエラーが発生する）
    import multiprocessing
    multiprocessing.freeze_support()
    parallel_reinforcement_training(iterations=1, num_parallel_games=4, epochs_per_iter=1)
