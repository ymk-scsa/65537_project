import numpy as np

class SPSA:
    """
    同時摂動確率近似 (Simultaneous Perturbation Stochastic Approximation) による
    探索ハイパーパラメータの自動最適化。
    """
    def __init__(self, init_theta, a=0.1, c=0.05, A=10, alpha=0.602, gamma=0.101):
        self.theta = np.array(init_theta, dtype=float)
        self.d = len(init_theta)
        
        # SPSAのハイパーパラメータ
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.k = 0

    def get_perturbed_parameters(self):
        """
        評価用の2つの摂動パラメータ (theta_plus, theta_minus) と摂動ベクトル delta を返す。
        """
        # ベルヌーイ分布（1 or -1）で摂動ベクトルを生成
        self.delta = np.random.choice([-1, 1], size=self.d)
        
        # 摂動ステップサイズ
        self.ck = self.c / ((self.k + 1) ** self.gamma)
        
        self.theta_plus = self.theta + self.ck * self.delta
        self.theta_minus = self.theta - self.ck * self.delta
        
        # 負のパラメータを防ぐための制約（重みは0以上）
        self.theta_plus = np.maximum(self.theta_plus, 0.0)
        self.theta_minus = np.maximum(self.theta_minus, 0.0)
        
        return self.theta_plus, self.theta_minus

    def update(self, loss_plus, loss_minus):
        """
        評価結果 (loss_plus, loss_minus) に基づいて theta を更新する。
        loss_plus  : theta_plus での損失（または負の勝率）
        loss_minus : theta_minus での損失（または負の勝率）
        """
        # 勾配の近似
        # 各パラメータ成分の勾配: (loss_plus - loss_minus) / (2 * ck * delta)
        grad = (loss_plus - loss_minus) / (2.0 * self.ck * self.delta)
        
        # 学習率の計算
        ak = self.a / ((self.k + 1 + self.A) ** self.alpha)
        
        # パラメータの更新
        self.theta = self.theta - ak * grad
        self.theta = np.maximum(self.theta, 0.0)  # 0未満にならないように制約
        
        self.k += 1
        return self.theta
