"""
Online Trainer - 人机对战同步训练模块

核心思路:
1. 每步收集 (obs, action, player, action_mask)
2. 局末结算: 赢+1, 输-1 (MC回报, 稀疏奖励游戏)
3. AI自己的步 → Q值回归 (MSE: Q(s,a) → return)
4. 人类赢的步 → 模仿学习 (CrossEntropy: 学人类的好招)
5. 后台线程训练, 不阻塞Flask主线程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import threading
import time
import os
from collections import deque
import copy


class EpisodeBuffer:
    """单局轨迹收集器"""

    def __init__(self):
        self.transitions = []  # [(obs, action, player, mask)]
        self.winner = None

    def add(self, obs, action, player, mask):
        self.transitions.append((
            obs.copy() if isinstance(obs, np.ndarray) else obs,
            int(action),
            int(player),
            mask.copy() if isinstance(mask, np.ndarray) else mask,
        ))

    def set_winner(self, winner):
        self.winner = winner

    def is_empty(self):
        return len(self.transitions) == 0


class ReplayBuffer:
    """经验回放池 - 存已结算的transition"""

    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def push(self, obs, action, player, mask, return_val):
        with self.lock:
            self.buffer.append((
                obs, action, player, mask, return_val
            ))

    def sample(self, batch_size):
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
        obs_list = np.array([b[0] for b in batch])
        action_list = np.array([b[1] for b in batch])
        player_list = np.array([b[2] for b in batch])
        mask_list = np.array([b[3] for b in batch])
        return_list = np.array([b[4] for b in batch])
        return obs_list, action_list, player_list, mask_list, return_list

    def __len__(self):
        with self.lock:
            return len(self.buffer)


class OnlineTrainer:
    """在线训练器 - 后台线程异步训练"""

    def __init__(self, model, device='cpu', config=None):
        self.model = model
        self.device = torch.device(device)
        self.model_lock = threading.Lock()

        # 超参数
        cfg = config or {}
        self.lr = cfg.get('lr', 1e-5)               # 学习率 (小一点, 别忘太快)
        self.batch_size = cfg.get('batch_size', 64)
        self.train_interval = cfg.get('train_interval', 5)    # 每N局训练一次
        self.gamma = cfg.get('gamma', 1.0)           # MC用gamma=1
        self.imitation_weight = cfg.get('imitation_weight', 0.3)  # 模仿学习权重
        self.value_weight = cfg.get('value_weight', 0.5)         # value loss权重
        self.max_grad_norm = cfg.get('max_grad_norm', 1.0)
        self.save_interval = cfg.get('save_interval', 50)        # 每N局存一次模型

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 缓冲区
        self.replay_buffer = ReplayBuffer(capacity=cfg.get('buffer_capacity', 50000))

        # 活跃对局 (user_id -> EpisodeBuffer)
        self.active_episodes = {}
        self.episode_lock = threading.Lock()

        # 统计
        self.stats = {
            'games_played': 0,
            'ai_wins': 0,
            'human_wins': 0,
            'train_steps': 0,
            'total_loss': 0.0,
            'q_loss': 0.0,
            'imitation_loss': 0.0,
            'value_loss': 0.0,
            'last_train_time': None,
            'buffer_size': 0,
        }

        # 训练线程控制
        self._running = False
        self._train_thread = None

        # 模型保存路径
        self.save_dir = cfg.get('save_dir', 'models')
        os.makedirs(self.save_dir, exist_ok=True)

    def start(self):
        """启动后台训练线程"""
        if self._running:
            return
        self._running = True
        self._train_thread = threading.Thread(target=self._train_loop, daemon=True)
        self._train_thread.start()
        print("[OnlineTrainer] 后台训练线程已启动")

    def stop(self):
        """停止训练线程"""
        self._running = False
        if self._train_thread:
            self._train_thread.join(timeout=5)
        print("[OnlineTrainer] 训练线程已停止")

    def begin_episode(self, user_id):
        """新对局开始"""
        with self.episode_lock:
            self.active_episodes[user_id] = EpisodeBuffer()

    def record_step(self, user_id, obs, action, player, mask):
        """记录一步"""
        with self.episode_lock:
            if user_id in self.active_episodes:
                self.active_episodes[user_id].add(obs, action, player, mask)

    def end_episode(self, user_id, winner, ai_player):
        """对局结束, 结算MC回报"""
        with self.episode_lock:
            ep = self.active_episodes.pop(user_id, None)
            if ep is None or ep.is_empty():
                return

        ep.set_winner(winner)

        # 计算MC回报: 对AI来说, 赢=+1, 输=-1
        # 对人类来说, 赢=+1, 输=-1 (用于模仿学习)
        for obs, action, player, mask in ep.transitions:
            if player == ai_player:
                # AI的步: return = AI赢? +1 : -1
                return_val = 1.0 if winner == ai_player else -1.0
            else:
                # 人类的步: return = 人类赢? +1 : -1
                # 赢了的人类动作值得模仿
                return_val = 1.0 if winner == player else -1.0
            self.replay_buffer.push(obs, action, player, mask, return_val)

        # 更新统计
        self.stats['games_played'] += 1
        if winner == ai_player:
            self.stats['ai_wins'] += 1
        else:
            self.stats['human_wins'] += 1
        self.stats['buffer_size'] = len(self.replay_buffer)

    def _train_loop(self):
        """后台训练循环"""
        while self._running:
            # 等攒够数据
            if len(self.replay_buffer) < self.batch_size:
                time.sleep(2.0)
                continue

            # 每train_interval局训练一次
            if self.stats['games_played'] % self.train_interval != 0:
                time.sleep(1.0)
                continue

            # 训练一个batch
            batch = self.replay_buffer.sample(self.batch_size)
            if batch is None:
                time.sleep(2.0)
                continue

            loss_dict = self._train_step(batch)
            if loss_dict:
                self.stats['train_steps'] += 1
                self.stats['total_loss'] += loss_dict['total']
                self.stats['q_loss'] += loss_dict['q']
                self.stats['imitation_loss'] += loss_dict['imitation']
                self.stats['value_loss'] += loss_dict['value']
                self.stats['last_train_time'] = time.strftime('%H:%M:%S')

            # 定期存模型
            if self.stats['games_played'] % self.save_interval == 0 and self.stats['train_steps'] > 0:
                self._save_checkpoint()

            time.sleep(0.5)

    def _train_step(self, batch):
        """执行一步训练"""
        obs_np, action_np, player_np, mask_np, return_np = batch

        obs_t = torch.FloatTensor(obs_np).to(self.device)
        actions_t = torch.LongTensor(action_np).to(self.device)
        players_t = torch.LongTensor(player_np).to(self.device)
        masks_t = torch.FloatTensor(mask_np).to(self.device)
        returns_t = torch.FloatTensor(return_np).to(self.device)

        with self.model_lock:
            self.model.train()

            # ---- Q值预测 ----
            q_values = self.model(obs_t)  # (B, 67)

            # AI步: MSE loss on Q(s, a) → return
            ai_mask = (players_t == 1)  # 假设AI是player 1
            if ai_mask.any():
                q_ai = q_values[ai_mask]
                a_ai = actions_t[ai_mask]
                r_ai = returns_t[ai_mask]
                q_pred = q_ai.gather(1, a_ai.unsqueeze(1)).squeeze(1)
                q_loss = F.mse_loss(q_pred, r_ai)
            else:
                q_loss = torch.tensor(0.0, device=self.device)

            # 人类赢的步: 模仿学习 (cross-entropy on human action)
            human_win_mask = (players_t == 0) & (returns_t > 0)
            if human_win_mask.any():
                q_human = q_values[human_win_mask]
                m_human = masks_t[human_win_mask]
                a_human = actions_t[human_win_mask]
                # 屏蔽非法动作后做softmax
                q_masked = q_human.clone()
                q_masked[m_human == 0] = float('-inf')
                log_probs = F.log_softmax(q_masked, dim=1)
                imitation_loss = F.nll_loss(log_probs, a_human)
            else:
                imitation_loss = torch.tensor(0.0, device=self.device)

            # Value loss (V5有value_head)
            if hasattr(self.model, 'get_value'):
                values = self.model.get_value(obs_t)
                # 只对AI步计算value loss
                if ai_mask.any():
                    v_loss = F.mse_loss(values[ai_mask], returns_t[ai_mask])
                else:
                    v_loss = torch.tensor(0.0, device=self.device)
            else:
                v_loss = torch.tensor(0.0, device=self.device)

            # 总loss
            total_loss = q_loss + self.imitation_weight * imitation_loss + self.value_weight * v_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.model.eval()

        return {
            'total': total_loss.item(),
            'q': q_loss.item(),
            'imitation': imitation_loss.item(),
            'value': v_loss.item(),
        }

    def _save_checkpoint(self):
        """保存模型checkpoint"""
        with self.model_lock:
            path = os.path.join(self.save_dir, 'dmc_v5_online_latest.pth')
            torch.save({
                'model_state_dict': copy.deepcopy(self.model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                'stats': dict(self.stats),
                'online_trained': True,
            }, path)
        print(f"[OnlineTrainer] 模型已保存: {path} (games={self.stats['games_played']}, train_steps={self.stats['train_steps']})")

    def get_stats(self):
        """获取训练统计"""
        s = dict(self.stats)
        s['buffer_size'] = len(self.replay_buffer)
        if s['train_steps'] > 0:
            s['avg_loss'] = s['total_loss'] / s['train_steps']
            s['avg_q_loss'] = s['q_loss'] / s['train_steps']
            s['avg_imitation_loss'] = s['imitation_loss'] / s['train_steps']
            s['avg_value_loss'] = s['value_loss'] / s['train_steps']
        s['ai_winrate'] = s['ai_wins'] / max(s['games_played'], 1)
        return s

    def predict(self, obs, mask):
        """线程安全的推理"""
        with self.model_lock:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q = self.model(obs_t).cpu().numpy()[0]
        q_masked = q.copy()
        q_masked[~mask] = float('-inf')
        return int(np.argmax(q_masked)), q
