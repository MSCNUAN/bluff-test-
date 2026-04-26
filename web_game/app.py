"""
吹牛骰子 Web 游戏服务器 - v1.4 正式上线版（完美支持互联网多用户并发）
"""

from flask import Flask, render_template, jsonify, request, session
import torch
import numpy as np
import os
import sys
import urllib.request
import uuid
import importlib
from importlib.machinery import SourcelessFileLoader
from importlib.util import spec_from_loader, module_from_spec

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, parent_dir)

from online_trainer import OnlineTrainer


def resolve_module(name):
    """
    兼容两种打包方式：
    1) 正常源码模块（*.py）
    2) 仅包含 __pycache__/*.pyc 的 sourceless 模块
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pyc_path = os.path.join(parent_dir, '__pycache__', f'{name}.cpython-310.pyc')
        if not os.path.exists(pyc_path):
            raise
        loader = SourcelessFileLoader(name, pyc_path)
        spec = spec_from_loader(name, loader)
        module = module_from_spec(spec)
        loader.exec_module(module)
        return module


env_module = resolve_module('bluff_dice_env_v3')
model_module = resolve_module('model_dmc')
BluffDiceEnvV3 = env_module.BluffDiceEnvV3
DMCNetwork = model_module.DMCNetwork
DMCNetworkV5 = model_module.DMCNetworkV5

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# 生产环境请通过环境变量注入 SECRET_KEY
app.secret_key = os.getenv('SECRET_KEY', 'dev-only-secret-change-me')

ai_model = None
ai_device = None
online_trainer = None  # 在线训练器
training_enabled = os.getenv('TRAINING_ENABLED', 'true').lower() in ('1', 'true', 'yes', 'on')
runtime_initialized = False


def env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


MODEL_STATE_DIM = env_int('MODEL_STATE_DIM', 44)
MODEL_NUM_ACTIONS = env_int('MODEL_NUM_ACTIONS', 67)
MODEL_HISTORY_LEN = env_int('MODEL_HISTORY_LEN', 5)
MODEL_V5_HIDDEN_DIM = env_int('MODEL_V5_HIDDEN_DIM', 384)
MODEL_V3_HIDDEN_DIM = env_int('MODEL_V3_HIDDEN_DIM', 256)
=======
main

# ==========================================
# 🌟 核心升级：多桌游戏大厅登记册 (替代全局变量)
# ==========================================
active_games = {}

def get_user_session():
    """发牌员：给每个访问的浏览器分配并获取独立的对局环境"""
    if 'user_id' not in session:
        # 给新玩家发一张唯一的 UUID 房卡
        session['user_id'] = str(uuid.uuid4())
        
    uid = session['user_id']
    
    # 如果登记册里没有这个玩家的桌子，给他开一桌新的
    if uid not in active_games:
        active_games[uid] = GameSession(user_id=uid)
        
    return active_games[uid]
# ==========================================

def load_ai_model():
    """加载 AI 模型，可通过环境变量覆盖默认下载源。"""
    global ai_model, ai_device
    
    ai_device = torch.device('cpu')

    custom_model_url = os.getenv('BLUFF_MODEL_URL', '').strip()
    custom_model_path = os.getenv('BLUFF_MODEL_PATH', '').strip()

    def _extract_state_dict(checkpoint_obj):
        if isinstance(checkpoint_obj, dict) and 'model_state_dict' in checkpoint_obj:
            return checkpoint_obj['model_state_dict']
        return checkpoint_obj

    def _build_model_from_state_dict(state_dict):
        try:
codex/connect-model-to-deployment-for-game-ze7ven
            model = DMCNetworkV5(
                state_dim=MODEL_STATE_DIM,
                num_actions=MODEL_NUM_ACTIONS,
                history_len=MODEL_HISTORY_LEN,
                hidden_dim=MODEL_V5_HIDDEN_DIM,
            ).to(ai_device)
=======
            model = DMCNetworkV5(state_dim=44, num_actions=67, history_len=5, hidden_dim=384).to(ai_device)
 main
            model.load_state_dict(state_dict)
            print("[OK] 模型按 V5 架构加载成功")
            return model
        except Exception:
 codex/connect-model-to-deployment-for-game-ze7ven
            model = DMCNetwork(
                hidden_dim=MODEL_V3_HIDDEN_DIM,
                num_actions=MODEL_NUM_ACTIONS,
                history_len=MODEL_HISTORY_LEN
            ).to(ai_device)
=======
            model = DMCNetwork(hidden_dim=256, num_actions=67, history_len=5).to(ai_device)
 main
            model.load_state_dict(state_dict)
            print("[OK] 模型按 V3 架构加载成功")
            return model

    # 优先：外部配置的模型（你上传到发行版后的直链）
    if custom_model_url or custom_model_path:
        source_desc = custom_model_path or custom_model_url
        print(f"[INFO] 尝试加载外部模型: {source_desc}")
        try:
            model_path = custom_model_path
            if custom_model_url:
                os.makedirs(os.path.join(script_dir, 'models'), exist_ok=True)
                model_path = os.path.join(script_dir, 'models', 'external_model_latest.pth')
                urllib.request.urlretrieve(custom_model_url, model_path)

            checkpoint = torch.load(model_path, map_location=ai_device, weights_only=False)
            state_dict = _extract_state_dict(checkpoint)
            ai_model = _build_model_from_state_dict(state_dict)
            ai_model.eval()
            print(f"[OK] 外部模型加载成功: {source_desc}")
            return True
        except Exception as e:
            print(f"[WARN] 外部模型加载失败，回退到内置模型: {e}")

    # 次优先：本地 V5 最强模型 (ELO=1140)
    v5_path = os.path.join(script_dir, 'models', 'dmc_v5_best.pth')
    if os.path.exists(v5_path):
        ai_model = DMCNetworkV5(
            state_dim=MODEL_STATE_DIM,
            num_actions=MODEL_NUM_ACTIONS,
            history_len=MODEL_HISTORY_LEN,
            hidden_dim=MODEL_V5_HIDDEN_DIM
        ).to(ai_device)
        checkpoint = torch.load(v5_path, map_location=ai_device, weights_only=False)
        ai_model.load_state_dict(checkpoint['model_state_dict'])
        ai_model.eval()
        print(f"[OK] V5 AI 模型加载成功 (ELO=1140): {v5_path}")
        return True

    # 回退: 加载旧 V3 模型
    local_path = os.path.join(script_dir, 'models', 'dmc_v3_final.pth')
    if not os.path.exists(local_path):
        print("[INFO] 本地未找到模型，正在从 GitHub 下载...")
        try:
            url = "https://github.com/MSCNUAN/bluff-dice/releases/download/v1.4-model/dmc_v3_final.pth"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(url, local_path)
            print("[OK] 模型下载完成")
        except Exception as e:
            print(f"[ERROR] 模型下载失败: {e}")
            print("游戏仍可运行，但AI将使用随机策略")
            return False

    ai_model = DMCNetwork(
        hidden_dim=MODEL_V3_HIDDEN_DIM,
        num_actions=MODEL_NUM_ACTIONS,
        history_len=MODEL_HISTORY_LEN
    ).to(ai_device)
    checkpoint = torch.load(local_path, map_location=ai_device, weights_only=False)
    ai_model.load_state_dict(_extract_state_dict(checkpoint))
    ai_model.eval()
    print(f"[OK] V3 AI 模型加载成功: {local_path}")
    return True


@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    """在线重载模型：支持传入发行版模型直链。"""
    global ai_model
    data = request.json or {}

    model_url = str(data.get('url', '')).strip()
    model_path = str(data.get('path', '')).strip()
    reset_default = bool(data.get('reset_default', False))

    if not reset_default and not model_url and not model_path:
        return jsonify({'success': False, 'error': '请传入 url 或 path；如需恢复默认模型请传 reset_default=true'}), 400

    if model_url:
        if not (model_url.startswith('http://') or model_url.startswith('https://')):
            return jsonify({'success': False, 'error': 'url 必须以 http:// 或 https:// 开头'}), 400
        os.environ['BLUFF_MODEL_URL'] = model_url
    elif 'BLUFF_MODEL_URL' in os.environ and (reset_default or model_path):
        os.environ.pop('BLUFF_MODEL_URL')

    if model_path:
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': 'path 不存在，请检查服务器本地路径'}), 400
        os.environ['BLUFF_MODEL_PATH'] = model_path
    elif 'BLUFF_MODEL_PATH' in os.environ and (reset_default or model_url):
        os.environ.pop('BLUFF_MODEL_PATH')

    loaded = load_ai_model()
    if not loaded or ai_model is None:
        return jsonify({'success': False, 'error': '模型加载失败，请检查 URL / PATH'}), 400

    return jsonify({
        'success': True,
        'message': '模型已重载',
        'source': model_path or model_url or 'default'
    })


def init_online_trainer():
    """初始化在线训练器"""
    global online_trainer, ai_model, ai_device, training_enabled
    if not training_enabled or ai_model is None:
        return
    trainer_config = {
        'lr': env_float('TRAIN_LR', 1e-5),
        'batch_size': env_int('TRAIN_BATCH_SIZE', 64),
        'train_interval': env_int('TRAIN_INTERVAL', 3),         # 每N局训练一次
        'imitation_weight': env_float('TRAIN_IMITATION_WEIGHT', 0.3),
        'value_weight': env_float('TRAIN_VALUE_WEIGHT', 0.5),
        'save_interval': env_int('TRAIN_SAVE_INTERVAL', 30),    # 每N局存模型
        'save_dir': os.path.join(script_dir, 'models'),
        'buffer_capacity': env_int('TRAIN_BUFFER_CAPACITY', 50000),
        'max_grad_norm': env_float('TRAIN_MAX_GRAD_NORM', 1.0),
    }
    online_trainer = OnlineTrainer(
        model=ai_model,
        device=ai_device,
        config=trainer_config
    )
    online_trainer.start()
    print(f"[OK] 在线训练器已启动: {trainer_config}")


def init_runtime_once():
    """确保在 gunicorn/import 场景下也会执行一次初始化。"""
    global runtime_initialized
    if runtime_initialized:
        return
    load_ai_model()
    init_online_trainer()
    runtime_initialized = True
    print("[OK] 运行时初始化完成")


# 关键：支持 gunicorn `web_game.app:app` 启动时自动初始化
init_runtime_once()


def init_runtime_once():
    """确保在 gunicorn/import 场景下也会执行一次初始化。"""
    global runtime_initialized
    if runtime_initialized:
        return
    load_ai_model()
    init_online_trainer()
    runtime_initialized = True
    print("[OK] 运行时初始化完成")


# 关键：支持 gunicorn `web_game.app:app` 启动时自动初始化
init_runtime_once()


class GameSession:
    def __init__(self, user_id=None):
        self.env = BluffDiceEnvV3(history_len=MODEL_HISTORY_LEN)
        self.human_player = 0
        self.ai_player = 1
        self.game_over = False
        self.winner = None
        self.last_action = None
        self.last_actor = None
        self.message = ""
        self.difficulty = "normal"
        self.obs = None
        self.info = None
        self.user_id = user_id  # 用于训练数据关联
        # 兜底初始化：即使前端未先调用 /api/start，也能安全返回状态与建议
        self.reset(human_first=True, difficulty="normal")

    def reset(self, human_first=True, difficulty="normal"):
        # 先结算上一局（如果有）
        self._finish_episode()

        self.difficulty = difficulty
        self.human_player = 0 if human_first else 1
        self.ai_player = 1 - self.human_player
        first = self.human_player if human_first else self.ai_player
        self.obs, self.info = self.env.reset(first_player=first)
        self.game_over = False
        self.winner = None
        self.last_action = None
        self.last_actor = None
        self.message = "游戏开始！" if human_first else "AI 先手"

        # 通知训练器：新对局开始
        if online_trainer and self.user_id:
            online_trainer.begin_episode(self.user_id)

    def get_state(self):
        return {
            'human_player': self.human_player,
            'ai_player': self.ai_player,
            'current_player': self.env.current_player,
            'human_hand': self.env.hands[self.human_player],
            'ai_hand': self.env.hands[self.ai_player],
            'last_call': self.env.last_call,
            'one_called': self.env.one_called,
            'game_over': self.game_over,
            'winner': self.winner,
            'last_action': self.last_action,
            'last_actor': self.last_actor,
            'message': self.message,
            'is_human_turn': self.env.current_player == self.human_player and not self.game_over,
            'difficulty': self.difficulty
        }

    # 帮你补全了获取合法动作的方法
    def get_legal_actions(self):
        if self.game_over: return []
        mask = self.info['action_mask']
        legal_indices = np.where(mask)[0]
        actions = []
        for idx in legal_indices:
            if idx == 0:
                actions.append({'action': int(idx), 'text': '开牌！'})
            else:
                M, N = self.env.action_to_call[idx]
                actions.append({'action': int(idx), 'text': f'{M} 个 {N}'})
        return actions

    # 帮你补全了玩家行动的方法，包含严谨的防报错拦截
    def human_action(self, action_idx):
        if self.game_over or self.env.current_player != self.human_player:
            return False, "非法操作：还没轮到你呢！"
            
        action_idx = int(action_idx)
        if not self.info['action_mask'][action_idx]:
            return False, "非法操作：必须比上家叫得更大哦！"

        # 记录人类动作（训练用）
        if online_trainer and self.user_id:
            online_trainer.record_step(
                self.user_id, self.obs, action_idx,
                self.human_player, self.info['action_mask']
            )
            
        self.last_action = self.env.action_to_string(action_idx)
        self.last_actor = "玩家"
        if action_idx == 0:
            self.message = "你选择了开牌！"
        else:
            M, N = self.env.action_to_call[action_idx]
            self.message = f"你叫了 {M} 个 {N}"
            
        self.obs, _, done, _, self.info = self.env.step(action_idx)
        if done:
            self.game_over = True
            self.winner = self.env.winner
            self.message = "🎉 太棒了，你赢了！" if self.winner == self.human_player else "😢 差一点，AI 赢了！"
            self._finish_episode()
        return True, self.message

    def get_ai_suggestions(self, top_k=5):
        global ai_model, ai_device
        if ai_model is None: return []
        mask = self.info['action_mask']
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(self.obs).unsqueeze(0).to(ai_device)
            q_values = ai_model(obs_tensor).cpu().numpy()[0]
        legal = np.where(mask)[0]
        scores = [(idx, q_values[idx]) for idx in legal]
        scores.sort(key=lambda x: x[1], reverse=True)

        # softmax: 把所有合法动作的 Q 值映射为概率
        q_vals = np.array([q for _, q in scores])
        q_vals -= q_vals.max()  # 防止 exp 溢出
        exps = np.exp(q_vals)
        probs = exps / exps.sum()

        suggestions = []
        for i, (idx, q) in enumerate(scores[:top_k]):
            suggestions.append({
                'action': int(idx),
                'text': self.env.action_to_string(idx),
                'prob': float(probs[i]),
                'q_value': float(q)
            })
        return suggestions

    def ai_action(self):
        global ai_model, ai_device, online_trainer
        if self.game_over or self.env.current_player != self.ai_player:
            return False, "非法操作"
        mask = self.info['action_mask']

        # 优先用训练器推理（线程安全）
        if online_trainer and ai_model is not None:
            action_idx, q_values = online_trainer.predict(self.obs, mask)
        elif ai_model is None:
            legal = np.where(mask)[0]
            action_idx = int(np.random.choice(legal))
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(self.obs).unsqueeze(0).to(ai_device)
                q_values = ai_model(obs_tensor).cpu().numpy()[0]
                q_masked = q_values.copy()
                q_masked[~mask] = float('-inf')

                if self.difficulty == "easy":
                    q_masked += np.random.normal(0, 3, q_masked.shape)
                elif self.difficulty == "hard":
                    q_masked += np.random.normal(0, 0.5, q_masked.shape)

                action_idx = int(np.argmax(q_masked))

        # 记录AI动作（训练用）
        if online_trainer and self.user_id:
            online_trainer.record_step(
                self.user_id, self.obs, action_idx,
                self.ai_player, mask
            )
        
        self.last_action = self.env.action_to_string(action_idx)
        self.last_actor = "AI"
        if action_idx == 0:
            self.message = "AI 选择开牌！"
        else:
            M, N = self.env.action_to_call[action_idx]
            self.message = f"AI 叫了 {M} 个 {N}"
        
        self.obs, _, done, _, self.info = self.env.step(action_idx)
        if done:
            self.game_over = True
            self.winner = self.env.winner
            self.message = "🎉 太棒了，你赢了！" if self.winner == self.human_player else "😢 差一点，AI 赢了！"
            self._finish_episode()
        return True, self.message

    def _finish_episode(self):
        """对局结束，通知训练器结算"""
        if online_trainer and self.user_id and self.winner is not None:
            online_trainer.end_episode(self.user_id, self.winner, self.ai_player)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/updates')
def updates():
    return render_template('updates.html')


@app.route('/promo')
def promo():
    return render_template('promo.html')


# --- 下方的所有路由接口都已升级，从 get_user_session() 获取数据 ---

@app.route('/api/start', methods=['POST'])
def start_game():
    data = request.json or {}
    
    # 每次点“重新开始”，确保给玩家创建一个全新的环境
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    uid = session['user_id']
    
    new_game = GameSession(user_id=uid)
    new_game.reset(human_first=data.get('human_first', True), difficulty=data.get('difficulty', 'normal'))
    if not data.get('human_first', True):
        new_game.ai_action()
        
    active_games[uid] = new_game # 将新开的桌子登记到大厅
    return jsonify({'success': True, 'state': new_game.get_state()})


@app.route('/api/state', methods=['GET'])
def get_state():
    game = get_user_session()
    return jsonify({'success': True, 'state': game.get_state()})


@app.route('/api/actions', methods=['GET'])
def get_actions():
    game = get_user_session()
    return jsonify({'success': True, 'actions': game.get_legal_actions()})


@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    game = get_user_session()
    return jsonify({'success': True, 'suggestions': game.get_ai_suggestions()})


@app.route('/api/action', methods=['POST'])
def do_action():
    game = get_user_session()
    data = request.json
    
    # 玩家执行操作
    success, msg = game.human_action(data.get('action'))
    if not success:
        return jsonify({'success': False, 'error': msg})
        
    # 如果游戏没结束且轮到 AI，AI自动执行操作
    if not game.game_over and game.env.current_player == game.ai_player:
        game.ai_action()
        
    return jsonify({'success': True, 'state': game.get_state()})


@app.route('/api/reveal', methods=['POST'])
def reveal_hands():
    """游戏结束时揭示双方手牌"""
    game = get_user_session()
    if not game.game_over:
        return jsonify({'success': False, 'error': 'game not over'})
    return jsonify({
        'success': True,
        'human_hand': game.env.hands[game.human_player],
        'ai_hand': game.env.hands[game.ai_player],
        'total_count': game.env.last_call[0] if game.env.last_call else 0,
        'called_point': game.env.last_call[1] if game.env.last_call else 0
    })


@app.route('/pick_dice')
def pick_dice():
    return render_template('dice_picker.html')


@app.route('/api/set_style', methods=['POST'])
def set_style():
    return jsonify({'success': True})


@app.route('/api/training_stats', methods=['GET'])
def get_training_stats():
    """获取在线训练统计"""
    if online_trainer is None:
        return jsonify({'success': True, 'training': False, 'message': '在线训练未启用'})
    stats = online_trainer.get_stats()
    return jsonify({'success': True, 'training': True, 'stats': stats})


@app.route('/api/training_toggle', methods=['POST'])
def toggle_training():
    """开关在线训练"""
    global online_trainer, training_enabled
    data = request.json or {}
    training_enabled = data.get('enabled', True)
    if not training_enabled and online_trainer:
        online_trainer.stop()
    elif training_enabled and online_trainer:
        online_trainer.start()
    return jsonify({'success': True, 'training_enabled': training_enabled})


if __name__ == '__main__':
    print("="*80)
    print("[Bluff Dice v2.0] - Online Learning Edition")
    print("="*80)
    init_runtime_once()
    port = int(os.getenv('PORT', '5000'))
    print(f"服务器运行中 → http://localhost:{port}")
    print("="*80)
    # 此处的 host='0.0.0.0' 允许外部网络访问
    app.run(debug=False, host='0.0.0.0', port=port)
