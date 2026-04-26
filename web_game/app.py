"""
吹牛骰子 Web 游戏服务器
Render / Gunicorn 兼容版
"""

from flask import Flask, render_template, jsonify, request, session
import importlib
import os
import sys
import uuid
import urllib.request
from importlib.machinery import SourcelessFileLoader
from importlib.util import module_from_spec, spec_from_loader

import numpy as np
import torch


# ==========================================
# 路径处理：保证 Render / 本地运行都能找到模块
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

for path in (script_dir, parent_dir):
    if path not in sys.path:
        sys.path.insert(0, path)


try:
    from online_trainer import OnlineTrainer
except Exception as e:
    OnlineTrainer = None
    print(f"[WARN] online_trainer 导入失败，在线训练将被禁用: {e}")


def resolve_module(name):
    """
    兼容两种打包方式：
    1) 正常源码模块：*.py
    2) 仅包含 __pycache__/*.pyc 的 sourceless 模块
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        cache_tag = getattr(sys.implementation, "cache_tag", "cpython-310")
        pyc_path = os.path.join(parent_dir, "__pycache__", f"{name}.{cache_tag}.pyc")

        if not os.path.exists(pyc_path):
            # 兼容你当前 Render 日志里的 Python 3.10 环境
            pyc_path = os.path.join(parent_dir, "__pycache__", f"{name}.cpython-310.pyc")

        if not os.path.exists(pyc_path):
            raise

        loader = SourcelessFileLoader(name, pyc_path)
        spec = spec_from_loader(name, loader)
        module = module_from_spec(spec)
        loader.exec_module(module)
        return module


env_module = resolve_module("bluff_dice_env_v3")
model_module = resolve_module("model_dmc")

BluffDiceEnvV3 = env_module.BluffDiceEnvV3
DMCNetwork = model_module.DMCNetwork
DMCNetworkV5 = model_module.DMCNetworkV5


# ==========================================
# Flask 基础配置
# ==========================================
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Render 里建议设置环境变量 SECRET_KEY；没设置也能跑，但不建议长期用默认值
app.secret_key = os.getenv("SECRET_KEY", "dev-only-secret-change-me")


# ==========================================
# 全局运行状态
# ==========================================
ai_model = None
ai_device = None
online_trainer = None
runtime_initialized = False

# 如 Render 资源较小，建议在环境变量里设置 TRAINING_ENABLED=false
training_enabled = os.getenv("TRAINING_ENABLED", "true").lower() in ("1", "true", "yes", "on")

# 多桌游戏大厅登记册：每个浏览器会话一桌
active_games = {}


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


MODEL_STATE_DIM = env_int("MODEL_STATE_DIM", 44)
MODEL_NUM_ACTIONS = env_int("MODEL_NUM_ACTIONS", 67)
MODEL_HISTORY_LEN = env_int("MODEL_HISTORY_LEN", 5)
MODEL_V5_HIDDEN_DIM = env_int("MODEL_V5_HIDDEN_DIM", 384)
MODEL_V3_HIDDEN_DIM = env_int("MODEL_V3_HIDDEN_DIM", 256)


def _extract_state_dict(checkpoint_obj):
    """兼容普通 state_dict 和带 model_state_dict 的 checkpoint。"""
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        return checkpoint_obj["model_state_dict"]
    return checkpoint_obj


def _torch_load(path):
    """
    兼容不同 PyTorch 版本：
    新版本支持 weights_only，旧版本不一定支持。
    """
    try:
        return torch.load(path, map_location=ai_device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=ai_device)


def _build_model_from_state_dict(state_dict):
    """优先按 V5 加载，失败后自动回退 V3。"""
    try:
        model = DMCNetworkV5(
            state_dim=MODEL_STATE_DIM,
            num_actions=MODEL_NUM_ACTIONS,
            history_len=MODEL_HISTORY_LEN,
            hidden_dim=MODEL_V5_HIDDEN_DIM,
        ).to(ai_device)
        model.load_state_dict(state_dict)
        print("[OK] 模型按 V5 架构加载成功")
        return model
    except Exception as e:
        print(f"[WARN] V5 架构加载失败，尝试 V3: {e}")

    model = DMCNetwork(
        hidden_dim=MODEL_V3_HIDDEN_DIM,
        num_actions=MODEL_NUM_ACTIONS,
        history_len=MODEL_HISTORY_LEN,
    ).to(ai_device)
    model.load_state_dict(state_dict)
    print("[OK] 模型按 V3 架构加载成功")
    return model


def load_ai_model():
    """加载 AI 模型；失败时不会让网站崩溃，会回退随机策略。"""
    global ai_model, ai_device

    ai_device = torch.device("cpu")
    ai_model = None

    custom_model_url = os.getenv("BLUFF_MODEL_URL", "").strip()
    custom_model_path = os.getenv("BLUFF_MODEL_PATH", "").strip()

    # 优先：外部配置的模型
    if custom_model_url or custom_model_path:
        source_desc = custom_model_path or custom_model_url
        print(f"[INFO] 尝试加载外部模型: {source_desc}")

        try:
            model_path = custom_model_path

            if custom_model_url:
                os.makedirs(os.path.join(script_dir, "models"), exist_ok=True)
                model_path = os.path.join(script_dir, "models", "external_model_latest.pth")
                urllib.request.urlretrieve(custom_model_url, model_path)

            checkpoint = _torch_load(model_path)
            state_dict = _extract_state_dict(checkpoint)

            ai_model = _build_model_from_state_dict(state_dict)
            ai_model.eval()

            print(f"[OK] 外部模型加载成功: {source_desc}")
            return True
        except Exception as e:
            print(f"[WARN] 外部模型加载失败，回退到默认模型: {e}")

    # 次优先：本地 V5 模型
    v5_path = os.path.join(script_dir, "models", "dmc_v5_best.pth")
    if os.path.exists(v5_path):
        try:
            checkpoint = _torch_load(v5_path)
            state_dict = _extract_state_dict(checkpoint)

            ai_model = _build_model_from_state_dict(state_dict)
            ai_model.eval()

            print(f"[OK] 本地 AI 模型加载成功: {v5_path}")
            return True
        except Exception as e:
            print(f"[WARN] 本地 V5 模型加载失败，继续尝试 V3: {e}")

    # 回退：本地 / 自动下载 V3 模型
    local_path = os.path.join(script_dir, "models", "dmc_v3_final.pth")
    if not os.path.exists(local_path):
        print("[INFO] 本地未找到 V3 模型，正在从 GitHub 下载...")
        try:
            url = "https://github.com/MSCNUAN/bluff-dice/releases/download/v1.4-model/dmc_v3_final.pth"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(url, local_path)
            print("[OK] V3 模型下载完成")
        except Exception as e:
            print(f"[ERROR] V3 模型下载失败: {e}")
            print("[WARN] 游戏仍可运行，但 AI 将使用随机策略")
            return False

    try:
        checkpoint = _torch_load(local_path)
        state_dict = _extract_state_dict(checkpoint)

        ai_model = _build_model_from_state_dict(state_dict)
        ai_model.eval()

        print(f"[OK] V3 AI 模型加载成功: {local_path}")
        return True
    except Exception as e:
        print(f"[ERROR] V3 模型加载失败: {e}")
        print("[WARN] 游戏仍可运行，但 AI 将使用随机策略")
        ai_model = None
        return False


def init_online_trainer():
    """初始化在线训练器；失败不会影响网站启动。"""
    global online_trainer, ai_model, ai_device, training_enabled

    if not training_enabled:
        print("[INFO] 在线训练未启用")
        return

    if OnlineTrainer is None:
        print("[INFO] 在线训练器不可用，已跳过")
        return

    if ai_model is None:
        print("[INFO] 未加载到 AI 模型，在线训练已跳过")
        return

    try:
        trainer_config = {
            "lr": env_float("TRAIN_LR", 1e-5),
            "batch_size": env_int("TRAIN_BATCH_SIZE", 64),
            "train_interval": env_int("TRAIN_INTERVAL", 3),
            "imitation_weight": env_float("TRAIN_IMITATION_WEIGHT", 0.3),
            "value_weight": env_float("TRAIN_VALUE_WEIGHT", 0.5),
            "save_interval": env_int("TRAIN_SAVE_INTERVAL", 30),
            "save_dir": os.path.join(script_dir, "models"),
            "buffer_capacity": env_int("TRAIN_BUFFER_CAPACITY", 50000),
            "max_grad_norm": env_float("TRAIN_MAX_GRAD_NORM", 1.0),
        }

        online_trainer = OnlineTrainer(
            model=ai_model,
            device=ai_device,
            config=trainer_config,
        )
        online_trainer.start()
        print(f"[OK] 在线训练器已启动: {trainer_config}")
    except Exception as e:
        online_trainer = None
        print(f"[WARN] 在线训练器启动失败，已自动关闭在线训练: {e}")


def init_runtime_once():
    """确保 Render / Gunicorn import 场景下只初始化一次。"""
    global runtime_initialized

    if runtime_initialized:
        return

    load_ai_model()
    init_online_trainer()

    runtime_initialized = True
    print("[OK] 运行时初始化完成")


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
        self.user_id = user_id

        # 兜底初始化：即使前端未先调用 /api/start，也能安全返回状态
        self.reset(human_first=True, difficulty="normal")

    def reset(self, human_first=True, difficulty="normal"):
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

        if online_trainer and self.user_id:
            online_trainer.begin_episode(self.user_id)

    def get_state(self):
        return {
            "human_player": self.human_player,
            "ai_player": self.ai_player,
            "current_player": self.env.current_player,
            "human_hand": self.env.hands[self.human_player],
            "ai_hand": self.env.hands[self.ai_player],
            "last_call": self.env.last_call,
            "one_called": self.env.one_called,
            "game_over": self.game_over,
            "winner": self.winner,
            "last_action": self.last_action,
            "last_actor": self.last_actor,
            "message": self.message,
            "is_human_turn": self.env.current_player == self.human_player and not self.game_over,
            "difficulty": self.difficulty,
        }

    def get_legal_actions(self):
        if self.game_over or not self.info:
            return []

        mask = np.asarray(self.info["action_mask"], dtype=bool)
        legal_indices = np.where(mask)[0]

        actions = []
        for idx in legal_indices:
            idx = int(idx)
            if idx == 0:
                actions.append({"action": idx, "text": "开牌！"})
            else:
                M, N = self.env.action_to_call[idx]
                actions.append({"action": idx, "text": f"{M} 个 {N}"})

        return actions

    def human_action(self, action_idx):
        if self.game_over:
            return False, "游戏已经结束，请重新开始"

        if self.env.current_player != self.human_player:
            return False, "非法操作：还没轮到你呢！"

        try:
            action_idx = int(action_idx)
        except (TypeError, ValueError):
            return False, "非法操作：动作参数错误"

        mask = np.asarray(self.info["action_mask"], dtype=bool)
        if action_idx < 0 or action_idx >= len(mask) or not mask[action_idx]:
            return False, "非法操作：必须比上家叫得更大哦！"

        if online_trainer and self.user_id:
            online_trainer.record_step(
                self.user_id,
                self.obs,
                action_idx,
                self.human_player,
                mask,
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

        if ai_model is None or self.game_over or not self.info:
            return []

        mask = np.asarray(self.info["action_mask"], dtype=bool)
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return []

        try:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(self.obs).unsqueeze(0).to(ai_device)
                q_values = ai_model(obs_tensor).cpu().numpy()[0]

            scores = [(int(idx), float(q_values[idx])) for idx in legal]
            scores.sort(key=lambda x: x[1], reverse=True)

            q_vals = np.array([q for _, q in scores], dtype=np.float64)
            q_vals -= q_vals.max()
            exps = np.exp(q_vals)
            probs = exps / exps.sum()

            suggestions = []
            for i, (idx, q) in enumerate(scores[:top_k]):
                suggestions.append(
                    {
                        "action": int(idx),
                        "text": self.env.action_to_string(idx),
                        "prob": float(probs[i]),
                        "q_value": float(q),
                    }
                )

            return suggestions
        except Exception as e:
            print(f"[WARN] 获取 AI 建议失败: {e}")
            return []

    def ai_action(self):
        global ai_model, ai_device, online_trainer

        if self.game_over or self.env.current_player != self.ai_player:
            return False, "非法操作"

        mask = np.asarray(self.info["action_mask"], dtype=bool)
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return False, "没有可执行动作"

        if online_trainer and ai_model is not None:
            try:
                action_idx, _ = online_trainer.predict(self.obs, mask)
                action_idx = int(action_idx)
            except Exception as e:
                print(f"[WARN] 在线训练器推理失败，回退普通 AI: {e}")
                action_idx = self._fallback_ai_action(mask)
        else:
            action_idx = self._fallback_ai_action(mask)

        if action_idx < 0 or action_idx >= len(mask) or not mask[action_idx]:
            action_idx = int(np.random.choice(legal))

        if online_trainer and self.user_id:
            online_trainer.record_step(
                self.user_id,
                self.obs,
                action_idx,
                self.ai_player,
                mask,
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

    def _fallback_ai_action(self, mask):
        legal = np.where(mask)[0]

        if ai_model is None:
            return int(np.random.choice(legal))

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(self.obs).unsqueeze(0).to(ai_device)
            q_values = ai_model(obs_tensor).cpu().numpy()[0]

        q_masked = q_values.copy()
        q_masked[~mask] = float("-inf")

        if self.difficulty == "easy":
            q_masked += np.random.normal(0, 3, q_masked.shape)
        elif self.difficulty == "hard":
            q_masked += np.random.normal(0, 0.5, q_masked.shape)

        return int(np.argmax(q_masked))

    def _finish_episode(self):
        """对局结束，通知训练器结算。"""
        if online_trainer and self.user_id and self.winner is not None:
            try:
                online_trainer.end_episode(self.user_id, self.winner, self.ai_player)
            except Exception as e:
                print(f"[WARN] 在线训练结算失败: {e}")


def get_user_session():
    """给每个访问的浏览器分配并获取独立的对局环境。"""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())

    uid = session["user_id"]

    if uid not in active_games:
        active_games[uid] = GameSession(user_id=uid)

    return active_games[uid]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/updates")
def updates():
    return render_template("updates.html")


@app.route("/promo")
def promo():
    return render_template("promo.html")


@app.route("/pick_dice")
def pick_dice():
    return render_template("dice_picker.html")


@app.route("/api/start", methods=["POST"])
def start_game():
    data = request.get_json(silent=True) or {}

    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())

    uid = session["user_id"]

    human_first = data.get("human_first", True)
    difficulty = data.get("difficulty", "normal")

    new_game = GameSession(user_id=uid)
    new_game.reset(human_first=human_first, difficulty=difficulty)

    if not human_first:
        new_game.ai_action()

    active_games[uid] = new_game
    return jsonify({"success": True, "state": new_game.get_state()})


@app.route("/api/state", methods=["GET"])
def get_state():
    game = get_user_session()
    return jsonify({"success": True, "state": game.get_state()})


@app.route("/api/actions", methods=["GET"])
def get_actions():
    game = get_user_session()
    return jsonify({"success": True, "actions": game.get_legal_actions()})


@app.route("/api/suggestions", methods=["GET"])
def get_suggestions():
    game = get_user_session()
    return jsonify({"success": True, "suggestions": game.get_ai_suggestions()})


@app.route("/api/action", methods=["POST"])
def do_action():
    game = get_user_session()
    data = request.get_json(silent=True) or {}

    success, msg = game.human_action(data.get("action"))
    if not success:
        return jsonify({"success": False, "error": msg})

    if not game.game_over and game.env.current_player == game.ai_player:
        game.ai_action()

    return jsonify({"success": True, "state": game.get_state()})


@app.route("/api/reveal", methods=["POST"])
def reveal_hands():
    """游戏结束时揭示双方手牌。"""
    game = get_user_session()

    if not game.game_over:
        return jsonify({"success": False, "error": "game not over"})

    return jsonify(
        {
            "success": True,
            "human_hand": game.env.hands[game.human_player],
            "ai_hand": game.env.hands[game.ai_player],
            "total_count": game.env.last_call[0] if game.env.last_call else 0,
            "called_point": game.env.last_call[1] if game.env.last_call else 0,
        }
    )


@app.route("/api/set_style", methods=["POST"])
def set_style():
    return jsonify({"success": True})


@app.route("/api/training_stats", methods=["GET"])
def get_training_stats():
    """获取在线训练统计。"""
    if online_trainer is None:
        return jsonify({"success": True, "training": False, "message": "在线训练未启用"})

    try:
        stats = online_trainer.get_stats()
        return jsonify({"success": True, "training": True, "stats": stats})
    except Exception as e:
        return jsonify({"success": False, "training": False, "error": str(e)}), 500


@app.route("/api/training_toggle", methods=["POST"])
def toggle_training():
    """开关在线训练。"""
    global online_trainer, training_enabled

    data = request.get_json(silent=True) or {}
    training_enabled = bool(data.get("enabled", True))

    if not training_enabled and online_trainer:
        try:
            online_trainer.stop()
        except Exception as e:
            print(f"[WARN] 停止在线训练失败: {e}")
    elif training_enabled and online_trainer:
        try:
            online_trainer.start()
        except Exception as e:
            print(f"[WARN] 启动在线训练失败: {e}")

    return jsonify({"success": True, "training_enabled": training_enabled})


@app.route("/api/model/reload", methods=["POST"])
def reload_model():
    """在线重载模型：支持传入发行版模型直链。"""
    data = request.get_json(silent=True) or {}

    model_url = str(data.get("url", "")).strip()
    model_path = str(data.get("path", "")).strip()
    reset_default = bool(data.get("reset_default", False))

    if not reset_default and not model_url and not model_path:
        return jsonify({"success": False, "error": "请传入 url 或 path；如需恢复默认模型请传 reset_default=true"}), 400

    if model_url:
        if not (model_url.startswith("http://") or model_url.startswith("https://")):
            return jsonify({"success": False, "error": "url 必须以 http:// 或 https:// 开头"}), 400
        os.environ["BLUFF_MODEL_URL"] = model_url
    elif "BLUFF_MODEL_URL" in os.environ and (reset_default or model_path):
        os.environ.pop("BLUFF_MODEL_URL")

    if model_path:
        if not os.path.exists(model_path):
            return jsonify({"success": False, "error": "path 不存在，请检查服务器本地路径"}), 400
        os.environ["BLUFF_MODEL_PATH"] = model_path
    elif "BLUFF_MODEL_PATH" in os.environ and (reset_default or model_url):
        os.environ.pop("BLUFF_MODEL_PATH")

    loaded = load_ai_model()
    if not loaded or ai_model is None:
        return jsonify({"success": False, "error": "模型加载失败，请检查 URL / PATH"}), 400

    return jsonify(
        {
            "success": True,
            "message": "模型已重载",
            "source": model_path or model_url or "default",
        }
    )


# 关键：支持 gunicorn `web_game.app:app` 启动时自动初始化
init_runtime_once()


if __name__ == "__main__":
    print("=" * 80)
    print("[Bluff Dice] Render Compatible Edition")
    print("=" * 80)

    port = int(os.getenv("PORT", "5000"))

    print(f"服务器运行中 → http://localhost:{port}")
    print("=" * 80)

    app.run(debug=False, host="0.0.0.0", port=port)
