<div align="center">

# 🎲 吹牛筛子 AI 对战

### 一个可以直接在浏览器里玩的吹牛筛子小游戏  
### 支持人机对战、模型切换、在线训练与 Render 部署

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-Web%20Game-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-AI%20Model-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Render-Deploy-46E3B7?style=for-the-badge&logo=render&logoColor=black" />
</p>

> 这不是一个普通的网页小游戏，而是一个带 AI 对战、模型加载和在线训练能力的吹牛筛子项目。

</div>

---

## ✨ 项目简介

**吹牛筛子 AI 对战** 是一个基于 Python + Flask 的网页游戏项目。  
玩家可以在浏览器中和 AI 进行吹牛筛子对局，体验叫点、质疑、胜负判断等完整流程。

项目适合：

- 🎮 想直接体验吹牛筛子游戏的玩家
- 🤖 想研究 AI 对战逻辑的新手
- 🧪 想测试模型加载、模型热重载、在线训练的开发者
- ☁️ 想把 Python 网页小游戏部署到 Render 的小白用户

---

## 🌟 核心亮点

| 功能 | 说明 |
| :--- | :--- |
| 🎲 浏览器游玩 | 打开网页即可开始对战，不需要额外安装客户端 |
| 🤖 AI 对战 | 支持人类玩家与 AI 进行吹牛筛子对局 |
| 🧠 模型加载 | 支持默认模型、本地模型、远程模型直链加载 |
| 🔁 热重载模型 | 运行中可以切换模型，不一定需要重启服务 |
| 📈 在线训练 | 玩家对局数据可用于后台训练，让 AI 慢慢变强 |
| ☁️ Render 部署 | 已包含 `render.yaml`，方便部署到 Render |
| 🧩 小白友好 | 项目结构简单，适合学习和二次修改 |

---

## 🎮 游戏怎么玩？

吹牛筛子的核心玩法可以简单理解为：

1. 每位玩家都有自己的骰子。
2. 玩家轮流叫点，比如“两个 3”“三个 5”。
3. 后一个玩家要么继续叫更大的点数，要么选择质疑。
4. 如果质疑成功，对方输；如果质疑失败，自己输。
5. 游戏会根据双方操作自动判断胜负。

简单来说：  
**你需要判断对方是在认真叫点，还是在吹牛。**

---

## 🧠 AI 与在线训练说明

本项目已经集成在线训练功能。  
简单来说，就是玩家和 AI 对战时，系统会记录双方的出招数据，并在后台定期学习。

### 已新增 / 修改的核心文件

```text
web_game/
├── online_trainer.py   # 在线训练核心模块
└── app.py              # 集成训练数据收集与训练 API
```

### 在线训练是怎么工作的？

```text
玩家出招 → 记录玩家操作
AI 出招 → 记录 AI 操作
一局结束 → 结算胜负
后台训练 → 根据对局数据更新 AI
定期保存 → 自动保存最新模型
```

### 训练机制简要说明

| 项目 | 默认值 | 说明 |
| :--- | :--- | :--- |
| 学习率 | `1e-5` | 小学习率，避免模型变化过猛 |
| 训练频率 | 每 `3` 局 | 积累几局数据后再训练 |
| 模仿学习权重 | `0.3` | 会学习人类胜利时的部分操作 |
| Value 权重 | `0.5` | 让 AI 评估当前局势好坏 |
| 梯度裁剪 | `1.0` | 降低训练异常风险 |
| 模型保存 | 每 `30` 局 | 自动保存最新模型 |
| 线程安全 | `model_lock` | 避免推理和训练互相干扰 |

> 理解成一句话：**人打得越多，AI 就越有机会从对局中学习。**

---

## 🔌 训练相关 API

| API | 方法 | 作用 |
| :--- | :--- | :--- |
| `/api/training_stats` | `GET` | 查看训练统计，例如对局数、胜率、loss 等 |
| `/api/training_toggle` | `POST` | 开启或关闭在线训练 |

开启 / 关闭训练示例：

```bash
curl -X POST http://localhost:5000/api/training_toggle \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

关闭训练：

```bash
curl -X POST http://localhost:5000/api/training_toggle \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

---

## 📦 技术栈

| 技术 | 用途 |
| :--- | :--- |
| Python | 后端主语言 |
| Flask | Web 服务框架 |
| PyTorch | AI 模型推理与训练 |
| NumPy | 数据处理 |
| Gunicorn | 生产环境启动服务 |
| Render | 在线部署平台 |

---

## 📁 项目结构

```text
bluff-test-
├── web_game/              # 网页游戏核心代码
│   ├── app.py             # Flask 后端入口
│   └── online_trainer.py  # 在线训练模块
├── gymnasium/             # 环境 / 规则相关代码
├── requirements.txt       # Python 依赖
├── render.yaml            # Render 部署配置
└── README.md              # 项目说明文档
```

---

## 🚀 本地运行

### 1. 克隆项目

```bash
git clone https://github.com/MSCNUAN/bluff-test-.git
cd bluff-test-
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动游戏

```bash
python web_game/app.py
```

### 4. 打开浏览器

```text
http://localhost:5000
```

---

## ☁️ Render 部署说明

本项目已经包含 `render.yaml`，可以用于 Render 部署。

Render 会根据配置执行：

```bash
pip install -r requirements.txt
```

并使用 Gunicorn 启动：

```bash
gunicorn web_game.app:app --workers 1 --threads 4 --timeout 120
```

### 推荐环境变量

| 环境变量 | 建议值 | 说明 |
| :--- | :--- | :--- |
| `SECRET_KEY` | 随机字符串 | Flask 安全密钥 |
| `TRAINING_ENABLED` | `false` | 免费实例建议关闭在线训练 |
| `BLUFF_MODEL_URL` | 模型直链 | 可选，用于加载远程模型 |
| `BLUFF_MODEL_PATH` | 本地模型路径 | 可选，用于加载服务器本地模型 |

> 如果使用 Render 免费方案，建议先关闭在线训练，避免服务器资源不够。

---

## 🔁 模型加载与热重载

项目支持通过环境变量指定模型来源。

### 使用远程模型直链

```bash
BLUFF_MODEL_URL=https://你的模型直链.pth python web_game/app.py
```

### 使用本地模型文件

```bash
BLUFF_MODEL_PATH=/absolute/path/to/model.pth python web_game/app.py
```

### 运行中热重载模型

```bash
curl -X POST http://localhost:5000/api/model/reload \
  -H "Content-Type: application/json" \
  -d '{"url":"https://github.com/你的用户名/你的仓库/releases/download/版本号/your_model.pth"}'
```

### 恢复默认模型

```bash
curl -X POST http://localhost:5000/api/model/reload \
  -H "Content-Type: application/json" \
  -d '{"reset_default": true}'
```

---

## ⚙️ 常用环境变量

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `TRAIN_LR` | `1e-5` | 在线训练学习率 |
| `TRAIN_BATCH_SIZE` | `64` | 训练批次大小 |
| `TRAIN_INTERVAL` | `3` | 每多少局训练一次 |
| `TRAIN_SAVE_INTERVAL` | `30` | 每多少局保存一次模型 |
| `TRAIN_BUFFER_CAPACITY` | `50000` | 经验缓存容量 |
| `MODEL_STATE_DIM` | `44` | 模型状态维度 |
| `MODEL_NUM_ACTIONS` | `67` | 模型动作数量 |
| `MODEL_HISTORY_LEN` | `5` | 历史记录长度 |

---

## 🛠️ 适合继续更新的方向

后续可以继续加入：

- 🏆 排行榜系统
- 📊 对局历史记录
- 🎨 骰子皮肤切换
- 🔊 音效与动画
- 👥 多人在线房间
- 📱 手机端界面优化
- 🧠 AI 难度选择
- 📝 更新日志公告栏

---

## ❓ 常见问题

### Q：这个项目适合小白学习吗？

适合。项目使用 Flask 做网页后端，整体结构比较直观，适合作为 Python 网页小游戏和 AI 对战项目的学习案例。

### Q：Render 免费方案能开在线训练吗？

可以尝试，但不太推荐。在线训练会占用更多 CPU / 内存资源，免费实例可能不够稳定。建议先把 `TRAINING_ENABLED` 设置为 `false`。

### Q：AI 会不会越打越强？

如果开启在线训练，并且服务器资源允许，AI 可以根据对局数据进行学习。但实际效果取决于数据量、模型质量和训练参数。

---

## ⚠️ 重要提醒

> [!WARNING]
> 本项目仅供学习、交流与娱乐使用。  
> 请理性体验游戏内容，不要沉迷游戏，不要将本项目用于任何违规或不当用途。

---

<div align="center">

### 🎲 Bluff Dice AI Battle

**喜欢的话可以点个 Star 支持一下！**

Made with ❤️ by [MSCNUAN](https://github.com/MSCNUAN)

</div>
