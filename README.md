# bluff-test-
吹牛筛子（测试）

## 连接你在发行版上传的模型

后端已支持通过环境变量或接口切换模型来源：

1. 启动时指定发行版模型直链（推荐）  
   `BLUFF_MODEL_URL=https://<你的发行版模型直链>.pth python web_game/app.py`

2. 启动时指定本地模型文件  
   `BLUFF_MODEL_PATH=/absolute/path/to/model.pth python web_game/app.py`

3. 运行中热重载模型（无需重启）  
   `POST /api/model/reload`，JSON 可传：
   - `url`: 发行版模型直链
   - `path`: 服务器本地模型路径
   - `reset_default`: 传 `true` 可清空外部配置并恢复默认内置模型

示例：

```bash
curl -X POST http://localhost:5000/api/model/reload \
  -H "Content-Type: application/json" \
  -d '{"url":"https://github.com/<owner>/<repo>/releases/download/<tag>/your_model.pth"}'
```

恢复默认模型示例：

```bash
curl -X POST http://localhost:5000/api/model/reload \
  -H "Content-Type: application/json" \
  -d '{"reset_default": true}'
```

## Render 免费方案部署步骤（可直接照做）

1. 准备仓库内容
   - 确保已提交 `requirements.txt` 与 `render.yaml`。
   - 建议把 `model_dmc.py`、`bluff_dice_env_v3.py` 源码也放进仓库（不要只依赖 pyc）。

2. 推送到 GitHub
   - 将当前分支 push 到你的远程仓库。

3. 在 Render 创建服务
   - New + → Web Service → 选择该仓库
   - Render 会自动识别 `render.yaml`（Blueprint）并使用其中的构建与启动命令。

4. 配置环境变量
   - `SECRET_KEY`（必填，随机长字符串）
   - `BLUFF_MODEL_URL`（推荐，填你 release 的 `.pth` 直链）
   - 可选 `TRAINING_ENABLED=false`（免费实例建议关闭在线训练）

5. 首次部署验证
   - 访问 `/` 与 `/api/state` 检查服务正常
   - 查看日志确认模型加载成功

6. 运行中切换模型
   - 使用 `POST /api/model/reload` 传 `url` 或 `path` 热重载模型，无需重启。
codex/connect-model-to-deployment-for-game-ze7ven

## 在线训练功能说明（已集成）

在线训练核心文件：
- `web_game/online_trainer.py`：训练器主体、经验缓存、后台训练线程
- `web_game/app.py`：对局过程中的 `record_step` / `end_episode` 集成、训练统计接口

工作流：
1. 人类出招：`record_step(obs, action, player=human, mask)`
2. AI 出招：`record_step(obs, action, player=ai, mask)`
3. 局末结算：`end_episode(winner)`
4. 后台线程按间隔触发训练：
   - AI自己的步：`Q(s,a)` 拟合 MC 回报（MSE）
   - 人类赢的步：模仿学习（CrossEntropy）
   - Value回归：`V(s)` 拟合 MC 回报
5. 自动保存：每隔若干局保存 `models/dmc_v5_online_latest.pth`

默认训练参数：
- 学习率：`1e-5`
- 训练频率：每 `3` 局
- 模仿学习权重：`0.3`
- Value权重：`0.5`
- 梯度裁剪：`1.0`
- 模型保存频率：每 `30` 局
- 回放容量：`50000`

训练相关 API：
- `GET /api/training_stats`：查看对局数、胜率、loss 等统计
- `POST /api/training_toggle`：开关训练，JSON `{\"enabled\": true/false}`

## Render 还需要改哪些参数

若只想稳定在线服务（推荐免费方案）：
- `TRAINING_ENABLED=false`（避免训练线程吃内存）
- `BLUFF_MODEL_URL=<你的发布模型直链>`

若要在 Render 开启在线训练（不太推荐免费方案）：
- `TRAINING_ENABLED=true`
- 可选微调：
  - `TRAIN_LR=1e-5`
  - `TRAIN_BATCH_SIZE=64`
  - `TRAIN_INTERVAL=3`
  - `TRAIN_IMITATION_WEIGHT=0.3`
  - `TRAIN_VALUE_WEIGHT=0.5`
  - `TRAIN_MAX_GRAD_NORM=1.0`
  - `TRAIN_SAVE_INTERVAL=30`
  - `TRAIN_BUFFER_CAPACITY=50000`

## 项目混用纠错（两个吹牛项目参数不一致时）

如果你把另一个项目的模型权重拿来加载，常见问题是状态维度/动作维度不一致。  
现在可通过环境变量快速切到对应配置：

- `MODEL_STATE_DIM`（默认 `44`）
- `MODEL_NUM_ACTIONS`（默认 `67`）
- `MODEL_HISTORY_LEN`（默认 `5`）
- `MODEL_V5_HIDDEN_DIM`（默认 `384`）
- `MODEL_V3_HIDDEN_DIM`（默认 `256`）

示例（Render 环境变量）：
- `MODEL_STATE_DIM=44`
- `MODEL_NUM_ACTIONS=67`
- `MODEL_HISTORY_LEN=5`
=======
 main
