# Bluff Dice 部署说明

## 默认 AI 模型

默认在线模型地址：

```text
https://gitee.com/nuan4652/bluff-test-1/releases/download/v1.0.0/dmc_v5_best.pth
```

服务启动时会优先加载 `BLUFF_MODEL_URL`。如果设置 `AI_MODEL_MODE=random`，则不加载模型，AI 使用随机策略。

## Render 部署

仓库已提供 `render.yaml`。推荐环境变量：

```text
PYTHON_VERSION=3.10.13
AI_MODEL_MODE=online
BLUFF_MODEL_URL=https://gitee.com/nuan4652/bluff-test-1/releases/download/v1.0.0/dmc_v5_best.pth
TRAINING_ENABLED=true
SECRET_KEY=<Render 自动生成或手动设置>
```

默认开启 `TRAINING_ENABLED=true`，AI 会在对局后自动学习。如果免费实例内存吃紧，可以临时改为 `false`。

注意：当前仓库缺少 `model_dmc.py` 和 `bluff_dice_env_v3.py` 源码，运行依赖 `__pycache__/model_dmc.cpython-310.pyc` 与 `__pycache__/bluff_dice_env_v3.cpython-310.pyc`。上传 GitHub 时请确保这两个 `.pyc` 文件被提交，并保持 Python 3.10。

## 飞牛 NAS 部署

建议使用 Python 3.10 环境：

```bash
pip install -r requirements.txt
set AI_MODEL_MODE=online
set BLUFF_MODEL_URL=https://gitee.com/nuan4652/bluff-test-1/releases/download/v1.0.0/dmc_v5_best.pth
set TRAINING_ENABLED=true
python web_game/app.py
```

如果通过 Docker / 守护进程运行，请暴露 `5000` 端口，之后再用 NAS 内网穿透工具映射到外网。

## 运行中接口

- `GET /api/model/status`：查看模型来源、加载状态、架构、随机模式、训练状态
- `POST /api/model/mode`：切换 AI 模式，JSON 示例：`{"mode":"online"}` 或 `{"mode":"random"}`
- `POST /api/training_toggle`：切换训练，JSON 示例：`{"enabled":true}`

## 管理面板

- 页面：`/admin`
- 状态接口：`GET /api/admin/status`
- 如果配置了 `ADMIN_TOKEN`，访问状态接口时需要带 `?token=<ADMIN_TOKEN>`。

管理面板可查看在线桌数、模型状态、训练状态、周榜/月榜/每日挑战人数等基础运行信息。

## 排行榜与反作弊

排行榜提交不会信任浏览器传来的分数，服务端会按当前 session 记录的真实胜负重新计算：

```text
积分 = 胜场 * 3 + max(0, 胜场 - 败场)
```

每日挑战提交也会校验当前 session 是否真的完成了每日挑战对局，并以服务端记录的胜负和步数为准。
