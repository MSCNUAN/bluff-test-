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
