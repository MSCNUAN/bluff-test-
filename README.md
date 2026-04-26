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
