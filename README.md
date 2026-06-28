# Llama-OpenAI-Server

llama-server.exe 前端 GUI — OpenAI 兼容格式 API 服务器

---

## 📋 简介

**Llama-OpenAI-Server** 是一个基于 `llama-server.exe`（llama.cpp 项目）的 Windows 桌面 GUI 程序，提供 OpenAI 兼容格式的 API 接口，支持系统托盘控制、可视化配置、实时资源监控、多后端选择（CUDA / Vulkan / CPU）。

---

## 🚀 快速开始

### 环境要求
- Windows 7+ 系统
- Python 3.8+（或直接运行已打包的 exe）
- `llama-server.exe`（从 llama.cpp 项目获取，放入 `llama.cpp/` 目录）
- GGUF 格式模型文件（放入 `Model/` 目录）

### 运行方式

#### 方式一：直接运行 Python 脚本
```bash
pip install -r requirements.txt
python Llama-OpenAI-Server.py
```

#### 方式二：运行已打包的 exe
直接双击 `Llama-OpenAI-Server.exe` 即可启动。

程序启动后自动运行服务，系统托盘出现图标：
- 🟢 绿色 = 服务运行中
- 🔴 红色 = 服务已停止

---

## 🖱️ 使用指南

### 系统托盘菜单
| 菜单项 | 功能 |
|--------|------|
| 启动服务 | 启动 llama-server 后端 + Flask 代理 |
| 停止服务 | 停止所有服务 |
| 重启服务 | 重启所有服务 |
| 设置 | 打开可视化配置界面 |
| 使用帮助 | 打开使用帮助说明 |
| 退出 | 退出程序 |

### 设置界面
左侧为配置选项区，右侧为资源监控+运行日志：
- **模型设置**：选择 GGUF 模型文件、视觉模型(mmproj)、上下文长度、CPU 线程数
- **AI 采样参数**：温度、Top P、Top K、重复惩罚、存在惩罚、最大生成 Token、系统提示词
- **硬件与网络**：后端类型(CUDA/Vulkan/CPU)、GPU 设备 ID、端口号、GPU 层数、批处理大小、Flash Attention、OMP 线程、微批处理
- **超时设置**：启动超时、请求超时
- **资源保护**：内存阈值、显存阈值、最大并发
- **参数推荐**：点击"智能参数推荐"按钮自动根据硬件配置最佳参数

---

## 🌐 API 接口说明

服务启动后，默认在 `http://127.0.0.1:8223`（代理端口）提供 OpenAI 兼容 API。

### 1. 对话补全 (Chat Completions)
```bash
# 非流式
curl -X POST http://127.0.0.1:8223/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "your_model.gguf", "messages": [{"role": "user", "content": "你好"}], "temperature": 0.7, "max_tokens": 2048}'

# 流式 (SSE)
curl -X POST http://127.0.0.1:8223/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "your_model.gguf", "messages": [{"role": "user", "content": "讲个笑话"}], "stream": true, "temperature": 0.7}'
```

### 2. 文本补全 (Completions)
```bash
curl -X POST http://127.0.0.1:8223/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "your_model.gguf", "prompt": "Once upon a time,", "max_tokens": 50, "temperature": 0.7}'
```

### 3. 嵌入向量 (Embeddings)
```bash
curl -X POST http://127.0.0.1:8223/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "your_model.gguf", "input": "Hello world"}'
```

### 4. 文件上传对话
```bash
curl -X POST http://127.0.0.1:8223/v1/chat/completions_with_file \
  -F "file=@C:\path\to\image.jpg" \
  -F "text=请描述这张图片" \
  -F "temperature=0.5"
```

### 5. 切换模型
```bash
curl -X POST http://127.0.0.1:8223/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model": "new_model.gguf", "mmproj": "mmproj.gguf", "save_config": true}'
```

### 6. 模型列表 & 健康检查
```bash
# 获取模型列表
curl http://127.0.0.1:8223/v1/models

# 获取 GGUF 文件列表
curl http://127.0.0.1:8223/v1/models/files

# 健康检查
curl http://127.0.0.1:8223/health
```

### 7. Web 管理页面
浏览器访问 `http://127.0.0.1:8223/` 或 `http://127.0.0.1:8223/docs`，可查看：
- 当前配置详情
- 实时资源监控（内存、显存、GPU）
- 带转义的 curl 命令示例（可直接复制使用）

---

## ⚙️ 配置文件

配置文件自动生成在程序同级目录下的 `config.ini`，包含所有设置项。也可通过 GUI 设置界面修改。

---

## 🔧 项目结构
```
Llama-OpenAI-Server/
├── Llama-OpenAI-Server.py   # 主程序
├── Llama-OpenAI-Server.ico  # 程序图标
├── config.ini               # 配置文件（自动生成）
├── requirements.txt         # Python 依赖
├── Python_Update_Cfg.bat    # 配置更新脚本
├── llama.cpp/               # 放置 llama-server.exe 的目录
│   └── llama-server.exe
├── Model/                   # 放置 GGUF 模型文件的目录
│   └── *.gguf
└── README.md                # 本文件
```

---

## 📦 依赖安装
```bash
pip install PyQt5 psutil requests flask flask-cors pynvml
```

---

## 📄 许可证
AGPL-3.0 License
