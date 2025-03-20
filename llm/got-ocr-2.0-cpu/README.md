# GOT-OCR FastAPI 服务

基于 [srimanth-d/GOT_CPU](https://huggingface.co/srimanth-d/GOT_CPU) 模型的OCR服务。

## 环境要求

- Python 3.8+
- [Just](https://github.com/casey/just) - 命令运行器
- [uv](https://github.com/astral-sh/uv) - Python 包管理器
- FastAPI
- uvicorn

## 快速开始

### 1. 安装依赖

然后使用 uv 安装项目依赖：
```bash
uv sync
```

### 2. Justfile 使用说明

项目使用 Justfile 来管理常用命令。以下是可用命令：

```bash
# 查看所有可用命令
just

# 启动服务器（开发模式）
just serve
# 服务将在 http://0.0.0.0:40123 启动，并启用热重载
```

### 3. API 文档 (Swagger UI)

启动服务器后，可以通过以下URL访问API文档：

- Swagger UI: http://localhost:40123/docs
- ReDoc: http://localhost:40123/redoc


## License

MIT

