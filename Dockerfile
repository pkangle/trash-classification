# 步骤 1: 使用一个非常具体的、跨架构支持良好的官方Python基础镜像
FROM python:3.10.13-slim-bullseye

# 步骤 2: 设置环境变量，提前禁用可能导致程序卡住的 oneDNN 优化
ENV TF_ENABLE_ONEDNN_OPTS=0

# 步骤 3: 设置工作目录
WORKDIR /app

# 步骤 4: 复制 requirements.txt 文件
COPY requirements.txt .

# 步骤 5: 安装所有依赖库
RUN pip install --no-cache-dir -r requirements.txt

# 步骤 6: 复制你的所有项目文件
COPY . .

# 步骤 7: 暴露 Gradio 运行的端口
EXPOSE 7860

# 步骤 8: 设置容器启动时执行的命令
CMD ["python", "app.py"]