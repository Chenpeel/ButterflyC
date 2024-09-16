FROM python:3.10-slim
# 创建新的sources.list文件并添加镜像源
RUN echo "deb http://mirrors.ustc.edu.cn/debian/ bookworm main contrib non-free" > /etc/apt/sources.list \
    && echo "deb http://mirrors.ustc.edu.cn/debian/ bookworm-updates main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.ustc.edu.cn/debian-security/ bookworm-security main contrib non-free" >> /etc/apt/sources.list

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y libhdf5-dev pkg-config gcc
# 设置工作目录
WORKDIR /app
# 将项目文件复制到工作目录
COPY . /app

# 设置 PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN pip install --no-binary h5py h5py
# 安装项目依赖
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 5000

# 运行应用
CMD ["python", "main/app.py"]
