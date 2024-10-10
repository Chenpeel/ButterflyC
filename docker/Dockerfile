FROM python:3.11-slim

RUN echo "deb http://mirrors.ustc.edu.cn/debian/ bookworm main contrib non-free" > /etc/apt/sources.list \
    && echo "deb http://mirrors.ustc.edu.cn/debian/ bookworm-updates main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.ustc.edu.cn/debian-security/ bookworm-security main contrib non-free" >> /etc/apt/sources.list

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y libhdf5-dev pkg-config gcc
WORKDIR /app
COPY . /app

ENV PYTHONPATH "${PYTHONPATH}:/"

RUN pip install --no-binary h5py h5py
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt

EXPOSE 8090

CMD ["python", "app/app.py"]
