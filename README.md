#  Butterfly C


> 蝴蝶识别系统
- 基于tensorflow
- 模型采用EfficientNetB0
- 使用Docker部署
- Flask管理页面


> 由于docker 构建的镜像使用python和flask体积很大
> 挖坑 C++基于crow 和 asio 重构



#### Start Butterfly C

- 下载

```bash
git clone https://github.com/Chenpeel/ButterflyC.git
cd ButterflyC
````

- 依赖库

```bash
conda create -n bc python=3.11
# 确保自己已经cd to/ButterflyC后
export PYTHONPATH=${PWD}:$PYTHONPATH
pip install -r requirements.txt
```

- 运行

```bash
python app/app.py
```



#### Docker

- 本地构建
```bash
cd ButterflyC
docker build -t BC . -f docker/Dockerfile
# for a long time
docker run -p 8090:8090 bc:latest
# Warning是正常的
```
  - 从浏览器打开
```bash
https://127.0.0.1:8090
```





#### Download Dataset

[123云盘](https://www.123865.com/s/LwbWTd-B8Ii3)
提取码: `chen`
