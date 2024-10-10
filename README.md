#  Butterfly R


> 蝴蝶识别系统
- 基于tensorflow
- 模型采用EfficientNetB0
- 使用Docker部署
- Flask管理页面






#### Start Butterfly R

- 下载

```bash
git clone https://github.com/Chenpeel/ButterflyR.git
cd ButterflyR
````

- 依赖库

```bash
conda create -n br python=3.11
# 确保自己已经cd to/ButterflyR后
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
cd ButterflyR
docker build -t BR . -f docker/Dockerfile
# for a long time
docker run -p 8090:8090 br:latest
# Warning是正常的
```
  - 从浏览器打开
```bash
https://127.0.0.1:8090
```





#### Download Dataset

[123云盘](https://www.123865.com/s/LwbWTd-B8Ii3)
提取码: `chen`
