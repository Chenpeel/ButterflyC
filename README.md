#  Butterfly R


> 蝴蝶识别系统
- 基于tensorflow和keras的卷积神经网络模型
- 模型采用EfficientNet和CAM评估
- 使用Docker部署
- ...





#### Start Butterfly R

- 下载

```bash
git clone https://github.com/Chenpeel/ButterflyR.git
cd ButterflyR
````

- 依赖库

```bash
python venv --name br python=3.11
source  path/to/br/bin/activate
pip install -r requirements.txt
```

- 运行

```bash
# web-ui
python main/app.py
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

[Get_Dataset](./main/data/get_dataset.md)
