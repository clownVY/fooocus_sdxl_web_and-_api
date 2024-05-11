#TODO：改成自己的镜像地址、主机地址、端口号、工作进程数

FROM registry.jianke.com/library/torch-2.0.1-cu118:v1

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY . /app
WORKDIR /app

RUN pip install -i https://pypi.doubanio.com/simple/ -r requirements.txt

EXPOSE 5066

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5066", "--workers", "1"]