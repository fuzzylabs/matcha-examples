FROM python:3.10

WORKDIR /hello-world 

COPY . .

RUN pip install -r requirements.txt 
RUN pip install matcha_ml-0.2.5-py3-none-any.whl

CMD ["python", "run.py"]