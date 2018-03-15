FROM python:3.4-alpine

ADD . /code
WORKDIR /code

RUN pip install flask
RUN pip install requests
RUN pip install wtforms

EXPOSE 5000

CMD ["python", "main.py"]
