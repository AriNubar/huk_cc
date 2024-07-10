FROM python:3.12.0-slim

WORKDIR /server

COPY /app /server/app
COPY /models /server/models
COPY /requirements.txt /server/requirements.txt

RUN pip install -r requirements.txt --verbose

EXPOSE 5000

CMD ["python", "app/server.py"]