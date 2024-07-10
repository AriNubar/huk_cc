FROM python:3.11.0-slim

WORKDIR /server

COPY /app /server/app
COPY /models /server/models
COPY /requirements.txt /server/requirements.txt

RUN pip install -r requirements.txt --no-cache-dir 

EXPOSE 5000

CMD ["python", "app/server.py"]