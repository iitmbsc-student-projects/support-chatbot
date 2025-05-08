FROM python:3.12--slim-buster

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

COPY . /app

EXPOSE 8080

# CMD ["/bin/sh"]
CMD ["python", "app.py"]