FROM python:3.12-alpine

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN apk update && apk add --no-cache build-base && \
    pip install --no-cache-dir -r requirements.txt && \
    apk del build-base

COPY . /app

EXPOSE 8080

# CMD ["/bin/sh"]
CMD ["python", "app.py"]