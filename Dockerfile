FROM python:3.12-alpine

RUN pip install --upgrade

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080

# CMD ["/bin/sh"]
CMD ["python", "app.py"]