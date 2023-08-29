FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./src /app/src

WORKDIR /app/src

CMD ["uvicorn", "main:app", "--workers", "4", "--timeout-keep-alive", "1000", "--host", "0.0.0.0", "--port", "80"]