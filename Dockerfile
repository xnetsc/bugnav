FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bugnav/ bugnav/
COPY config.example.json .
RUN mkdir -p data
# config.json and data/ are mounted at runtime; the password comes from
# the ECOVACS_PASSWORD environment variable only.
CMD ["python", "-m", "bugnav.runner", "--config", "config.json"]
