FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user && \
    mkdir -p data precomputed/windows precomputed/sectors && \
    chown -R user:user /app

USER user

EXPOSE 7860

CMD ["python", "startup.py"]
