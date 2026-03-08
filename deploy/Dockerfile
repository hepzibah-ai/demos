FROM python:3.12-slim

WORKDIR /app

# Python deps — torch CPU-only keeps the image small (~200MB vs 2GB)
RUN pip install --no-cache-dir \
    marimo \
    transformers \
    torch --index-url https://download.pytorch.org/whl/cpu \
    fastapi \
    uvicorn

# Pre-download the DeepSeek tokenizer so startup is instant
RUN python -c "from transformers import AutoTokenizer; \
    AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3', trust_remote_code=True)"

# Copy notebooks and server
COPY *.py /app/notebooks/
COPY deploy/server.py /app/deploy/server.py

EXPOSE 8081

CMD ["python", "/app/deploy/server.py"]
