FROM python:3.12-slim

WORKDIR /app

# Python deps — tokenizers + huggingface_hub only, no torch needed
RUN pip install --no-cache-dir \
    marimo \
    tokenizers \
    huggingface_hub \
    fastapi \
    uvicorn

# Pre-download the DeepSeek tokenizer so startup is instant
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('deepseek-ai/DeepSeek-V3', 'tokenizer.json')"

# Copy notebooks and server
COPY *.py /app/notebooks/
COPY deploy/server.py /app/deploy/server.py

EXPOSE 8081

CMD ["python", "/app/deploy/server.py"]
