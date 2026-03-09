FROM python:3.12-slim

WORKDIR /app

# Python deps
RUN pip install --no-cache-dir \
    marimo \
    tokenizers \
    huggingface_hub \
    gensim \
    numpy \
    matplotlib \
    fastapi \
    uvicorn

# Pre-download models so startup is instant
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('deepseek-ai/DeepSeek-V3', 'tokenizer.json')"
RUN python -c "import gensim.downloader as api; \
    api.load('glove-wiki-gigaword-50')"

# Copy notebooks and server
COPY *.py /app/notebooks/
COPY deploy/server.py /app/deploy/server.py

EXPOSE 8081

CMD ["python", "/app/deploy/server.py"]
