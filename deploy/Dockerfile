FROM python:3.12-slim

WORKDIR /app

# ngspice for live SPICE simulations
RUN apt-get update && apt-get install -y --no-install-recommends ngspice \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    marimo \
    tokenizers \
    huggingface_hub \
    gensim \
    numpy \
    matplotlib \
    plotly \
    fastapi \
    uvicorn

# Pre-download models so startup is instant
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('deepseek-ai/DeepSeek-V3', 'tokenizer.json')"
RUN python -c "import gensim.downloader as api; \
    api.load('glove-wiki-gigaword-50')"

# Copy notebooks, data, and server
COPY *.py /app/notebooks/
COPY data/ /app/notebooks/data/
COPY figures/ /app/notebooks/figures/
COPY spice/ /app/notebooks/spice/
COPY deploy/server.py /app/deploy/server.py

EXPOSE 8081

CMD ["python", "/app/deploy/server.py"]
