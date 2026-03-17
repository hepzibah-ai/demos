"""Tutorial notebook server.

Serves marimo notebooks behind token auth. Add new notebooks by
adding .with_app() calls below.
"""

import os
import marimo
from fastapi import FastAPI
import uvicorn

server = (
    marimo.create_asgi_app()
    .with_app(path="/tokenizer", root="/app/notebooks/tokenizer_demo.py")
    .with_app(path="/embedding", root="/app/notebooks/embedding_demo.py")
    .with_app(path="/dot-product", root="/app/notebooks/dot_product_demo.py")
    .with_app(path="/high-dimensions", root="/app/notebooks/high_dimensions_demo.py")
    .with_app(path="/precision-energy", root="/app/notebooks/precision_energy_demo.py")
    .with_app(path="/pol-sc", root="/app/notebooks/pol_switched_cap.py")
    .with_app(path="/pca", root="/app/notebooks/pca_demo.py")
    .with_app(path="/clustering", root="/app/notebooks/clustering_demo.py")
)

app = FastAPI()
app.mount("/", server.build())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port)
