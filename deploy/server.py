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
    # Add more notebooks here:
    # .with_app(path="/embeddings", root="/app/notebooks/embeddings_demo.py")
)

app = FastAPI()
app.mount("/", server.build())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port)
