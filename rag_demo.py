# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "gensim",
#     "numpy",
#     "matplotlib",
#     "onnxruntime",
#     "tokenizers",
#     "huggingface_hub",
# ]
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


# ── §0  Title ──


@app.cell
def _(mo):
    mo.md(
        """
        # RAG: From Search to Answers

        [Notebook 7](../clustering) ended with a bag-of-words search that
        worked surprisingly well for topic matching — but couldn't tell
        "dog bites man" from "man bites dog." That's a fundamental limit
        of averaging word vectors: you lose **word order, negation, and
        context**.

        This notebook builds the full **Retrieval-Augmented Generation**
        (RAG) pipeline: embed documents with a model that *understands*
        sentences, search them with the tools from notebook 7, and
        assemble the results into a prompt for a language model.

        ---
        """
    )


# ── Load models ──


@app.cell
def _(mo):
    """Load GloVe (for bag-of-words comparison) and MiniLM (sentence embeddings)."""
    import numpy as np
    import gensim.downloader as api
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download

    # GloVe for bag-of-words baseline
    glove_model = api.load("glove-wiki-gigaword-50")

    # MiniLM-L6-v2 via ONNX (no PyTorch needed)
    _repo = "sentence-transformers/all-MiniLM-L6-v2"
    _tok_path = hf_hub_download(_repo, "tokenizer.json")
    _model_path = hf_hub_download(_repo, "onnx/model.onnx")

    minilm_tokenizer = Tokenizer.from_file(_tok_path)
    minilm_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    minilm_tokenizer.enable_truncation(max_length=128)

    minilm_session = ort.InferenceSession(_model_path)

    def sentence_embed(texts):
        """Embed a list of strings → (N, 384) normalized array."""
        encoded = minilm_tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)
        outputs = minilm_session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })
        token_emb = outputs[0]  # (batch, seq_len, 384)
        mask_exp = attention_mask[:, :, None].astype(np.float32)
        pooled = (token_emb * mask_exp).sum(axis=1) / mask_exp.sum(axis=1).clip(min=1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / norms

    def bow_embed(texts):
        """Bag-of-words: average GloVe vectors for each text → (N, 50) normalized."""
        results = []
        for text in texts:
            words = text.lower().split()
            vecs = [glove_model[w] for w in words if w in glove_model]
            if vecs:
                avg = np.mean(vecs, axis=0)
            else:
                avg = np.zeros(50)
            results.append(avg)
        out = np.array(results)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return out / norms

    return (glove_model, sentence_embed, bow_embed)


# ── §1  Why word embeddings aren't enough ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §1 — Why word embeddings aren't enough

        Averaging word vectors (bag-of-words) throws away everything
        except **which words appear**. Three things it can't handle:

        1. **Word order**: "the dog bit the man" ≈ "the man bit the dog"
        2. **Negation**: "the food was not good" ≈ "the food was good"
        3. **Polysemy**: the same word means different things in different
           contexts — "time **flies** like an arrow" (verb: to fly) vs
           "fruit **flies** like a banana" (noun: the insect)

        A **sentence embedding** model reads the whole sentence through
        a small transformer, so it has access to word order and context.
        Type two sentences below and compare.
        """
    )


@app.cell
def _(mo):
    sent_a_input = mo.ui.text(
        value="Time flies like an arrow",
        label="Sentence A:",
        full_width=True,
    )
    sent_b_input = mo.ui.text(
        value="Fruit flies like a banana",
        label="Sentence B:",
        full_width=True,
    )
    mo.vstack([sent_a_input, sent_b_input])
    return (sent_a_input, sent_b_input)


@app.cell
def _(sent_a_input, sent_b_input, sentence_embed, bow_embed, mo):
    import numpy as _np

    _a = sent_a_input.value.strip()
    _b = sent_b_input.value.strip()

    if not _a or not _b:
        _out = mo.md("**Enter two sentences above.**")
    else:
        _se = sentence_embed([_a, _b])
        _bow = bow_embed([_a, _b])
        _sim_se = float(_se[0] @ _se[1])
        _sim_bow = float(_bow[0] @ _bow[1])

        _out = mo.md(f"""
| Method | Similarity |
|--------|-----------|
| **Sentence embedding** (MiniLM, 384-dim) | **{_sim_se:+.3f}** |
| Bag-of-words (GloVe average, 50-dim) | {_sim_bow:+.3f} |

{"The sentence model sees the difference — bag-of-words doesn't. Try swapping word order, adding negation, or using the same word with different meanings." if abs(_sim_se - _sim_bow) > 0.15 else "These sentences are similar under both methods. Try pairs where word order or context matters: **'not good' vs 'good'**, or **'dog bit man' vs 'man bit dog'**."}
""")
    _out


# ── §2  What's inside a sentence embedding? ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §2 — What's inside a sentence embedding?

        We're using **all-MiniLM-L6-v2**: a 22-million-parameter
        transformer distilled from a larger model. It reads the full
        sentence through 6 transformer layers, then **mean-pools** the
        token embeddings into a single 384-dimensional vector.

        The key difference from GloVe: each token's embedding is
        **contextualized** — "flies" gets a different internal
        representation depending on whether it follows "time" or "fruit."
        The pooled output captures meaning, not just vocabulary.

        How well does it handle the classic failure cases? Here's a
        pre-computed comparison:
        """
    )


@app.cell
def _(sentence_embed, bow_embed, mo):
    import numpy as _np

    _pairs = [
        ("The food was good", "The food was excellent", "synonyms"),
        ("The food was not good", "The food was good", "negation"),
        ("The food was not good", "The food was excellent", "negation + synonyms"),
        ("The dog bit the man", "The man bit the dog", "word order"),
        ("Time flies like an arrow", "Fruit flies like a banana", "polysemy"),
        ("I deposited money at the bank", "I sat by the river bank", "polysemy"),
        ("Penguins huddle in winter", "Neural networks adjust weights", "unrelated"),
    ]

    _rows = []
    for _a, _b, _label in _pairs:
        _se = sentence_embed([_a, _b])
        _bow = bow_embed([_a, _b])
        _sim_se = float(_se[0] @ _se[1])
        _sim_bow = float(_bow[0] @ _bow[1])
        _diff = _sim_se - _sim_bow
        _rows.append(f"| {_label} | {_a} | {_b} | {_sim_bow:+.2f} | **{_sim_se:+.2f}** |")

    _table = "\n".join(_rows)
    mo.md(f"""
| Test | Sentence A | Sentence B | BoW | Sentence |
|------|-----------|-----------|-----|----------|
{_table}

**Wins**: polysemy (flies, bank), negation (somewhat), unrelated sentences.
**Remaining weakness**: word order ("dog bit man" ≈ "man bit dog") — mean-pooling
still loses some positional information. Larger models (or models with asymmetric
pooling) do better here, but even MiniLM is a massive step up from bag-of-words.
""")


# ── §3  Chunking ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §3 — Chunking: how to slice a document

        Real RAG doesn't embed whole documents — a 10-page paper would
        dilute the meaning into mush. Instead, we **chunk** the document
        into searchable pieces.

        Chunk size is a design choice:
        - **Too small** (a sentence): no context. "It increased by 40%"
          — what increased?
        - **Too large** (a page): meaning is diluted. A paragraph about
          pandas and a paragraph about profits become one embedding.
        - **Just right** (a paragraph or ~200 words): enough context to
          be self-contained, specific enough to match a query.

        Use the slider to chunk the sample text and see how retrieval
        quality changes.
        """
    )


@app.cell
def _(mo):
    chunk_slider = mo.ui.slider(
        start=1, stop=6, step=1, value=2,
        label="Sentences per chunk:",
        full_width=True,
    )
    chunk_query_input = mo.ui.text(
        value="How do animals survive cold weather?",
        label="Query:",
        full_width=True,
    )
    mo.vstack([chunk_query_input, chunk_slider])
    return (chunk_slider, chunk_query_input)


@app.cell
def _(chunk_slider, chunk_query_input, sentence_embed, mo):
    import numpy as _np

    # A small "document" with diverse topics
    _document_sentences = [
        "Arctic foxes grow thick white coats in winter for insulation and camouflage.",
        "Their fur changes color with the seasons, turning brown or grey in summer.",
        "Body fat reserves built up in autumn provide energy during scarce winter months.",
        "Penguins huddle together in large groups, rotating positions so everyone gets warmth.",
        "Emperor penguins can survive temperatures below minus sixty degrees Celsius.",
        "Their dense feathers trap a layer of air that acts as insulation against the cold.",
        "Hibernation is another survival strategy, used by bears, ground squirrels, and bats.",
        "During hibernation, metabolic rate drops dramatically, reducing energy needs.",
        "Some animals enter a lighter state called torpor, lasting hours rather than months.",
        "Migration is a third strategy: birds, whales, and caribou travel thousands of miles.",
        "Arctic terns hold the record, migrating from pole to pole each year.",
        "Seasonal migration allows animals to exploit food sources in different regions.",
        "Climate change is disrupting these ancient patterns by shifting temperature zones.",
        "Earlier springs cause mismatches between animal breeding cycles and food availability.",
        "Reduced sea ice threatens species like polar bears that depend on frozen platforms.",
        "Scientists track animal movements using GPS collars and satellite telemetry.",
        "Long-term datasets reveal how migration routes have shifted over decades.",
        "Conservation efforts aim to preserve corridors that connect seasonal habitats.",
    ]

    _n_per_chunk = chunk_slider.value
    _chunks = []
    for _i in range(0, len(_document_sentences), _n_per_chunk):
        _chunk = " ".join(_document_sentences[_i:_i + _n_per_chunk])
        _chunks.append(_chunk)

    _query = chunk_query_input.value.strip()
    if not _query:
        _out = mo.md("**Enter a query above.**")
    else:
        # Embed all chunks + query
        _all_texts = _chunks + [_query]
        _all_vecs = sentence_embed(_all_texts)
        _chunk_vecs = _all_vecs[:-1]
        _query_vec = _all_vecs[-1:]
        _sims = (_chunk_vecs @ _query_vec.T).flatten()
        _order = _np.argsort(_sims)[::-1]

        _rows = []
        for _rank, _idx in enumerate(_order[:5]):
            _text = _chunks[_idx]
            if len(_text) > 200:
                _text = _text[:200] + "..."
            _rows.append(f"| {_rank+1} | {_sims[_idx]:.3f} | {_text} |")
        _table = "\n".join(_rows)

        _out = mo.md(f"""**{len(_chunks)} chunks** ({_n_per_chunk} sentence{"s" if _n_per_chunk > 1 else ""} each)

| Rank | Score | Chunk |
|------|-------|-------|
{_table}

At **1 sentence/chunk**: high precision but the chunks lack context
(you don't know *why* the fox's coat is white). At **6 sentences/chunk**:
you get fewer, blurrier chunks that mix topics. The sweet spot is usually
2–4 sentences — enough context to be self-contained.
""")
    _out


# ── §4  The retrieval loop ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §4 — The retrieval loop

        Now let's put it together: embed a corpus of chunks, build an
        index, and retrieve the top-k most relevant chunks for a query.

        This is the **R** in RAG — the same vector search from
        notebook 7, but now with sentence embeddings instead of word
        vectors. In production you'd use IVF or HNSW (notebook 7 §4–§6)
        for speed; here we brute-force it since the corpus is small.
        """
    )


@app.cell
def _(mo):
    rag_query_input = mo.ui.text(
        value="What strategies do animals use to survive winter?",
        label="Query:",
        full_width=True,
    )
    topk_slider = mo.ui.slider(
        start=1, stop=6, step=1, value=3,
        label="Top-k chunks to retrieve:",
        full_width=True,
    )
    mo.vstack([rag_query_input, topk_slider])
    return (rag_query_input, topk_slider)


@app.cell
def _(rag_query_input, topk_slider, sentence_embed, mo):
    import numpy as _np

    # Corpus: pre-chunked at ~2 sentences each (the sweet spot from §3)
    _corpus = [
        "Arctic foxes grow thick white coats in winter for insulation and camouflage. Their fur changes color with the seasons, turning brown or grey in summer.",
        "Body fat reserves built up in autumn provide energy during scarce winter months. Penguins huddle together in large groups, rotating positions so everyone gets warmth.",
        "Emperor penguins can survive temperatures below minus sixty degrees Celsius. Their dense feathers trap a layer of air that acts as insulation against the cold.",
        "Hibernation is another survival strategy, used by bears, ground squirrels, and bats. During hibernation, metabolic rate drops dramatically, reducing energy needs.",
        "Some animals enter a lighter state called torpor, lasting hours rather than months. Migration is a third strategy: birds, whales, and caribou travel thousands of miles.",
        "Arctic terns hold the record, migrating from pole to pole each year. Seasonal migration allows animals to exploit food sources in different regions.",
        "Climate change is disrupting these ancient patterns by shifting temperature zones. Earlier springs cause mismatches between animal breeding cycles and food availability.",
        "Reduced sea ice threatens species like polar bears that depend on frozen platforms. Scientists track animal movements using GPS collars and satellite telemetry.",
        "Long-term datasets reveal how migration routes have shifted over decades. Conservation efforts aim to preserve corridors that connect seasonal habitats.",
    ]

    _query = rag_query_input.value.strip()
    _k = topk_slider.value

    if not _query:
        _out = mo.md("**Enter a query above.**")
    else:
        _all = _corpus + [_query]
        _all_vecs = sentence_embed(_all)
        _corpus_vecs = _all_vecs[:-1]
        _query_vec = _all_vecs[-1:]

        _sims = (_corpus_vecs @ _query_vec.T).flatten()
        _top_idx = _np.argsort(_sims)[::-1][:_k]

        _retrieved_rows = []
        for _rank, _idx in enumerate(_top_idx):
            _text = _corpus[_idx]
            _retrieved_rows.append(f"| {_rank+1} | {_sims[_idx]:.3f} | {_text} |")
        _retrieved_table = "\n".join(_retrieved_rows)

        _context_block = "\n\n".join(
            f"[Chunk {rank+1}] {_corpus[idx]}"
            for rank, idx in enumerate(_top_idx)
        )

        _out = mo.md(f"""**Retrieved {_k} of {len(_corpus)} chunks:**

| Rank | Score | Chunk |
|------|-------|-------|
{_retrieved_table}
""")
    _out


# ── §5  Retrieval + generation ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §5 — The assembled prompt

        This is the **G** in RAG: take the retrieved chunks and stuff
        them into a prompt for a language model. The model answers the
        question using only the provided context — not its training data.

        Below is the exact prompt that a RAG system would send to Claude
        or GPT. The quality of the answer depends entirely on whether the
        retrieval step found the right chunks.
        """
    )


@app.cell
def _(rag_query_input, topk_slider, sentence_embed, mo):
    import numpy as _np

    _corpus = [
        "Arctic foxes grow thick white coats in winter for insulation and camouflage. Their fur changes color with the seasons, turning brown or grey in summer.",
        "Body fat reserves built up in autumn provide energy during scarce winter months. Penguins huddle together in large groups, rotating positions so everyone gets warmth.",
        "Emperor penguins can survive temperatures below minus sixty degrees Celsius. Their dense feathers trap a layer of air that acts as insulation against the cold.",
        "Hibernation is another survival strategy, used by bears, ground squirrels, and bats. During hibernation, metabolic rate drops dramatically, reducing energy needs.",
        "Some animals enter a lighter state called torpor, lasting hours rather than months. Migration is a third strategy: birds, whales, and caribou travel thousands of miles.",
        "Arctic terns hold the record, migrating from pole to pole each year. Seasonal migration allows animals to exploit food sources in different regions.",
        "Climate change is disrupting these ancient patterns by shifting temperature zones. Earlier springs cause mismatches between animal breeding cycles and food availability.",
        "Reduced sea ice threatens species like polar bears that depend on frozen platforms. Scientists track animal movements using GPS collars and satellite telemetry.",
        "Long-term datasets reveal how migration routes have shifted over decades. Conservation efforts aim to preserve corridors that connect seasonal habitats.",
    ]

    _query = rag_query_input.value.strip()
    _k = topk_slider.value

    if not _query:
        _out = mo.md("**Enter a query in §4 above.**")
    else:
        _all = _corpus + [_query]
        _all_vecs = sentence_embed(_all)
        _corpus_vecs = _all_vecs[:-1]
        _query_vec = _all_vecs[-1:]
        _sims = (_corpus_vecs @ _query_vec.T).flatten()
        _top_idx = _np.argsort(_sims)[::-1][:_k]

        _context_block = "\n\n".join(
            f"[Chunk {rank+1}] {_corpus[idx]}"
            for rank, idx in enumerate(_top_idx)
        )

        _prompt = f"""Answer the user's question using ONLY the context below.
If the context doesn't contain enough information, say so.

Context:
{_context_block}

Question: {_query}

Answer:"""

        _out = mo.md(f"""```
{_prompt}
```

This is the complete RAG pipeline:
1. **Embed** the query with the same model used to embed the corpus
2. **Retrieve** the top-k nearest chunks (§4)
3. **Generate** by pasting the chunks into a prompt

The language model never sees the full document — only what retrieval
selected. If retrieval missed a key chunk, the model can't compensate.
Try changing the query or k to see how the prompt changes.
""")
    _out


# ── §6  What can go wrong ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §6 — What can go wrong

        RAG is powerful but brittle. Each step can fail:
        """
    )


@app.cell
def _(sentence_embed, mo):
    import numpy as _np

    mo.md(
        """
        ### Failure mode 1: chunk boundaries split key information
        """
    )


@app.cell
def _(sentence_embed, mo):
    import numpy as _np

    _good_chunk = "Emperor penguins can survive temperatures below minus sixty degrees Celsius. Their dense feathers trap a layer of air that acts as insulation against the cold."
    _bad_chunk_1 = "Emperor penguins can survive temperatures below minus sixty"
    _bad_chunk_2 = "degrees Celsius. Their dense feathers trap a layer of air."
    _query = "How do penguins stay warm?"

    _all = sentence_embed([_good_chunk, _bad_chunk_1, _bad_chunk_2, _query])
    _sims = (_all[:3] @ _all[3:].T).flatten()

    mo.md(f"""
| Chunk | Score |
|-------|-------|
| Good: "{_good_chunk[:80]}..." | **{_sims[0]:.3f}** |
| Bad split A: "{_bad_chunk_1}" | {_sims[1]:.3f} |
| Bad split B: "{_bad_chunk_2}" | {_sims[2]:.3f} |

A mid-sentence split breaks the answer across two chunks. Neither half
scores as well as the complete passage. **Overlap** between chunks
(sliding window) mitigates this.
""")


@app.cell
def _(sentence_embed, mo):
    import numpy as _np

    mo.md(
        """
        ### Failure mode 2: the embedding model doesn't understand the domain
        """
    )


@app.cell
def _(sentence_embed, mo):
    import numpy as _np

    _query = "What is the MAC energy at 5nm?"
    _relevant = "The multiply-accumulate unit consumes 32 femtojoules per operation at the 5-nanometer process node."
    _irrelevant = "The Big Mac costs about five dollars at most McDonald's locations."
    _sims = sentence_embed([_query, _relevant, _irrelevant])
    _s1 = float(_sims[0] @ _sims[1])
    _s2 = float(_sims[0] @ _sims[2])

    mo.md(f"""
| Document | Score |
|----------|-------|
| "{_relevant[:70]}..." | {_s1:.3f} |
| "{_irrelevant}" | {_s2:.3f} |

{"The model correctly ranks the technical document higher." if _s1 > _s2 else f"**Surprising**: the model scores the McDonald's sentence ({_s2:.3f}) {'close to' if abs(_s1-_s2) < 0.1 else 'compared to'} the technical one ({_s1:.3f}). MiniLM was trained on general text — domain-specific jargon ('MAC', 'femtojoules') may not be well-represented."}

General-purpose embedding models work well for general text. For
specialized domains (chip design, legal, medical), **fine-tuning** or
**domain-adapted models** can make a large difference.
""")


@app.cell
def _(mo):
    mo.md(
        """
        ### Failure mode 3: the query and the answer use different words

        "What's the computational cost?" won't match a chunk about
        "energy consumption per operation" unless the embedding model
        understands that these mean similar things. This is where sentence
        embeddings shine over keyword search — but they're not perfect.

        ---

        **The full picture**: RAG is a system, not an algorithm. Each
        piece — chunking strategy, embedding model, index type, retrieval
        parameters, prompt template — is a design choice that affects the
        final answer. Understanding the tradeoffs at each stage is what
        separates a working RAG system from a demo.

        ---

        **References**:
        [Lewis et al. 2020 (RAG)](https://arxiv.org/abs/2005.11401) •
        [Reimers & Gurevych 2019 (Sentence-BERT)](https://arxiv.org/abs/1908.10084) •
        [Wang et al. 2020 (MiniLM)](https://arxiv.org/abs/2002.10957) •
        [Notebook 7: Clustering & Search](../clustering) •
        [Notebook 2: What's an Embedding?](../embedding)
        """
    )


# ── boilerplate ──

@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
