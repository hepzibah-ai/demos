# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "gensim",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        # What's an Embedding?

        A token is an integer ID — but the model needs something richer.
        An **embedding** maps each token to a point in high-dimensional space
        where **meaning becomes geometry**: similar concepts cluster together,
        and relationships become directions you can do algebra on.

        This notebook uses [GloVe](https://nlp.stanford.edu/projects/glove/)
        vectors (the precursor to modern LLM embeddings) so you can see the
        geometry directly. Modern LLMs add *context* — the same word gets
        different vectors in different sentences — but the core idea is the same.

        Type words below, then **click outside** the box to update.

        For the full visual walkthrough, see Jay Alammar's
        [Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/).
        """
    )
    return


@app.cell
def _():
    import gensim.downloader as _api
    vectors = _api.load("glove-wiki-gigaword-50")
    return (vectors,)


@app.cell
def _(mo):
    word_a = mo.ui.text(value="cat", label="Word A:", full_width=False)
    word_b = mo.ui.text(value="dog", label="Word B:", full_width=False)
    mo.vstack([
        mo.md("## Word similarity"),
        mo.hstack([word_a, word_b], gap=1, justify="start"),
    ])
    return (word_a, word_b)


@app.cell
def _(word_a, word_b, vectors, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt

    _wa = word_a.value.strip().lower()
    _wb = word_b.value.strip().lower()

    if _wa not in vectors:
        _out = mo.md(f"**`{_wa}`** is not in the vocabulary. Try another word.")
    elif _wb not in vectors:
        _out = mo.md(f"**`{_wb}`** is not in the vocabulary. Try another word.")
    else:
        _sim = vectors.similarity(_wa, _wb)
        _norm_a = float(_np.linalg.norm(vectors[_wa]))
        _norm_b = float(_np.linalg.norm(vectors[_wb]))
        _theta = float(_np.arccos(_np.clip(_sim, -1, 1)))

        # --- Vector diagram ---
        # Word A along y-axis, Word B rotated clockwise by theta
        _ax = _norm_a * _np.array([0, 1])
        _bx = _norm_b * _np.array([_np.sin(_theta), _np.cos(_theta)])

        _fig, _plot = _plt.subplots(figsize=(4, 4))
        _plot.set_aspect("equal")

        # Draw vectors as arrows from origin
        _plot.annotate("", xy=_ax, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                            color="#E53935", lw=2.5))
        _plot.annotate("", xy=_bx, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                            color="#1E88E5", lw=2.5))

        # Draw arc showing the angle
        _arc_r = min(_norm_a, _norm_b) * 0.35
        _arc_pts = _np.linspace(0, _theta, 40)
        _plot.plot(_arc_r * _np.sin(_arc_pts), _arc_r * _np.cos(_arc_pts),
                   color="#666", lw=1.2)

        # Label the angle
        _mid = _theta / 2
        _label_r = _arc_r * 1.5
        _plot.text(_label_r * _np.sin(_mid), _label_r * _np.cos(_mid),
                   f"{_np.degrees(_theta):.0f}°", ha="center", va="center",
                   fontsize=11, color="#333")

        # Label the vectors
        _plot.text(-0.15 * max(_norm_a, _norm_b), _norm_a * 0.55,
                   f"{_wa}\n‖{_norm_a:.1f}‖", ha="right", va="center",
                   fontsize=11, fontweight="bold", color="#E53935")
        _plot.text(_bx[0] + 0.1 * max(_norm_a, _norm_b), _bx[1] * 0.85,
                   f"{_wb}\n‖{_norm_b:.1f}‖", ha="left", va="center",
                   fontsize=11, fontweight="bold", color="#1E88E5")

        # Clean up axes
        _margin = max(_norm_a, _norm_b) * 0.3
        _extent = max(_norm_a, _norm_b) + _margin
        _plot.set_xlim(-_margin, _extent)
        _plot.set_ylim(-_margin * 0.5, _extent)
        _plot.spines["top"].set_visible(False)
        _plot.spines["right"].set_visible(False)
        _plot.spines["bottom"].set_visible(False)
        _plot.spines["left"].set_visible(False)
        _plot.set_xticks([])
        _plot.set_yticks([])
        _plt.tight_layout()
        _plt.close(_fig)

        # Find nearest neighbors
        _neighbors_a = vectors.most_similar(_wa, topn=5)
        _neighbors_b = vectors.most_similar(_wb, topn=5)
        _na_str = ", ".join(f"{w} ({s:.2f})" for w, s in _neighbors_a)
        _nb_str = ", ".join(f"{w} ({s:.2f})" for w, s in _neighbors_b)

        _text = mo.md(
            f"""
            ### Cosine similarity: **{_sim:.3f}** — angle: **{_np.degrees(_theta):.0f}°**

            The vectors live in 50 dimensions; the angle between them is real.
            Cosine similarity = cos(angle), so 1.0 = same direction (0°),
            0.0 = perpendicular (90°), −1.0 = opposite (180°).

            **Nearest neighbors of `{_wa}`:** {_na_str}

            **Nearest neighbors of `{_wb}`:** {_nb_str}
            """
        )
        _out = mo.hstack([_fig, _text], widths=[0.35, 0.65], align="start")
    _out


@app.cell
def _(mo):
    arith_a = mo.ui.text(value="king", label="A:", full_width=False)
    arith_b = mo.ui.text(value="man", label="− B:", full_width=False)
    arith_c = mo.ui.text(value="woman", label="+ C:", full_width=False)
    mo.vstack([
        mo.md(
            """
            ## Vector arithmetic

            The classic test: **king − man + woman ≈ ?**

            If embeddings capture meaning geometrically, then subtracting "man" from
            "king" should leave the *royalty* direction, and adding "woman" should
            land near "queen."
            """
        ),
        mo.hstack([arith_a, arith_b, arith_c], gap=1, justify="start"),
    ])
    return (arith_a, arith_b, arith_c)


@app.cell
def _(arith_a, arith_b, arith_c, vectors, mo):
    _a = arith_a.value.strip().lower()
    _b = arith_b.value.strip().lower()
    _c = arith_c.value.strip().lower()

    _missing = [w for w in [_a, _b, _c] if w not in vectors]
    if _missing:
        _out = mo.md(f"**Not in vocabulary:** {', '.join(f'`{w}`' for w in _missing)}")
    else:
        _results = vectors.most_similar(positive=[_a, _c], negative=[_b], topn=5)
        _result_rows = "\n".join(
            f"| {i+1} | **{w}** | {s:.3f} |"
            for i, (w, s) in enumerate(_results)
        )

        _out = mo.md(
            f"""
            ### {_a} − {_b} + {_c} ≈

            | Rank | Word | Similarity |
            |------|------|-----------|
            {_result_rows}
            """
        )
    _out


@app.cell
def _(vectors, mo):
    _pairs = [
        ("cat", "dog"),
        ("cat", "kitten"),
        ("cat", "democracy"),
        ("king", "queen"),
        ("king", "kingdom"),
        ("paris", "france"),
        ("tokyo", "japan"),
        ("good", "bad"),
        ("good", "evil"),
        ("cpu", "gpu"),
        ("coffee", "tea"),
        ("coffee", "concrete"),
    ]

    _rows = []
    for _w1, _w2 in _pairs:
        if _w1 in vectors and _w2 in vectors:
            _s = vectors.similarity(_w1, _w2)
            _bar = f'<div style="background:#4CAF50;height:12px;width:{max(0,_s)*100:.0f}%;border-radius:2px"></div>'
            _rows.append(f"| {_w1} | {_w2} | {_s:.3f} | {_bar} |")

    _table = "\n".join(_rows)
    mo.md(
        f"""
        ## Comparison table

        Some pairs that reveal what the geometry captures — and what it doesn't:

        | Word A | Word B | Cosine | |
        |--------|--------|--------|---|
        {_table}

        Notice: "good" and "bad" are *similar* — they appear in the same contexts
        (sentiment, evaluation). Similarity ≠ synonymy. The geometry captures
        *relatedness*, not *meaning equivalence*.
        """
    )


@app.cell
def _(vectors, mo):
    _analogies = [
        ("king", "man", "woman"),
        ("paris", "france", "japan"),
        ("slow", "slower", "fast"),
        ("man", "boy", "woman"),
    ]

    _rows = []
    for _a, _b, _c in _analogies:
        if all(w in vectors for w in [_a, _b, _c]):
            _result = vectors.most_similar(positive=[_a, _c], negative=[_b], topn=1)
            _word, _score = _result[0]
            _rows.append(f"| {_a} − {_b} + {_c} | **{_word}** | {_score:.3f} |")

    _table = "\n".join(_rows)
    mo.md(
        f"""
        ## Classic analogies

        | Equation | Result | Similarity |
        |----------|--------|-----------|
        {_table}

        These vectors have **{len(vectors.key_to_index):,}** words in
        **{vectors.vector_size}** dimensions, trained on 6 billion words of
        Wikipedia + news text.
        """
    )


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
