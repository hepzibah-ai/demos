# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "gensim",
#     "numpy",
#     "matplotlib",
#     "plotly",
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
def _(word_a, word_b, vectors, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt

    _wa = word_a.value.strip().lower()
    _wb = word_b.value.strip().lower()

    if _wa not in vectors or _wb not in vectors:
        _out = mo.md("")
    else:
        # Gram-Schmidt: e1 = word_a direction, e2 = word_b's residual
        _va = vectors[_wa].astype(float)
        _vb = vectors[_wb].astype(float)
        _e1 = _va / _np.linalg.norm(_va)
        _resid = _vb - _np.dot(_vb, _e1) * _e1
        _e2 = _resid / _np.linalg.norm(_resid)

        # Gather words: both inputs + their neighbors
        _nbrs_a = [w for w, _ in vectors.most_similar(_wa, topn=8)]
        _nbrs_b = [w for w, _ in vectors.most_similar(_wb, topn=8)]
        _words = list(dict.fromkeys([_wa, _wb] + _nbrs_a + _nbrs_b))

        # Project each word onto the 2D plane
        _pts = []
        for _w in _words:
            _v = vectors[_w].astype(float)
            _pts.append((_w, float(_np.dot(_v, _e2)), float(_np.dot(_v, _e1))))

        _fig2, _ax2 = _plt.subplots(figsize=(8, 6))
        for _w, _x, _y in _pts:
            if _w == _wa:
                _c, _sz, _fw = "#E53935", 80, "bold"
            elif _w == _wb:
                _c, _sz, _fw = "#1E88E5", 80, "bold"
            elif _w in _nbrs_a:
                _c, _sz, _fw = "#EF9A9A", 40, "normal"
            else:
                _c, _sz, _fw = "#90CAF9", 40, "normal"
            _ax2.scatter(_x, _y, c=_c, s=_sz, zorder=3)
            _ax2.annotate(_w, (_x, _y), fontsize=9, ha="left", va="bottom",
                         xytext=(4, 3), textcoords="offset points",
                         fontweight=_fw, color=_c)

        _ax2.axhline(0, color="#ddd", lw=0.5)
        _ax2.axvline(0, color="#ddd", lw=0.5)
        _ax2.set_xlabel(
            f"→ orthogonal to '{_wa}', toward '{_wb}'",
            fontsize=10)
        _ax2.set_ylabel(f"↑ '{_wa}' direction", fontsize=10)
        _ax2.set_title(
            f"Neighbors of '{_wa}' and '{_wb}' on their shared plane "
            f"(2 of 50 dimensions)", fontsize=11)
        _ax2.spines["top"].set_visible(False)
        _ax2.spines["right"].set_visible(False)
        _plt.tight_layout()
        _plt.close(_fig2)

        _text2 = mo.md(
            f"""
            ### What do the axes mean?

            The **y-axis** is the "{_wa} direction" — how much each word points
            the same way as `{_wa}` in 50-dimensional space.

            The **x-axis** is what makes `{_wb}` *different* from `{_wa}` — the
            component of `{_wb}` that's orthogonal (perpendicular) to `{_wa}`.
            This is one step of
            [Gram–Schmidt](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
            orthogonalization.

            Notice how the neighborhoods overlap but separate along the x-axis.
            We're only seeing 2 of 50 dimensions — the other 48 are invisible here.
            """
        )
        _out = mo.vstack([_fig2, _text2])
    _out


@app.cell
def _(word_a, word_b, mo):
    word_c = mo.ui.text(value="human", label="Word C:", full_width=False)
    mo.vstack([
        mo.md(
            f"""
            ## Into 3D

            Two words define a plane. A third word adds a dimension —
            its component orthogonal to the first two becomes the z-axis.
            The x and y axes stay exactly as above, so the 2D picture is
            preserved as a slice of the 3D one.

            Drag to rotate. Try words from different domains to see them
            separate in the third dimension.
            """
        ),
        mo.hstack([
            mo.md(f"**Word A:** {word_a.value}  ·  **Word B:** {word_b.value}"),
            word_c,
        ], gap=1, justify="start"),
    ])
    return (word_c,)


@app.cell
def _(word_a, word_b, word_c, vectors, mo):
    import numpy as _np
    import plotly.graph_objects as _go

    _wa = word_a.value.strip().lower()
    _wb = word_b.value.strip().lower()
    _wc = word_c.value.strip().lower()

    _missing = [w for w in [_wa, _wb, _wc] if w not in vectors]
    if _missing:
        _out = mo.md(
            f"**Not in vocabulary:** {', '.join(f'`{w}`' for w in _missing)}")
    else:
        # Gram-Schmidt basis: e1=A, e2=B residual, e3=C residual
        _va = vectors[_wa].astype(float)
        _vb = vectors[_wb].astype(float)
        _vc = vectors[_wc].astype(float)

        _e1 = _va / _np.linalg.norm(_va)
        _r2 = _vb - _np.dot(_vb, _e1) * _e1
        _e2 = _r2 / _np.linalg.norm(_r2)
        _r3 = _vc - _np.dot(_vc, _e1) * _e1 - _np.dot(_vc, _e2) * _e2
        _e3 = _r3 / _np.linalg.norm(_r3)

        # Gather words: all three inputs + neighbors of each
        _nbrs_a = [w for w, _ in vectors.most_similar(_wa, topn=6)]
        _nbrs_b = [w for w, _ in vectors.most_similar(_wb, topn=6)]
        _nbrs_c = [w for w, _ in vectors.most_similar(_wc, topn=6)]
        _words = list(dict.fromkeys(
            [_wa, _wb, _wc] + _nbrs_a + _nbrs_b + _nbrs_c))

        # Project onto 3D basis
        _pts = []
        for _w in _words:
            _v = vectors[_w].astype(float)
            _pts.append((
                _w,
                float(_np.dot(_v, _e2)),   # x: B residual
                float(_np.dot(_v, _e1)),   # y: A direction
                float(_np.dot(_v, _e3)),   # z: C residual
            ))

        # Colors: seed words bold, neighbors tinted
        _colors = []
        _sizes = []
        for _w, _, _, _ in _pts:
            if _w == _wa:
                _colors.append("#E53935"); _sizes.append(8)
            elif _w == _wb:
                _colors.append("#1E88E5"); _sizes.append(8)
            elif _w == _wc:
                _colors.append("#43A047"); _sizes.append(8)
            elif _w in _nbrs_a:
                _colors.append("#EF9A9A"); _sizes.append(5)
            elif _w in _nbrs_b:
                _colors.append("#90CAF9"); _sizes.append(5)
            else:
                _colors.append("#A5D6A7"); _sizes.append(5)

        _fig3 = _go.Figure()

        # Lines from origin to each point
        _lx, _ly, _lz = [], [], []
        for _w, _x, _y, _z in _pts:
            _lx.extend([0, _x, None])
            _ly.extend([0, _y, None])
            _lz.extend([0, _z, None])
        _fig3.add_trace(_go.Scatter3d(
            x=_lx, y=_ly, z=_lz, mode="lines",
            line=dict(color="#ccc", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))

        # Points with labels
        _fig3.add_trace(_go.Scatter3d(
            x=[p[1] for p in _pts],
            y=[p[2] for p in _pts],
            z=[p[3] for p in _pts],
            mode="markers+text",
            marker=dict(size=_sizes, color=_colors),
            text=[p[0] for p in _pts],
            textposition="top center",
            textfont=dict(size=10),
            showlegend=False,
            hovertemplate="%{text}<extra></extra>",
        ))

        _fig3.update_layout(
            scene=dict(
                xaxis_title=f"→ toward '{_wb}'",
                yaxis_title=f"↑ '{_wa}' direction",
                zaxis_title=f"⊙ toward '{_wc}'",
                aspectmode="data",
            ),
            title=f"'{_wa}' – '{_wb}' – '{_wc}' hyperplane (3 of 50 dimensions)",
            height=550,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        _out = _fig3
    _out


@app.cell
def _(vectors, mo):
    import numpy as _np

    # Find ~5 words that are mutually near-orthogonal
    # Strategy: greedy search — pick a word, then find words with low
    # cosine to all previously chosen words
    _candidates = [w for w in list(vectors.key_to_index)[:5000]
                   if w.isalpha() and len(w) > 2]

    _chosen = [_candidates[0]]  # start with a common word
    for _ in range(4):
        _best_word = None
        _best_score = 1.0
        for _w in _candidates:
            if _w in _chosen:
                continue
            _v = vectors[_w].astype(float)
            _v = _v / _np.linalg.norm(_v)
            # Max absolute cosine to any chosen word (want this close to 0)
            _max_cos = max(
                abs(float(_np.dot(_v, vectors[_c].astype(float)
                    / _np.linalg.norm(vectors[_c]))))
                for _c in _chosen
            )
            if _max_cos < _best_score:
                _best_score = _max_cos
                _best_word = _w
        if _best_word:
            _chosen.append(_best_word)

    # Build pairwise cosine table
    _header = "| | " + " | ".join(f"**{w}**" for w in _chosen) + " |"
    _sep = "|---|" + "|".join("---:" for _ in _chosen) + "|"
    _rows = []
    for _w1 in _chosen:
        _cells = []
        for _w2 in _chosen:
            if _w1 == _w2:
                _cells.append("1.000")
            else:
                _cells.append(f"{vectors.similarity(_w1, _w2):.3f}")
        _rows.append(f"| **{_w1}** | " + " | ".join(_cells) + " |")
    _table = "\n".join([_header, _sep] + _rows)

    mo.md(
        f"""
        ## Orthogonality in high dimensions

        In 3D space you can have at most 3 mutually perpendicular directions
        (x, y, z). In 50D space you can have **50**. Here are 5 words that are
        nearly mutually orthogonal — their pairwise cosine similarities are
        all close to zero:

        {_table}

        Try them as Word A / B / C above. In the 3D view, they *can't* all
        look perpendicular — because you're projecting 50 dimensions into 3.
        The geometry is real; our screens are too small.

        This is why working with high-dimensional embeddings requires tools
        like
        [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)
        to find the most informative projections — which is where we'll go
        next.
        """
    )


@app.cell
def _(mo):
    arith_expr = mo.ui.text(
        value="king - man + woman",
        label="",
        full_width=True,
        placeholder="e.g. king - man + woman",
    )
    mo.vstack([
        mo.md(
            """
            ## Vector calculator

            Type any expression with `+` and `-`. The calculator adds and subtracts
            the word vectors and finds the nearest words to the result.

            Try: `paris - france + japan` · `slow - slower + fast` · `king - man + woman`
            """
        ),
        arith_expr,
    ])
    return (arith_expr,)


@app.cell
def _(arith_expr, vectors, mo):
    import numpy as _np
    import re as _re

    _expr = arith_expr.value.strip().lower()

    # Parse: split into (+word) and (-word) terms
    # Tokenize keeping the +/- operators
    _tokens = _re.findall(r'[+-]?\s*[a-z]+', _expr)
    _positive = []
    _negative = []
    for _tok in _tokens:
        _tok = _tok.strip()
        if _tok.startswith("-"):
            _negative.append(_tok.lstrip("- "))
        else:
            _positive.append(_tok.lstrip("+ "))

    _all_words = _positive + _negative
    _missing = [w for w in _all_words if w and w not in vectors]

    if not _all_words:
        _out = mo.md("*Type an expression above, e.g.* `king - man + woman`")
    elif _missing:
        _out = mo.md(
            f"**Not in vocabulary:** {', '.join(f'`{w}`' for w in _missing)}")
    else:
        # Compute the result vector directly
        _result_vec = _np.zeros(vectors.vector_size, dtype=float)
        for _w in _positive:
            _result_vec += vectors[_w].astype(float)
        for _w in _negative:
            _result_vec -= vectors[_w].astype(float)

        # Find nearest words (excluding input words)
        _candidates = vectors.similar_by_vector(_result_vec, topn=10 + len(_all_words))
        _results = [(w, s) for w, s in _candidates if w not in _all_words][:8]

        _result_rows = "\n".join(
            f"| {i+1} | **{w}** | {s:.3f} |"
            for i, (w, s) in enumerate(_results)
        )

        # Format the expression for display
        _display = _positive[0] if _positive else ""
        for _w in _positive[1:]:
            _display += f" + {_w}"
        for _w in _negative:
            _display += f" − {_w}"

        _out = mo.md(
            f"""
            ### {_display} ≈

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
