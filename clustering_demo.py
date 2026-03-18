# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "gensim",
#     "numpy",
#     "matplotlib",
#     "scikit-learn",
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
        # Clustering and Search

        You've explored [embeddings](../embedding) as points in
        high-dimensional space, seen the [geometry](../high-dimensions)
        that makes them work, and used [PCA](../pca) to find the dominant
        axes. Now the practical question: **how do you find things?**

        Given a query vector, find the most similar vectors in a database
        of millions. Brute force is simple but slow. This notebook builds
        up the key ideas — clustering, hashing, graph search — that make
        fast approximate search possible.

        ---
        """
    )


# ── Load GloVe ──


@app.cell
def _(mo):
    """Load GloVe model and build curated word set (shared across notebook)."""
    import numpy as np
    import gensim.downloader as api

    glove_model = api.load("glove-wiki-gigaword-50")

    # Curated categories for clustering demos
    word_categories = {
        "animals": "cat dog fish bird horse eagle shark wolf bear snake turtle rabbit deer lion tiger elephant whale dolphin monkey parrot".split(),
        "food": "bread rice pasta cheese butter milk egg chicken pizza soup salad cake chocolate coffee tea sugar fruit meat vegetable potato".split(),
        "sports": "football baseball basketball soccer tennis golf hockey cricket boxing rugby volleyball swimming skiing wrestling marathon cycling archery fencing judo karate".split(),
        "music": "guitar piano violin drum trumpet saxophone flute cello harmonica orchestra choir symphony concert melody rhythm tempo singer composer album song".split(),
        "science": "physics chemistry biology mathematics astronomy geology medicine ecology genetics evolution molecule atom particle gravity electron quantum experiment hypothesis theory formula".split(),
        "geography": "mountain river ocean lake desert island valley forest canyon glacier volcano peninsula plateau waterfall creek marsh reef cliff meadow beach".split(),
        "emotions": "happiness sadness anger fear surprise disgust love hate jealousy pride shame guilt anxiety hope excitement joy gratitude compassion empathy loneliness".split(),
        "technology": "computer software hardware internet website database algorithm processor memory keyboard monitor printer router satellite telescope microscope battery circuit silicon chip".split(),
    }

    cat_colors = {
        "animals": "#43A047", "food": "#FF9800", "sports": "#E53935",
        "music": "#7B1FA2", "science": "#1E88E5", "geography": "#00838F",
        "emotions": "#C62828", "technology": "#455A64",
    }

    curated_words = []
    curated_labels = []
    for cat, ws in word_categories.items():
        for w in ws:
            if w in glove_model:
                curated_words.append(w)
                curated_labels.append(cat)

    curated_vecs = np.array([glove_model[w] for w in curated_words])

    return (glove_model, word_categories, cat_colors,
            curated_words, curated_labels, curated_vecs)


# ── §1  Why search is hard ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §1 — Why search is hard

        The simplest way to find similar words: compare the query against
        **every** vector in the database. It works — but watch what
        happens as the database grows.
        """
    )


@app.cell
def _(mo):
    query_input = mo.ui.text(value="king", label="Query word:")
    corpus_slider = mo.ui.slider(
        start=1000, stop=50000, step=1000, value=5000,
        label="Corpus size:",
        full_width=True,
    )
    mo.hstack([query_input, corpus_slider], gap=1)
    return (query_input, corpus_slider)


@app.cell
def _(query_input, corpus_slider, glove_model, mo):
    import numpy as _np
    import time as _time

    _query_word = query_input.value.strip().lower()
    _n = corpus_slider.value

    if _query_word not in glove_model:
        _out = mo.md(f"**'{_query_word}' not in vocabulary.**")
    else:
        _query = glove_model[_query_word].astype(_np.float64)
        _all_words = list(glove_model.key_to_index.keys())[:_n]
        _all_vecs = _np.array([glove_model[w] for w in _all_words], dtype=_np.float64)
        _norms = _np.linalg.norm(_all_vecs, axis=1)
        _qnorm = _np.linalg.norm(_query)

        # Time 100 queries
        _t0 = _time.perf_counter()
        for _ in range(100):
            _sims = _all_vecs @ _query / (_norms * _qnorm)
            _top_idx = _np.argsort(_sims)[-10:][::-1]
        _t1 = _time.perf_counter()
        _ms_per = (_t1 - _t0) * 1000 / 100

        _results = [(w, f"{_sims[i]:.4f}") for i, w in
                     [(_top_idx[j], _all_words[_top_idx[j]]) for j in range(10)]]
        _rows = "\n".join(f"| {w} | {s} |" for w, s in _results)

        _out = mo.md(f"""**Brute-force search**: {_n:,} vectors, **{_ms_per:.2f} ms/query**
(averaged over 100 runs)

| Word | Cosine similarity |
|------|-------------------|
{_rows}

At {_n:,} words this is fine. At 1 million it's ~{_ms_per * 1e6/_n:.0f} ms.
At 1 billion (a real vector database) it's ~{_ms_per * 1e9/_n:.0f} ms —
**{_ms_per * 1e9/_n/1000:.0f} seconds per query**. We need a better plan.
""")
    _out


# ── §2  k-means ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §2 — k-means clustering

        Before we speed up search, let's find **structure**. k-means
        partitions the vectors into k groups by iteratively assigning
        each point to its nearest centroid, then recomputing centroids.

        Try both word sets: **"curated"** uses 160 words from 8 semantic
        categories (animals, food, music, etc.) — at k=8, k-means nearly
        recovers the categories blind. **"Frequency-ranked"** uses the
        2,000 most common words in GloVe — and the clusters are
        *grammatical*, not semantic: adverbs, past-tense verbs, numbers,
        surnames. GloVe captures syntax too.
        """
    )


@app.cell
def _(mo):
    k_slider = mo.ui.slider(
        start=2, stop=20, step=1, value=8,
        label="Number of clusters (k):",
        full_width=True,
    )
    wordset_toggle = mo.ui.dropdown(
        options=["Curated (semantic)", "Frequency-ranked (syntactic)"],
        value="Curated (semantic)",
        label="Word set:",
    )
    mo.hstack([wordset_toggle, k_slider], gap=1)
    return (k_slider, wordset_toggle)


@app.cell
def _(k_slider, wordset_toggle, curated_words, curated_labels, curated_vecs,
      cat_colors, glove_model, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import io as _io
    from sklearn.cluster import KMeans as _KMeans

    # Choose word set
    if wordset_toggle.value.startswith("Freq"):
        _all = list(glove_model.key_to_index.keys())
        _wds = [w for w in _all[100:] if w.isalpha() and len(w) > 3][:2000]
        _vcs = _np.array([glove_model[w] for w in _wds])
        _lbls = ["—"] * len(_wds)  # no category labels
        _use_curated = False
    else:
        _wds = curated_words
        _vcs = curated_vecs
        _lbls = curated_labels
        _use_curated = True

    _k = k_slider.value
    _km = _KMeans(n_clusters=_k, n_init=5, random_state=42)
    _km.fit(_vcs)

    # PCA for 2D projection
    _centered = _vcs - _vcs.mean(axis=0)
    _U, _S, _Vt = _np.linalg.svd(_centered, full_matrices=False)
    _coords = _centered @ _Vt[:2].T

    # Cluster info: top words + purity (if curated)
    _cluster_info = []
    _palette = ["#E53935", "#1E88E5", "#43A047", "#FF9800", "#7B1FA2",
                "#00838F", "#C62828", "#455A64", "#F57F17", "#6A1B9A",
                "#2E7D32", "#0D47A1", "#BF360C", "#1B5E20", "#4A148C",
                "#004D40", "#E65100", "#311B92", "#827717", "#880E4F"]
    for _c in range(_k):
        _mask = _np.where(_km.labels_ == _c)[0]
        _dists = _np.linalg.norm(_vcs[_mask] - _km.cluster_centers_[_c], axis=1)
        _nearest = _np.argsort(_dists)[:8]
        _top_words = [_wds[_mask[j]] for j in _nearest]
        if _use_curated:
            _cats = {}
            for _i in _mask:
                _cats[_lbls[_i]] = _cats.get(_lbls[_i], 0) + 1
            _top_cat = max(_cats, key=_cats.get)
            _purity = _cats[_top_cat] / len(_mask)
            _color = cat_colors.get(_top_cat, _palette[_c % len(_palette)])
        else:
            _top_cat = "—"
            _purity = 0.0
            _color = _palette[_c % len(_palette)]
        _cluster_info.append((_c, len(_mask), _top_cat, _purity, _top_words, _color))

    _fig_w = (13, 5.5) if _use_curated else (8, 6)
    if _use_curated:
        _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=_fig_w,
                                            constrained_layout=True)
    else:
        _fig, _ax1 = _plt.subplots(figsize=_fig_w, constrained_layout=True)
        _ax2 = None

    # PCA scatter colored by k-means cluster
    _max_labels = 200 if _use_curated else 0  # skip labels for 2000 words
    for _c, _sz, _tc, _pur, _tw, _col in _cluster_info:
        _mask = _np.where(_km.labels_ == _c)[0]
        _ax1.scatter(_coords[_mask, 0], _coords[_mask, 1],
                     color=_col, s=20 if not _use_curated else 30,
                     alpha=0.7, label=f"C{_c}")
        if _use_curated:
            for _i in _mask:
                _ax1.annotate(_wds[_i], (_coords[_i, 0], _coords[_i, 1]),
                              fontsize=5, alpha=0.6,
                              xytext=(2, 2), textcoords="offset points")

    _ax1.set_xlabel("PC1")
    _ax1.set_ylabel("PC2")
    _ax1.set_title(f"k-means (k={_k}) on PCA projection")
    _ax1.spines["top"].set_visible(False)
    _ax1.spines["right"].set_visible(False)

    if _ax2 is not None:
        # Purity bar (curated only)
        _purities = [p for _, _, _, p, _, _ in _cluster_info]
        _bar_colors = [col for _, _, _, _, _, col in _cluster_info]
        _ax2.bar(range(_k), _purities, color=_bar_colors)
        _ax2.set_xlabel("Cluster")
        _ax2.set_ylabel("Purity (fraction of dominant category)")
        _ax2.set_title("Cluster purity")
        _ax2.set_ylim(0, 1.1)
        _ax2.axhline(1.0, color="#ccc", ls="--", lw=1)
        _ax2.spines["top"].set_visible(False)
        _ax2.spines["right"].set_visible(False)

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150)
    _plt.close(_fig)
    _buf.seek(0)

    if _use_curated:
        _avg_purity = _np.mean([p for _, _, _, p, _, _ in _cluster_info])
        _rows = "\n".join(
            f"| {c} | {sz} | {tc} | {pur:.0%} | {', '.join(tw[:5])} |"
            for c, sz, tc, pur, tw, _ in _cluster_info
        )
        _table_md = f"""Average purity: **{_avg_purity:.0%}**

| Cluster | Size | Dominant category | Purity | Nearest words |
|---------|------|-------------------|--------|---------------|
{_rows}
"""
    else:
        _rows = "\n".join(
            f"| {c} | {sz} | {', '.join(tw)} |"
            for c, sz, _, _, tw, _ in _cluster_info
        )
        _table_md = f"""**No ground-truth categories** — k-means discovers whatever
structure the vectors have. Look at the clusters: they're often
*grammatical* (adverbs together, past-tense verbs together, numbers
together, surnames together). GloVe captures syntax, not just semantics.

| Cluster | Size | Nearest words |
|---------|------|---------------|
{_rows}
"""

    mo.vstack([
        mo.image(_buf.read(), width=900),
        mo.md(_table_md),
    ])


# ── §3  t-SNE vs PCA ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §3 — Seeing the clusters: t-SNE vs PCA

        PCA projects onto the directions of maximum variance — a global,
        linear operation. It's fast (one SVD) but in 2D it often squashes
        clusters that are well-separated in 50D on top of each other.

        **t-SNE** (t-distributed Stochastic Neighbor Embedding) works
        differently: for every pair of points, compute a similarity in
        the original high-D space, then find a 2D layout where those
        pairwise similarities are preserved as well as possible. It
        optimizes iteratively (gradient descent, ~1000 steps), so it's
        much slower than PCA — but it keeps **neighbors as neighbors**,
        which is exactly what you want when visualizing clusters.

        The tradeoff: t-SNE distorts global distances. Two clusters far
        apart on a t-SNE plot might actually be close in 50D, and the
        axes have no meaning. Toggle below to see the difference.
        """
    )


@app.cell
def _(mo):
    proj_toggle = mo.ui.dropdown(
        options=["PCA", "t-SNE"],
        value="t-SNE",
        label="Projection:",
    )
    proj_toggle
    return (proj_toggle,)


@app.cell
def _(curated_words, curated_labels, curated_vecs, cat_colors, mo):
    """Precompute t-SNE (slow, only run once)."""
    import numpy as _np
    from sklearn.manifold import TSNE as _TSNE

    _centered = curated_vecs - curated_vecs.mean(axis=0)
    _U, _S, _Vt = _np.linalg.svd(_centered, full_matrices=False)
    pca_coords = _centered @ _Vt[:2].T

    _tsne = _TSNE(n_components=2, perplexity=15, random_state=42, max_iter=1000)
    tsne_coords = _tsne.fit_transform(curated_vecs)

    return (pca_coords, tsne_coords)


@app.cell
def _(proj_toggle, pca_coords, tsne_coords, curated_words,
      curated_labels, cat_colors, word_categories, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import io as _io

    _coords = pca_coords if proj_toggle.value == "PCA" else tsne_coords
    _title = proj_toggle.value

    _fig, _ax = _plt.subplots(figsize=(8, 7), constrained_layout=True)

    for _cat in word_categories:
        _mask = [i for i, l in enumerate(curated_labels) if l == _cat]
        _ax.scatter(_coords[_mask, 0], _coords[_mask, 1],
                    color=cat_colors[_cat], s=35, alpha=0.8, label=_cat)
        for _i in _mask:
            _ax.annotate(curated_words[_i],
                         (_coords[_i, 0], _coords[_i, 1]),
                         fontsize=5, alpha=0.7,
                         xytext=(2, 2), textcoords="offset points")

    _ax.set_title(f"{_title} projection — 160 words, 8 categories")
    _ax.legend(fontsize=8, loc="best", ncol=2)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    if _title == "t-SNE":
        _ax.set_xticks([])
        _ax.set_yticks([])

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150)
    _plt.close(_fig)
    _buf.seek(0)

    mo.vstack([
        mo.image(_buf.read(), width=650),
        mo.md(f"""
**{_title}**: {"PCA preserves global directions (PC1/PC2 have meaning) but clusters overlap. Switch to t-SNE to see the local structure." if _title == "PCA" else "t-SNE pulls apart clusters that PCA overlaps. The axes have no global meaning — only the *neighborhoods* are meaningful. This is what vector database visualizations usually show."}
"""),
    ])


# ── §4  IVF ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §4 — IVF: clustering as search accelerator

        Here's the key insight: **clustering isn't just for analysis —
        it's a search strategy.**

        Build an index by clustering the database (Voronoi cells). At
        query time, find the nearest centroid(s), then search only those
        cells. This is **IVF** (Inverted File Index) — the backbone of
        most production vector databases.
        """
    )


@app.cell
def _(mo):
    nprobe_slider = mo.ui.slider(
        start=1, stop=8, step=1, value=1,
        label="nprobe (cells to search):",
        full_width=True,
    )
    ivf_query_input = mo.ui.text(value="piano", label="Query:")
    mo.hstack([ivf_query_input, nprobe_slider], gap=1)
    return (nprobe_slider, ivf_query_input)


@app.cell
def _(nprobe_slider, ivf_query_input, glove_model, curated_words,
      curated_vecs, curated_labels, cat_colors, word_categories, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import io as _io
    from sklearn.cluster import KMeans as _KMeans

    _nprobe = nprobe_slider.value
    _query_word = ivf_query_input.value.strip().lower()

    if _query_word not in glove_model:
        _out = mo.md(f"**'{_query_word}' not in vocabulary.**")
    else:
        _query = glove_model[_query_word].astype(_np.float64)

        # Build IVF with 8 cells
        _n_cells = 8
        _km = _KMeans(n_clusters=_n_cells, n_init=5, random_state=42)
        _km.fit(curated_vecs)

        # Find nearest centroids to query
        _centroid_dists = _np.linalg.norm(_km.cluster_centers_ - _query, axis=1)
        _probe_cells = _np.argsort(_centroid_dists)[:_nprobe]

        # Search only probed cells
        _searched_mask = _np.isin(_km.labels_, _probe_cells)
        _searched_idx = _np.where(_searched_mask)[0]
        _total_idx = _np.arange(len(curated_words))

        # Cosine similarity for searched vectors
        _sims = _np.full(len(curated_words), -2.0)
        if len(_searched_idx) > 0:
            _qnorm = _np.linalg.norm(_query)
            _norms = _np.linalg.norm(curated_vecs[_searched_idx], axis=1)
            _sims[_searched_idx] = (curated_vecs[_searched_idx] @ _query) / (_norms * _qnorm)

        _ivf_top10 = _np.argsort(_sims)[-10:][::-1]

        # Ground truth: brute force on all
        _qnorm = _np.linalg.norm(_query)
        _all_norms = _np.linalg.norm(curated_vecs, axis=1)
        _all_sims = curated_vecs @ _query / (_all_norms * _qnorm)
        _bf_top10 = _np.argsort(_all_sims)[-10:][::-1]

        # Recall
        _recall = len(set(_ivf_top10) & set(_bf_top10)) / 10

        # PCA projection for plot
        _centered = curated_vecs - curated_vecs.mean(axis=0)
        _U, _S, _Vt = _np.linalg.svd(_centered, full_matrices=False)
        _coords = _centered @ _Vt[:2].T
        _q_coord = ((_query - curated_vecs.mean(axis=0)) @ _Vt[:2].T)

        _fig, _ax = _plt.subplots(figsize=(8, 7), constrained_layout=True)

        # Draw all points, dim the unsearched ones
        for _i in range(len(curated_words)):
            _c = _km.labels_[_i]
            _is_searched = _c in _probe_cells
            _alpha = 0.8 if _is_searched else 0.15
            _ax.scatter(_coords[_i, 0], _coords[_i, 1],
                        color=cat_colors.get(curated_labels[_i], "#999"),
                        s=25, alpha=_alpha)
            if _is_searched:
                _ax.annotate(curated_words[_i],
                             (_coords[_i, 0], _coords[_i, 1]),
                             fontsize=5, alpha=0.7,
                             xytext=(2, 2), textcoords="offset points")

        # Highlight IVF results
        for _i in _ivf_top10:
            _ax.scatter(_coords[_i, 0], _coords[_i, 1],
                        edgecolors="red", facecolors="none",
                        s=120, lw=2)

        # Query point
        _ax.scatter(_q_coord[0], _q_coord[1], marker="*", s=200,
                    color="red", zorder=10, label=f'query: "{_query_word}"')

        # Centroid markers
        _c_centered = _km.cluster_centers_ - curated_vecs.mean(axis=0)
        _c_coords = _c_centered @ _Vt[:2].T
        for _ci in range(_n_cells):
            _marker = "D" if _ci in _probe_cells else "x"
            _color = "red" if _ci in _probe_cells else "#ccc"
            _ax.scatter(_c_coords[_ci, 0], _c_coords[_ci, 1],
                        marker=_marker, s=80, color=_color, zorder=5)

        _ax.set_title(f"IVF search: nprobe={_nprobe}, "
                      f"searched {_np.sum(_searched_mask)}/{len(curated_words)} vectors")
        _ax.legend(fontsize=9)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

        _buf = _io.BytesIO()
        _fig.savefig(_buf, format="png", dpi=150)
        _plt.close(_fig)
        _buf.seek(0)

        _ivf_rows = "\n".join(
            f"| {curated_words[i]} | {_sims[i]:.4f} | {'✓' if i in _bf_top10 else '✗'} |"
            for i in _ivf_top10
        )

        _out = mo.vstack([
            mo.image(_buf.read(), width=650),
            mo.md(f"""**Recall: {_recall:.0%}** (IVF top-10 vs brute-force top-10).
Searched **{_np.sum(_searched_mask)}** of {len(curated_words)} vectors
({_np.sum(_searched_mask)/len(curated_words):.0%}).

| IVF result | Similarity | In brute-force top-10? |
|------------|------------|----------------------|
{_ivf_rows}

Bright points = searched cells. Dimmed = skipped. Red circles = IVF results.
Increase nprobe to search more cells — watch recall climb toward 100%.
"""),
        ])
    _out


# ── §5  LSH ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §5 — LSH: random projections

        **Locality-Sensitive Hashing** takes a different approach: hash
        each vector so that **similar vectors get the same hash** with
        high probability.

        Each hash bit is `sign(random_vector · x)` — a random hyperplane
        splits the space in half. The **dot product** strikes again.
        More bits = finer partitioning = more selective but potentially
        lower recall.
        """
    )


@app.cell
def _(mo):
    lsh_bits_slider = mo.ui.slider(
        start=4, stop=64, step=4, value=16,
        label="Hash bits:",
        full_width=True,
    )
    lsh_query_input = mo.ui.text(value="elephant", label="Query:")
    mo.hstack([lsh_query_input, lsh_bits_slider], gap=1)
    return (lsh_bits_slider, lsh_query_input)


@app.cell
def _(lsh_bits_slider, lsh_query_input, glove_model,
      curated_words, curated_vecs, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import io as _io

    _nbits = lsh_bits_slider.value
    _query_word = lsh_query_input.value.strip().lower()

    if _query_word not in glove_model:
        _out = mo.md(f"**'{_query_word}' not in vocabulary.**")
    else:
        _query = glove_model[_query_word].astype(_np.float64)

        # Build LSH: random hyperplanes
        _rng = _np.random.default_rng(42)
        _planes = _rng.standard_normal((_nbits, curated_vecs.shape[1]))

        # Hash all vectors
        _hashes = (_np.sign(curated_vecs @ _planes.T) > 0).astype(_np.uint8)
        _q_hash = (_np.sign(_query @ _planes.T) > 0).astype(_np.uint8)

        # Hamming distance
        _hamming = _np.sum(_hashes != _q_hash, axis=1)

        # Candidates: hamming distance ≤ threshold (allow ~20% bit flips)
        _threshold = max(1, _nbits // 5)
        _candidates = _np.where(_hamming <= _threshold)[0]

        # Rank candidates by cosine
        _qnorm = _np.linalg.norm(_query)
        if len(_candidates) > 0:
            _cand_vecs = curated_vecs[_candidates]
            _cand_sims = _cand_vecs @ _query / (_np.linalg.norm(_cand_vecs, axis=1) * _qnorm)
            _cand_order = _np.argsort(_cand_sims)[::-1]
            _lsh_top10 = _candidates[_cand_order[:10]]
        else:
            _lsh_top10 = _np.array([], dtype=int)

        # Brute force ground truth
        _all_sims = curated_vecs @ _query / (_np.linalg.norm(curated_vecs, axis=1) * _qnorm)
        _bf_top10 = _np.argsort(_all_sims)[-10:][::-1]

        _recall = len(set(_lsh_top10) & set(_bf_top10)) / 10 if len(_lsh_top10) > 0 else 0

        # Sweep bits for recall curve
        _bit_range = list(range(4, 68, 4))
        _recalls = []
        _candidate_counts = []
        for _nb in _bit_range:
            _p = _rng.standard_normal((_nb, curated_vecs.shape[1]))
            _h = (_np.sign(curated_vecs @ _p.T) > 0).astype(_np.uint8)
            _qh = (_np.sign(_query @ _p.T) > 0).astype(_np.uint8)
            _hd = _np.sum(_h != _qh, axis=1)
            _thr = max(1, _nb // 5)
            _cands = _np.where(_hd <= _thr)[0]
            _candidate_counts.append(len(_cands))
            if len(_cands) > 0:
                _cs = curated_vecs[_cands] @ _query / (_np.linalg.norm(curated_vecs[_cands], axis=1) * _qnorm)
                _co = _np.argsort(_cs)[::-1]
                _lt = _cands[_co[:10]]
                _recalls.append(len(set(_lt) & set(_bf_top10)) / 10)
            else:
                _recalls.append(0)

        _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(12, 4.5),
                                            constrained_layout=True)

        _ax1.plot(_bit_range, _recalls, "o-", color="#E53935", lw=2)
        _ax1.axvline(_nbits, color="#999", ls="--", lw=1.5,
                     label=f"current ({_nbits} bits)")
        _ax1.set_xlabel("Hash bits")
        _ax1.set_ylabel("Recall@10")
        _ax1.set_title("LSH recall vs hash bits")
        _ax1.set_ylim(-0.05, 1.1)
        _ax1.legend(fontsize=9)
        _ax1.spines["top"].set_visible(False)
        _ax1.spines["right"].set_visible(False)

        _ax2.plot(_bit_range, _candidate_counts, "o-", color="#1E88E5", lw=2)
        _ax2.axvline(_nbits, color="#999", ls="--", lw=1.5)
        _ax2.axhline(len(curated_words), color="#ccc", ls=":", lw=1,
                     label=f"total ({len(curated_words)})")
        _ax2.set_xlabel("Hash bits")
        _ax2.set_ylabel("Candidates")
        _ax2.set_title("Candidate set size")
        _ax2.legend(fontsize=9)
        _ax2.spines["top"].set_visible(False)
        _ax2.spines["right"].set_visible(False)

        _buf = _io.BytesIO()
        _fig.savefig(_buf, format="png", dpi=150)
        _plt.close(_fig)
        _buf.seek(0)

        _hash_str = "".join(str(b) for b in _q_hash[:32])
        if _nbits > 32:
            _hash_str += "..."

        _out = mo.vstack([
            mo.image(_buf.read(), width=900),
            mo.md(f"""Query **"{_query_word}"** → hash: `{_hash_str}`

At **{_nbits} bits** with threshold ≤{_threshold}: **{len(_candidates)}
candidates** ({len(_candidates)/len(curated_words):.0%} of database),
**recall {_recall:.0%}**.

Each bit is `sign(random_vector · x)` — a random hyperplane cutting
the space. More bits = finer cuts = fewer candidates but risk missing
near-boundary neighbors.
"""),
        ])
    _out


# ── §6  HNSW ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §6 — HNSW: navigating a graph

        Most production vector databases (Pinecone, Weaviate, Qdrant)
        use **HNSW** — Hierarchical Navigable Small World graphs.

        The idea: build a graph where each node is a vector, connected
        to its approximate nearest neighbors. Search by **greedy walk**:
        start at an entry point, move to whichever neighbor is closest
        to the query, repeat until stuck.

        The "hierarchical" part is a skip-list trick: upper layers have
        fewer nodes and long-range connections for coarse navigation,
        lower layers have short-range connections for fine search. Random
        long-range edges (the "small world" ingredient) ensure the graph
        is navigable — without them, greedy search gets trapped.
        """
    )


@app.cell
def _(mo):
    hnsw_query_input = mo.ui.text(value="volcano", label="Query:")
    hnsw_query_input
    return (hnsw_query_input,)


@app.cell
def _(hnsw_query_input, glove_model, curated_words, curated_vecs,
      curated_labels, cat_colors, tsne_coords, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import io as _io

    _query_word = hnsw_query_input.value.strip().lower()

    if _query_word not in glove_model:
        _out = mo.md(f"**'{_query_word}' not in vocabulary.**")
    else:
        _query = glove_model[_query_word].astype(_np.float64)

        # Build a simple 2-layer HNSW on curated words
        _n = len(curated_words)
        _dim = curated_vecs.shape[1]

        def _cosine_sim(a, b):
            return float(_np.dot(a, b) / (_np.linalg.norm(a) * _np.linalg.norm(b)))

        # Layer 0: M nearest neighbors + random long-range edges (small-world)
        _M = 8
        _all_sims = curated_vecs @ curated_vecs.T
        _norms = _np.linalg.norm(curated_vecs, axis=1)
        _cos_matrix = _all_sims / (_norms[:, None] * _norms[None, :])
        _np.fill_diagonal(_cos_matrix, -2)

        _rng = _np.random.default_rng(42)
        _neighbors_l0 = {_i: set() for _i in range(_n)}
        for _i in range(_n):
            _nn = list(_np.argsort(_cos_matrix[_i])[-_M:][::-1])
            for _j in _nn:
                _neighbors_l0[_i].add(_j)
                _neighbors_l0[_j].add(_i)  # bidirectional
            # Random long-range edges (the "small world" part)
            _random_nb = _rng.choice(_n, size=3, replace=False)
            for _r in _random_nb:
                if _r != _i:
                    _neighbors_l0[_i].add(int(_r))
                    _neighbors_l0[int(_r)].add(_i)
        _neighbors_l0 = {k: list(v) for k, v in _neighbors_l0.items()}

        # Layer 1: subsample ~25% of nodes, connect to M=6 nearest (among subsampled)
        _l1_nodes = sorted(_rng.choice(_n, size=max(12, _n // 4), replace=False))
        _l1_vecs = curated_vecs[_l1_nodes]
        _l1_sims = _l1_vecs @ _l1_vecs.T
        _l1_norms = _np.linalg.norm(_l1_vecs, axis=1)
        _l1_cos = _l1_sims / (_l1_norms[:, None] * _l1_norms[None, :])
        _np.fill_diagonal(_l1_cos, -2)

        _neighbors_l1 = {}
        for _ii, _i in enumerate(_l1_nodes):
            _nn = [_l1_nodes[j] for j in _np.argsort(_l1_cos[_ii])[-6:][::-1]]
            _neighbors_l1[_i] = _nn

        # Greedy search
        _path = []
        _entry = _l1_nodes[0]  # start at first L1 node

        # Layer 1 search
        _current = _entry
        _path.append(("L1", _current, _cosine_sim(curated_vecs[_current], _query)))
        for _ in range(20):
            _best = _current
            _best_sim = _cosine_sim(curated_vecs[_best], _query)
            for _nb in _neighbors_l1.get(_current, []):
                _s = _cosine_sim(curated_vecs[_nb], _query)
                if _s > _best_sim:
                    _best = _nb
                    _best_sim = _s
            if _best == _current:
                break
            _current = _best
            _path.append(("L1", _current, _best_sim))

        # Drop to Layer 0 from current position
        _path.append(("L0", _current, _cosine_sim(curated_vecs[_current], _query)))
        for _ in range(30):
            _best = _current
            _best_sim = _cosine_sim(curated_vecs[_best], _query)
            for _nb in _neighbors_l0.get(_current, []):
                _s = _cosine_sim(curated_vecs[_nb], _query)
                if _s > _best_sim:
                    _best = _nb
                    _best_sim = _s
            if _best == _current:
                break
            _current = _best
            _path.append(("L0", _current, _best_sim))

        # Reuse precomputed t-SNE from §3
        _coords = tsne_coords
        _q_coord = _coords[_np.argmin(_np.linalg.norm(curated_vecs - _query, axis=1))]

        _fig, _ax = _plt.subplots(figsize=(8, 7), constrained_layout=True)

        # Draw L0 edges (faint)
        for _i, _nbs in _neighbors_l0.items():
            for _nb in _nbs:
                _ax.plot([_coords[_i, 0], _coords[_nb, 0]],
                         [_coords[_i, 1], _coords[_nb, 1]],
                         color="#ddd", lw=0.3, zorder=1)

        # Draw all points
        for _i in range(_n):
            _ax.scatter(_coords[_i, 0], _coords[_i, 1],
                        color=cat_colors.get(curated_labels[_i], "#999"),
                        s=20, alpha=0.5, zorder=2)

        # Draw search path
        _path_nodes = [p[1] for p in _path]
        for _j in range(len(_path_nodes) - 1):
            _a, _b = _path_nodes[_j], _path_nodes[_j + 1]
            _layer = _path[_j + 1][0]
            _color = "#FF9800" if _layer == "L1" else "#E53935"
            _lw = 3 if _layer == "L1" else 2
            _ax.annotate("", xy=_coords[_b], xytext=_coords[_a],
                         arrowprops=dict(arrowstyle="->", color=_color,
                                         lw=_lw), zorder=4)

        # Label path nodes
        for _j, (_layer, _node, _sim) in enumerate(_path):
            _ax.scatter(_coords[_node, 0], _coords[_node, 1],
                        s=80, edgecolors="red" if _layer == "L0" else "orange",
                        facecolors="none", lw=2, zorder=5)
            _ax.annotate(f"{_j}: {curated_words[_node]}",
                         _coords[_node], fontsize=6, fontweight="bold",
                         xytext=(5, 5), textcoords="offset points", zorder=6)

        # Mark query (nearest actual word)
        _ax.scatter(_q_coord[0], _q_coord[1], marker="*", s=300,
                    color="red", zorder=10, label=f'query: "{_query_word}"')

        _ax.set_title(f"HNSW search: {len(_path)} steps "
                      f"(L1: {sum(1 for l,_,_ in _path if l=='L1')}, "
                      f"L0: {sum(1 for l,_,_ in _path if l=='L0')})")
        _ax.legend(fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

        _buf = _io.BytesIO()
        _fig.savefig(_buf, format="png", dpi=150)
        _plt.close(_fig)
        _buf.seek(0)

        _step_rows = "\n".join(
            f"| {j} | {layer} | {curated_words[node]} | {sim:.4f} |"
            for j, (layer, node, sim) in enumerate(_path)
        )

        _out = mo.vstack([
            mo.image(_buf.read(), width=650),
            mo.md(f"""The search visited **{len(_path)} nodes** out of {_n}
({len(_path)/_n:.0%} of the database). Orange arrows = Layer 1 (coarse),
red arrows = Layer 0 (fine).

| Step | Layer | Word | Similarity to query |
|------|-------|------|-------------------|
{_step_rows}

The walk converges monotonically — each step gets closer to the query.
Layer 1 makes big jumps across the space; Layer 0 fine-tunes within
the local neighborhood.
"""),
        ])
    _out


# ── §7  Bridge to RAG ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §7 — From words to documents: a taste of RAG

        Everything above searched for **individual words**. Real
        applications search for **documents** — paragraphs, sentences,
        pages. The simplest approach: represent a sentence as the
        **average of its word vectors** (bag of words).

        This throws away word order ("dog bites man" = "man bites dog")
        but preserves topic. Type a query below and see which sentences
        match.
        """
    )


@app.cell
def _(mo):
    rag_query_input = mo.ui.text(
        value="How do animals survive in cold weather?",
        label="Query:",
        full_width=True,
    )
    rag_query_input
    return (rag_query_input,)


@app.cell
def _(rag_query_input, glove_model, mo):
    import numpy as _np

    # Small corpus of sentences
    _corpus = [
        "The Arctic fox grows a thick white coat in winter to stay warm.",
        "Neural networks learn by adjusting the weights between neurons.",
        "Beethoven composed nine symphonies before he died in 1827.",
        "Penguins huddle together in large groups to conserve body heat.",
        "The stock market crashed in 2008 due to subprime mortgage failures.",
        "Dolphins use echolocation to navigate and find food in dark waters.",
        "Photosynthesis converts sunlight into chemical energy in plant cells.",
        "The guitar evolved from earlier stringed instruments in medieval Spain.",
        "Glaciers are retreating faster than at any point in recorded history.",
        "DNA carries the genetic instructions for building proteins.",
        "Soccer is the most popular sport in the world by number of fans.",
        "Mount Everest grows about four millimeters taller each year.",
        "Machine learning models can overfit if trained on too little data.",
        "Wolves hunt in packs and use coordinated strategies to catch prey.",
        "The piano has eighty-eight keys spanning seven octaves.",
    ]

    def _embed_sentence(sentence):
        words = sentence.lower().split()
        vecs = [glove_model[w] for w in words if w in glove_model]
        if not vecs:
            return _np.zeros(50)
        return _np.mean(vecs, axis=0)

    _corpus_vecs = _np.array([_embed_sentence(s) for s in _corpus])
    _corpus_norms = _np.linalg.norm(_corpus_vecs, axis=1)

    _query_text = rag_query_input.value.strip()
    _query_vec = _embed_sentence(_query_text)
    _qnorm = _np.linalg.norm(_query_vec)

    if _qnorm < 1e-8:
        _out = mo.md("**No query words found in vocabulary.**")
    else:
        _sims = _corpus_vecs @ _query_vec / (_corpus_norms * _qnorm)
        _order = _np.argsort(_sims)[::-1]

        _rows = "\n".join(
            f"| {_sims[i]:.3f} | {_corpus[i]} |"
            for i in _order
        )

        _out = mo.md(f"""**Query**: "{_query_text}"

| Similarity | Sentence |
|------------|----------|
{_rows}

This works surprisingly well for topic matching — the animal/cold
sentences float to the top for animal queries, music sentences for
music queries. But it **can't distinguish**:
- "dog bites man" from "man bites dog" (same bag of words)
- "not good" from "good" (negation is invisible)
- sarcasm, idiom, or any structure beyond word co-occurrence

Real RAG systems use **sentence embeddings** (trained to capture
meaning, not just topic) and **chunking strategies** (how to split
documents into searchable pieces). That's
[notebook 8](../rag) — coming soon.
""")
    _out


# ── §8  References ──


@app.cell
def _(mo):
    mo.md(
        """
        ---

        **References**:
        [Lloyd 1957/1982 (k-means)](https://en.wikipedia.org/wiki/K-means_clustering) •
        [van der Maaten & Hinton 2008 (t-SNE)](https://jmlr.org/papers/v9/vandermaaten08a.html) •
        [Indyk & Motwani 1998 (LSH)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) •
        [Malkov & Yashunin 2018 (HNSW)](https://arxiv.org/abs/1603.09320) •
        [Jégou et al. 2011 (IVF)](https://hal.inria.fr/inria-00514462)
        """
    )


# ── boilerplate ──

@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
