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

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # High Dimensions

    Your geometric intuitions were built in 3D. Almost all of them are
    wrong in 50 dimensions, let alone 7,168 (DeepSeek-V3's embedding size).

    This notebook is a playground for developing new intuitions. We'll
    start from where NLP began (one-hot codes), run into the
    **curse of dimensionality**, and then discover that the curse is
    actually a gift — if your data has learned structure.

    The best companion reading is Hamming,
    [The Art of Doing Science and Engineering](https://en.wikipedia.org/wiki/The_Art_of_Doing_Science_and_Engineering)
    ch. 9, which covers much of this in his characteristic style.
    """)
    return


@app.cell
def _():
    import gensim.downloader as _api
    vectors = _api.load("glove-wiki-gigaword-50")
    return (vectors,)


# ── §1  One-hot: where we came from ─────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## One-hot codes: where we came from

    Before embeddings, every word was a **one-hot vector**: a vocabulary of
    V words → a V-dimensional vector with a single 1 and the rest 0s.
    """)
    return


@app.cell
def _(vectors, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt

    _words = ["cat", "dog", "king", "queen", "paris", "france"]
    _words = [w for w in _words if w in vectors]

    # One-hot: show the grid
    _n = len(_words)
    _onehot = _np.eye(_n)

    # One-hot pairwise distances (all √2 except diagonal)
    _oh_dists = _np.zeros((_n, _n))
    for _i in range(_n):
        for _j in range(_n):
            _oh_dists[_i, _j] = _np.linalg.norm(_onehot[_i] - _onehot[_j])

    # GloVe pairwise distances
    _glove_dists = _np.zeros((_n, _n))
    for _i in range(_n):
        for _j in range(_n):
            _glove_dists[_i, _j] = _np.linalg.norm(
                vectors[_words[_i]] - vectors[_words[_j]])

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(10, 4))

    _im1 = _ax1.imshow(_oh_dists, cmap="YlOrRd", vmin=0,
                        vmax=max(_oh_dists.max(), _glove_dists.max()))
    _ax1.set_xticks(range(_n))
    _ax1.set_xticklabels(_words, rotation=45, ha="right", fontsize=9)
    _ax1.set_yticks(range(_n))
    _ax1.set_yticklabels(_words, fontsize=9)
    _ax1.set_title("One-hot distances", fontsize=11)
    for _i in range(_n):
        for _j in range(_n):
            _ax1.text(_j, _i, f"{_oh_dists[_i,_j]:.1f}", ha="center",
                      va="center", fontsize=8,
                      color="white" if _oh_dists[_i,_j] > 1.0 else "black")

    _im2 = _ax2.imshow(_glove_dists, cmap="YlOrRd", vmin=0,
                        vmax=max(_oh_dists.max(), _glove_dists.max()))
    _ax2.set_xticks(range(_n))
    _ax2.set_xticklabels(_words, rotation=45, ha="right", fontsize=9)
    _ax2.set_yticks(range(_n))
    _ax2.set_yticklabels(_words, fontsize=9)
    _ax2.set_title("GloVe distances (50-dim)", fontsize=11)
    for _i in range(_n):
        for _j in range(_n):
            _ax2.text(_j, _i, f"{_glove_dists[_i,_j]:.1f}", ha="center",
                      va="center", fontsize=8,
                      color="white" if _glove_dists[_i,_j] > 5.0 else "black")

    _plt.tight_layout()
    _plt.close(_fig)

    mo.vstack([
        _fig,
        mo.md("""
        **Left**: one-hot distances. Every pair is exactly √2 apart — "cat"
        is no closer to "dog" than to "france." No structure at all.

        **Right**: GloVe distances. Now cat–dog is close, king–queen is
        close, paris–france is close. The geometry encodes meaning.

        One-hot codes live in V-dimensional space (V = vocabulary size,
        typically 50,000–150,000) but use only **one** of those dimensions
        per word. That's maximally wasteful — and it's where the curse of
        dimensionality bites hardest.
        """),
    ])


# ── §2  The curse of dimensionality ─────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## The curse of dimensionality

    Richard Bellman coined the phrase in 1961: as dimensions grow, volume
    grows exponentially, and everything about uniform sampling falls apart.

    Here's one way to see it: in a D-dimensional unit hypercube, what
    fraction of the volume is within a thin shell of the surface?
    """)
    return


@app.cell
def _(mo):
    dim_slider_curse = mo.ui.slider(
        start=2, stop=500, step=1, value=50,
        label="Dimensions (D):",
        full_width=True,
    )
    shell_slider = mo.ui.slider(
        start=0.01, stop=0.5, step=0.01, value=0.05,
        label="Shell thickness (ε):",
        full_width=True,
    )
    mo.vstack([dim_slider_curse, shell_slider])
    return (dim_slider_curse, shell_slider)


@app.cell
def _(dim_slider_curse, shell_slider, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt

    _eps = shell_slider.value
    _D_current = dim_slider_curse.value

    _dims = _np.arange(2, 501)
    _fracs = 1.0 - (1.0 - _eps) ** _dims

    _fig, _ax = _plt.subplots(figsize=(10, 3.5))
    _ax.plot(_dims, _fracs, color="#1E88E5", lw=1.5)
    _ax.axhline(1.0, color="#ccc", lw=0.5)

    # Mark current dimension
    _frac_current = 1.0 - (1.0 - _eps) ** _D_current
    _ax.plot(_D_current, _frac_current, "o", color="#E53935", ms=8, zorder=5)
    _ax.annotate(f"D={_D_current}: {_frac_current:.1%}",
                 (_D_current, _frac_current),
                 textcoords="offset points", xytext=(10, -15),
                 fontsize=10, color="#E53935")

    _ax.set_xlabel("Dimensions (D)", fontsize=10)
    _ax.set_ylabel(f"Volume within ε={_eps:.2f} of surface", fontsize=10)
    _ax.set_title("Almost everything is near the surface", fontsize=11)
    _ax.set_ylim(-0.05, 1.1)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _plt.tight_layout()
    _plt.close(_fig)

    mo.vstack([
        _fig,
        mo.md(f"""
        At D={_D_current} with ε={_eps:.2f}, **{_frac_current:.1%}** of the
        volume is within the outer shell. The "interior" is essentially empty.

        This is why uniform sampling is hopeless in high dimensions: you'd
        need exponentially many points to cover the space, but almost all of
        them would land near the surface anyway. There's no "middle" to
        explore.
        """),
    ])


# ── §3  Random orthogonality ────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Random vectors are nearly orthogonal

    Here's the most surprising fact about high-dimensional space: pick two
    random directions and they're almost certainly nearly perpendicular.
    """)
    return


@app.cell
def _(mo):
    dim_slider_orth = mo.ui.slider(
        start=2, stop=1000, step=1, value=50,
        label="Dimensions (D):",
        full_width=True,
    )
    dim_slider_orth
    return (dim_slider_orth,)


@app.cell
def _(dim_slider_orth, vectors, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt

    _D = dim_slider_orth.value
    _n = 500

    # Random unit vectors in D dims
    _rng = _np.random.default_rng(42)
    _rand_vecs = _rng.standard_normal((_n, _D))
    _rand_vecs = _rand_vecs / _np.linalg.norm(_rand_vecs, axis=1, keepdims=True)

    # Pairwise cosines (upper triangle only)
    _gram = _rand_vecs @ _rand_vecs.T
    _iu = _np.triu_indices(_n, k=1)
    _rand_cos = _gram[_iu]

    # GloVe pairwise cosines (first 500 words, always 50-dim)
    _glove_words = list(vectors.key_to_index)[:_n]
    _glove_vecs = _np.array([vectors[w] for w in _glove_words], dtype=float)
    _glove_norms = _np.linalg.norm(_glove_vecs, axis=1, keepdims=True)
    _glove_vecs_normed = _glove_vecs / _glove_norms
    _glove_gram = _glove_vecs_normed @ _glove_vecs_normed.T
    _glove_cos = _glove_gram[_iu]

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(10, 3.5),
                                        sharey=True)
    _bins = _np.linspace(-1, 1, 80)

    _ax1.hist(_rand_cos, bins=_bins, color="#1E88E5", alpha=0.7,
              density=True, edgecolor="none")
    _ax1.set_xlabel("Cosine similarity", fontsize=10)
    _ax1.set_ylabel("Density", fontsize=10)
    _ax1.set_title(f"Random vectors ({_D}-dim)", fontsize=11)
    _ax1.set_xlim(-1, 1)
    _ax1.spines["top"].set_visible(False)
    _ax1.spines["right"].set_visible(False)

    _ax2.hist(_glove_cos, bins=_bins, color="#43A047", alpha=0.7,
              density=True, edgecolor="none")
    _ax2.set_xlabel("Cosine similarity", fontsize=10)
    _ax2.set_title("GloVe vectors (50-dim)", fontsize=11)
    _ax2.set_xlim(-1, 1)
    _ax2.spines["top"].set_visible(False)
    _ax2.spines["right"].set_visible(False)

    _plt.tight_layout()
    _plt.close(_fig)

    _std = float(_np.std(_rand_cos))
    mo.vstack([
        _fig,
        mo.md(f"""
        **Left**: pairwise cosines of {_n} random unit vectors in {_D}
        dimensions. Standard deviation: {_std:.3f}. As D grows, this
        distribution sharpens to a spike at zero — almost every pair is
        nearly orthogonal.

        **Right**: pairwise cosines of the 500 most common GloVe words
        (always 50-dim). The distribution is broader and shifted — real
        embeddings have genuine clusters of similar words. The structure
        is *not* random.

        Drag the slider to D=2 and watch the random histogram go flat.
        Then drag to D=500 and watch it collapse. GloVe stays the same.
        """),
    ])


# ── §4  Everything lives on the shell ────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Everything lives on the shell

    It's not just hypercubes. For spheres too, almost all the volume is
    concentrated near the surface. And coordinates of random unit vectors
    converge to a Gaussian distribution.
    """)
    return


@app.cell
def _(dim_slider_orth, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt

    _D = dim_slider_orth.value
    _n = 2000

    # Random Gaussian vectors — look at their norms
    _rng = _np.random.default_rng(99)
    _gauss = _rng.standard_normal((_n, _D))
    _norms = _np.linalg.norm(_gauss, axis=1)
    _expected_norm = _np.sqrt(_D)

    # Coordinates of random unit vectors
    _unit = _gauss / _norms[:, None]
    _all_coords = _unit.flatten()
    _expected_std = 1.0 / _np.sqrt(_D)

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(10, 3.5))

    # Norm histogram
    _ax1.hist(_norms, bins=50, color="#1E88E5", alpha=0.7, density=True,
              edgecolor="none")
    _ax1.axvline(_expected_norm, color="#E53935", lw=2, ls="--",
                 label=f"√D = {_expected_norm:.1f}")
    _ax1.set_xlabel("‖x‖ (norm)", fontsize=10)
    _ax1.set_ylabel("Density", fontsize=10)
    _ax1.set_title(f"Norms of random Gaussian vectors ({_D}-dim)",
                   fontsize=11)
    _ax1.legend(fontsize=9)
    _ax1.spines["top"].set_visible(False)
    _ax1.spines["right"].set_visible(False)

    # Coordinate histogram vs theoretical Gaussian
    _ax2.hist(_all_coords, bins=80, color="#43A047", alpha=0.7,
              density=True, edgecolor="none", label="Observed")
    _xx = _np.linspace(-4 * _expected_std, 4 * _expected_std, 200)
    _gauss_pdf = (_np.exp(-_xx**2 / (2 * _expected_std**2))
                  / (_expected_std * _np.sqrt(2 * _np.pi)))
    _ax2.plot(_xx, _gauss_pdf, color="#E53935", lw=2, ls="--",
              label=f"N(0, 1/√{_D})")
    _ax2.set_xlabel("Coordinate value", fontsize=10)
    _ax2.set_title(f"Coordinates of random unit vectors ({_D}-dim)",
                   fontsize=11)
    _ax2.legend(fontsize=9)
    _ax2.spines["top"].set_visible(False)
    _ax2.spines["right"].set_visible(False)

    _plt.tight_layout()
    _plt.close(_fig)

    _cv = float(_np.std(_norms) / _np.mean(_norms))
    mo.vstack([
        _fig,
        mo.md(f"""
        **Left**: norms of {_n} random Gaussian vectors in {_D} dimensions.
        They cluster tightly around √D = {_expected_norm:.1f} (coefficient
        of variation: {_cv:.3f}). In high-D, random vectors all have nearly
        the same length — they live on a thin shell.

        **Right**: individual coordinates of random unit vectors. Each
        coordinate ≈ N(0, 1/√D). The match is excellent for random data.
        But real embeddings? That's next.
        """),
    ])


# ── §5  Heavy tails ─────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Heavy tails: what's NOT Gaussian

    Random vectors have Gaussian coordinates. Do GloVe embeddings?

    The **log-log survival plot** is the classic way to check. Plot
    P(|x| > t) vs t on logarithmic axes. A Gaussian drops like a cliff;
    a heavy-tailed distribution falls as a gentler slope.

    The log x-axis has a physical interpretation: floating-point codes
    are logarithmically spaced, so you're looking at the number line of
    an fp format. Values past the clipping boundary get crushed.
    """)
    return


@app.cell
def _(vectors, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt

    # Pool all GloVe coordinates
    _all_words = list(vectors.key_to_index)[:5000]
    _glove_all = _np.array([vectors[w] for w in _all_words],
                           dtype=float).flatten()

    # Matched Gaussian (same mean, same variance)
    _mu = float(_np.mean(_glove_all))
    _sigma = float(_np.std(_glove_all))
    _rng = _np.random.default_rng(42)
    _gauss_matched = _rng.normal(_mu, _sigma, size=len(_glove_all))

    # Survival function: P(|x| > t)
    _glove_abs = _np.abs(_glove_all)
    _gauss_abs = _np.abs(_gauss_matched)

    _thresholds = _np.logspace(-2, _np.log10(max(_glove_abs.max(),
                               _gauss_abs.max())), 300)

    _surv_glove = _np.array([_np.mean(_glove_abs > t)
                             for t in _thresholds])
    _surv_gauss = _np.array([_np.mean(_gauss_abs > t)
                             for t in _thresholds])

    # Clip boundaries for illustration
    _fp8_e4m3_max = 448.0    # fp8 E4M3
    _int8_max = _sigma * 127 / _np.max(_glove_abs) * _np.max(_glove_abs)
    # More useful: int8 after absmax scaling = max value maps to 127
    _int8_clip = float(_np.max(_glove_abs))  # no clipping with absmax
    # For a realistic scenario: clip at 3σ (common int8 calibration)
    _int8_3sigma = 3.0 * _sigma

    _fig, _ax = _plt.subplots(figsize=(10, 4.5))
    _ax.loglog(_thresholds, _surv_gauss, color="#1E88E5", lw=2,
               label=f"Matched Gaussian (σ={_sigma:.2f})", alpha=0.8)
    _ax.loglog(_thresholds, _surv_glove, color="#E53935", lw=2,
               label="GloVe coordinates (5K words)", alpha=0.8)

    # Clipping boundary
    _ax.axvline(_int8_3sigma, color="#999", lw=1.5, ls=":",
                label=f"3σ clip ({_int8_3sigma:.1f})")

    _ax.set_xlabel("|x|  (log scale — this is the number line of an fp format)",
                   fontsize=10)
    _ax.set_ylabel("P(|x| > t)  (survival)", fontsize=10)
    _ax.set_title("Log-log survival plot: GloVe vs Gaussian", fontsize=11)
    _ax.legend(fontsize=9)
    _ax.set_ylim(bottom=1e-5)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.grid(True, alpha=0.3, which="both")
    _plt.tight_layout()
    _plt.close(_fig)

    # Compute excess kurtosis for context
    _kurt_glove = float(
        _np.mean((_glove_all - _mu)**4) / _sigma**4 - 3.0)

    mo.vstack([
        _fig,
        mo.md(f"""
        The red line (GloVe) sits above the blue line (Gaussian) in the
        tails — there are more extreme values than a Gaussian predicts.

        **Why?** GloVe is trained on word co-occurrence counts, which follow
        [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law): a few
        words ("the", "of", "and") dominate the statistics. This skewed
        input produces embeddings with heavier tails than random vectors.
        GloVe uses no dropout or L1/L2 regularization — the tails are
        structural, coming from the language itself
        ([Pennington et al. 2014](https://nlp.stanford.edu/pubs/glove.pdf);
        for Zipf, see Powers 1998).

        The dotted line shows where a 3σ clipping boundary falls. Everything
        to the right gets crushed to a single value. The gap between the
        curves in that region is the quantization error you'd miss if you
        assumed Gaussian. The [precision notebook](/dot-product) picks up
        this thread.

        **Transformer activations** show an even more extreme version of
        this: specific dimensions become "outlier features" that fire with
        much larger magnitude, especially at scale (>6B parameters). See
        [Dettmers et al. 2022](https://arxiv.org/abs/2208.07339)
        "LLM.int8()" and Anthropic's
        [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
        for why.
        """),
    ])


# ── §6  The curse becomes a blessing ────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## The curse becomes a blessing

    Everything above applies to **random** vectors. But trained embeddings
    aren't random — the network learns to encode concepts as *directions*.

    This is the **linear representation hypothesis**: "king" minus "man"
    plus "woman" lands near "queen" because gender, royalty, and other
    concepts correspond to directions in the space. The high dimensionality
    that cursed one-hot codes is exactly what makes this possible — there's
    room for thousands of near-orthogonal meaningful directions.

    References: [Mikolov et al. 2013](https://arxiv.org/abs/1301.3781);
    Anthropic's
    [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html).
    """)
    return


@app.cell
def _(vectors, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt

    # Gender direction: average offset across multiple pairs
    _pairs = [("king", "queen"), ("man", "woman"), ("uncle", "aunt"),
              ("boy", "girl"), ("husband", "wife"), ("brother", "sister")]
    _pairs = [(a, b) for a, b in _pairs
              if a in vectors and b in vectors]

    _offsets = _np.array([vectors[b] - vectors[a] for a, b in _pairs],
                         dtype=float)
    _gender_dir = _np.mean(_offsets, axis=0)
    _gender_dir = _gender_dir / _np.linalg.norm(_gender_dir)

    # Project a broader set of words onto the gender direction
    _test_words = [a for a, _ in _pairs] + [b for _, b in _pairs]
    _extra = ["doctor", "nurse", "president", "secretary", "engineer",
              "teacher", "scientist", "dancer", "pilot", "nurse"]
    _test_words += [w for w in _extra if w in vectors]
    _test_words = list(dict.fromkeys(_test_words))

    _projs = []
    for _w in _test_words:
        _v = vectors[_w].astype(float)
        _projs.append((_w, float(_np.dot(_v, _gender_dir))))

    _projs.sort(key=lambda x: x[1])

    _fig, _ax = _plt.subplots(figsize=(10, 4))
    _colors = []
    for _w, _p in _projs:
        if _w in [a for a, _ in _pairs]:
            _colors.append("#1E88E5")
        elif _w in [b for _, b in _pairs]:
            _colors.append("#E53935")
        else:
            _colors.append("#666")

    _y_pos = range(len(_projs))
    _ax.barh(_y_pos, [p for _, p in _projs], color=_colors, height=0.7)
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels([w for w, _ in _projs], fontsize=9)
    _ax.set_xlabel("Projection onto gender direction", fontsize=10)
    _ax.set_title("One direction in 50-dimensional space", fontsize=11)
    _ax.axvline(0, color="#333", lw=0.5)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _plt.tight_layout()
    _plt.close(_fig)

    mo.vstack([
        _fig,
        mo.md("""
        The "gender direction" is the average of (queen − king),
        (woman − man), (aunt − uncle), etc. It's a single direction in
        50-dimensional space.

        Blue bars are the "male" words from each pair; red bars are
        "female." Grey words weren't used to compute the direction —
        they're a test. Notice that they still separate.

        This works because the network learned to *use* a direction for
        this concept. There's nothing special about this direction — it's
        one of many near-orthogonal concept directions the space can hold.
        In 50 dimensions, there's room for dozens. In 7,168 dimensions
        (DeepSeek-V3), there's room for thousands.
        """),
    ])


# ── §7  Superposition and sparsity ──────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Superposition and sparsity

    How many concepts can 50 dimensions hold? If each concept is an
    orthogonal direction, the answer is exactly 50. But there are far
    more than 50 concepts in the vocabulary.

    The trick is **superposition**: concepts don't need to be *exactly*
    orthogonal, just *nearly* orthogonal. In high dimensions, there's
    an enormous number of nearly-orthogonal directions available
    (you saw this in the random-cosine histogram above).

    The consequence is **sparsity**: if many concepts are packed into
    the same space via superposition, then for any given word only a
    few concepts are "active" — most coordinates are small.
    """)
    return


@app.cell
def _(mo):
    sparse_word_a = mo.ui.text(value="king", label="Word:", full_width=False)
    sparse_word_b = mo.ui.text(value="cat", label="Compare:",
                               full_width=False)
    mo.hstack([sparse_word_a, sparse_word_b], gap=1, justify="start")
    return (sparse_word_a, sparse_word_b)


@app.cell
def _(sparse_word_a, sparse_word_b, vectors, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt

    _wa = sparse_word_a.value.strip().lower()
    _wb = sparse_word_b.value.strip().lower()

    if _wa not in vectors:
        _out = mo.md(f"**`{_wa}`** not in vocabulary.")
    elif _wb not in vectors:
        _out = mo.md(f"**`{_wb}`** not in vocabulary.")
    else:
        _va = vectors[_wa].astype(float)
        _vb = vectors[_wb].astype(float)

        _fig, _ax = _plt.subplots(figsize=(10, 3.5))
        _x = _np.arange(50)
        _width = 0.35
        _ax.bar(_x - _width/2, _va, _width, color="#1E88E5", alpha=0.7,
                label=_wa)
        _ax.bar(_x + _width/2, _vb, _width, color="#E53935", alpha=0.7,
                label=_wb)
        _ax.axhline(0, color="#333", lw=0.5)
        _ax.set_xlabel("Dimension", fontsize=10)
        _ax.set_ylabel("Value", fontsize=10)
        _ax.set_title(f"Coordinates of '{_wa}' and '{_wb}' (50 dims)",
                      fontsize=11)
        _ax.legend(fontsize=9)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _plt.tight_layout()
        _plt.close(_fig)

        # Sparsity stats
        _thresh = 0.5
        _sparse_a = float(_np.mean(_np.abs(_va) < _thresh))
        _sparse_b = float(_np.mean(_np.abs(_vb) < _thresh))

        # Which dims are "active" (large magnitude) in each
        _active_a = set(_np.where(_np.abs(_va) > 1.0)[0])
        _active_b = set(_np.where(_np.abs(_vb) > 1.0)[0])
        _overlap = len(_active_a & _active_b)
        _union = len(_active_a | _active_b)

        _out = mo.vstack([
            _fig,
            mo.md(f"""
            Most coordinates are small. Fraction with |value| < {_thresh}:
            **{_wa}**: {_sparse_a:.0%},  **{_wb}**: {_sparse_b:.0%}.

            Dimensions with |value| > 1.0:
            **{_wa}** uses {len(_active_a)}, **{_wb}** uses {len(_active_b)},
            overlap: {_overlap} of {_union} total.

            Different words use different subsets of dimensions — that's
            superposition in action. The space isn't wasted; it's *shared*.
            Each word activates the directions that encode its concepts
            and leaves the rest near zero.
            """),
        ])
    _out


@app.cell
def _(vectors, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt

    # Distribution of coordinate magnitudes across many words
    _words = list(vectors.key_to_index)[:5000]
    _all_coords = _np.array([vectors[w] for w in _words],
                            dtype=float).flatten()
    _abs_coords = _np.abs(_all_coords)

    _fig, _ax = _plt.subplots(figsize=(10, 3.5))
    _ax.hist(_abs_coords, bins=100, color="#43A047", alpha=0.7,
             density=True, edgecolor="none")
    _ax.axvline(0.5, color="#E53935", lw=1.5, ls="--",
                label="|x| = 0.5")
    _ax.axvline(1.0, color="#1E88E5", lw=1.5, ls="--",
                label="|x| = 1.0")

    _frac_small = float(_np.mean(_abs_coords < 0.5))
    _frac_large = float(_np.mean(_abs_coords > 1.0))

    _ax.set_xlabel("|coordinate value|", fontsize=10)
    _ax.set_ylabel("Density", fontsize=10)
    _ax.set_title("Distribution of coordinate magnitudes (5K GloVe words)",
                  fontsize=11)
    _ax.legend(fontsize=9)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _plt.tight_layout()
    _plt.close(_fig)

    mo.vstack([
        _fig,
        mo.md(f"""
        Across 5,000 words × 50 dimensions = 250,000 coordinates:
        **{_frac_small:.0%}** have |value| < 0.5, and only
        **{_frac_large:.0%}** have |value| > 1.0.

        The distribution is heavily concentrated near zero. This is
        sparsity — and it's a direct consequence of superposition: each
        word only "uses" a subset of the available directions.

        This is a fundamental property of high-dimensional learned
        representations, not a compression trick.
        """),
    ])


# ── §8  Nearest-neighbor collapse ───────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Nearest-neighbor collapse

    Here's the curse of dimensionality at its most practical: for random
    data, the ratio of farthest to nearest distance converges to 1 as
    dimensions grow. All points become equidistant — search is meaningless.

    But trained embeddings defeat this.
    """)
    return


@app.cell
def _(mo):
    dim_slider_nn = mo.ui.slider(
        start=2, stop=500, step=1, value=50,
        label="Dimensions (D):",
        full_width=True,
    )
    dim_slider_nn
    return (dim_slider_nn,)


@app.cell
def _(dim_slider_nn, vectors, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt

    _n_points = 200
    _rng = _np.random.default_rng(77)

    # Compute farthest/nearest ratio across a range of dimensions
    _test_dims = [2, 5, 10, 20, 50, 100, 200, 500]
    _ratios_rand = []
    for _d in _test_dims:
        _pts = _rng.standard_normal((_n_points, _d))
        _pts = _pts / _np.linalg.norm(_pts, axis=1, keepdims=True)
        # Pairwise distances from first point
        _dists = _np.linalg.norm(_pts[1:] - _pts[0], axis=1)
        _ratios_rand.append(float(_np.max(_dists) / _np.min(_dists)))

    # GloVe: fixed at 50 dims, use first 200 words
    _glove_words = list(vectors.key_to_index)[:_n_points]
    _glove_vecs = _np.array([vectors[w] for w in _glove_words], dtype=float)
    _glove_norms = _np.linalg.norm(_glove_vecs, axis=1, keepdims=True)
    _glove_unit = _glove_vecs / _glove_norms
    _glove_dists = _np.linalg.norm(_glove_unit[1:] - _glove_unit[0], axis=1)
    _glove_ratio = float(_np.max(_glove_dists) / _np.min(_glove_dists))

    # Also compute for current slider value
    _D = dim_slider_nn.value
    _pts_cur = _rng.standard_normal((_n_points, _D))
    _pts_cur = _pts_cur / _np.linalg.norm(_pts_cur, axis=1, keepdims=True)
    _dists_cur = _np.linalg.norm(_pts_cur[1:] - _pts_cur[0], axis=1)
    _ratio_cur = float(_np.max(_dists_cur) / _np.min(_dists_cur))

    _fig, _ax = _plt.subplots(figsize=(10, 4))
    _ax.plot(_test_dims, _ratios_rand, "o-", color="#1E88E5", lw=2,
             label="Random unit vectors", ms=6)
    _ax.plot(_D, _ratio_cur, "s", color="#1E88E5", ms=10, zorder=5)
    _ax.axhline(_glove_ratio, color="#E53935", lw=2, ls="--",
                label=f"GloVe 50-dim (ratio = {_glove_ratio:.2f})")
    _ax.axhline(1.0, color="#999", lw=1, ls=":", alpha=0.5)

    _ax.set_xlabel("Dimensions (D)", fontsize=10)
    _ax.set_ylabel("Farthest / nearest distance", fontsize=10)
    _ax.set_title("Distance ratio collapse", fontsize=11)
    _ax.legend(fontsize=9)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.set_ylim(bottom=0.8)
    _plt.tight_layout()
    _plt.close(_fig)

    mo.vstack([
        _fig,
        mo.md(f"""
        At D={_D}, the farthest/nearest ratio for random data is
        **{_ratio_cur:.2f}** — heading toward 1.0 (everything equidistant).

        GloVe at 50 dimensions: **{_glove_ratio:.2f}**. Nearest neighbors
        are genuinely closer than far points. The learned structure defeats
        the curse.

        This is why embeddings are *useful* despite high dimensionality.
        And it's why [vector databases](https://en.wikipedia.org/wiki/Vector_database)
        work: approximate nearest-neighbor search makes sense because
        trained embeddings have real neighborhoods, not the uniform
        porridge that random data converges to.
        """),
    ])


# ── Closing ──────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## What to take away

    1. **One-hot codes** are maximally wasteful — no structure, the curse
       at its worst.
    2. **High dimensions are weird**: everything lives on the shell, random
       vectors are orthogonal, distances concentrate. Your 3D intuitions
       fail.
    3. **Trained embeddings exploit the weirdness**: the enormous number of
       near-orthogonal directions becomes a *resource* for encoding
       concepts via superposition.
    4. **Sparsity is fundamental**: superposition implies sparse
       activations. Most coordinates are near zero for any given input.
    5. **Heavy tails are real**: GloVe coordinates are not Gaussian. The
       gap between the Gaussian model and reality is exactly where
       quantization gets interesting — which is where we go
       [next](/dot-product).

    For depth: Hamming ch. 9 covers the geometry;
    [Anthropic's "Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html)
    covers why neural networks learn superposed representations;
    [3Blue1Brown on higher dimensions](https://www.youtube.com/watch?v=zwAD6dRSVyI)
    is the best visual introduction.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
