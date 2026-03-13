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

    _n = len(_words)

    # One-hot cosine similarities (0 for all off-diagonal, 1 on diagonal)
    _onehot = _np.eye(_n)
    _oh_cos = _onehot @ _onehot.T  # identity matrix

    # GloVe cosine similarities
    _glove_vecs = _np.array([vectors[w] for w in _words], dtype=float)
    _glove_norms = _np.linalg.norm(_glove_vecs, axis=1, keepdims=True)
    _glove_unit = _glove_vecs / _glove_norms
    _glove_cos = _glove_unit @ _glove_unit.T

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(10, 4))

    _im1 = _ax1.imshow(_oh_cos, cmap="RdYlGn", vmin=-1, vmax=1)
    _ax1.set_xticks(range(_n))
    _ax1.set_xticklabels(_words, rotation=45, ha="right", fontsize=9)
    _ax1.set_yticks(range(_n))
    _ax1.set_yticklabels(_words, fontsize=9)
    _ax1.set_title("One-hot cosine similarity", fontsize=11)
    for _i in range(_n):
        for _j in range(_n):
            _ax1.text(_j, _i, f"{_oh_cos[_i,_j]:.1f}", ha="center",
                      va="center", fontsize=8,
                      color="white" if abs(_oh_cos[_i,_j]) < 0.3 else "black")

    _im2 = _ax2.imshow(_glove_cos, cmap="RdYlGn", vmin=-1, vmax=1)
    _ax2.set_xticks(range(_n))
    _ax2.set_xticklabels(_words, rotation=45, ha="right", fontsize=9)
    _ax2.set_yticks(range(_n))
    _ax2.set_yticklabels(_words, fontsize=9)
    _ax2.set_title("GloVe cosine similarity (50-dim)", fontsize=11)
    for _i in range(_n):
        for _j in range(_n):
            _ax2.text(_j, _i, f"{_glove_cos[_i,_j]:.2f}", ha="center",
                      va="center", fontsize=8,
                      color="white" if abs(_glove_cos[_i,_j]) < 0.3 else "black")

    _plt.tight_layout()
    _plt.close(_fig)

    mo.vstack([
        _fig,
        mo.md("""
        **Left**: one-hot cosine similarities. Every pair is exactly 0
        — "cat" is no more similar to "dog" than to "france." No structure
        at all.

        **Right**: GloVe cosine similarities. Now cat–dog, king–queen,
        and paris–france show real similarity. The geometry encodes meaning.

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

    # GloVe pairwise cosines — sample from rank 1K-50K to avoid the
    # most common words (which cluster in one hemisphere due to training;
    # see note below)
    _all_words = list(vectors.key_to_index)[1000:50000]
    _glove_words = list(_rng.choice(_all_words, _n, replace=False))
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

        **Right**: pairwise cosines of 500 GloVe words (always 50-dim).
        The distribution is broader — real embeddings have genuine clusters
        of similar words. The structure is *not* random.

        (We sample from rank 1K–50K in the vocabulary. The top ~500 words
        — "the," "of," "and" — are almost all positively correlated with
        each other because they co-occur with everything. That's a real
        property of how GloVe was trained, and the
        [training notebook](https://jalammar.github.io/illustrated-word2vec/)
        is the place to understand it.)

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

        **Why?** Language itself is skewed: a few words ("the", "of",
        "and") appear vastly more often than others — that's
        [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law). GloVe
        learns from word co-occurrence counts, so this skew flows straight
        into the embedding geometry. The heavy tails aren't noise; they're
        the structure of language showing through. (How exactly training
        produces this is a story for the
        [training notebook](https://jalammar.github.io/illustrated-word2vec/);
        for the original paper, see
        [Pennington et al. 2014](https://nlp.stanford.edu/pubs/glove.pdf).)

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

    How many directions can a space hold? Let's start with cases you can
    draw. If you want N+1 vectors to be **maximally spread** in N
    dimensions (every pair at the same angle), there's exactly one answer:
    """)
    return


@app.cell
def _(mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D

    _fig = _plt.figure(figsize=(10, 4.5))

    # ── Left: 2D — 3 vectors at 120° ──
    _ax1 = _fig.add_subplot(1, 2, 1)
    _angles_deg = [0, 120, 240]
    _angles_rad = [a * _np.pi / 180 for a in _angles_deg]
    _colors = ["#1E88E5", "#E53935", "#43A047"]

    _ax1.set_xlim(-1.4, 1.4)
    _ax1.set_ylim(-1.4, 1.4)
    _ax1.set_aspect("equal")

    # Unit circle (faint)
    _theta = _np.linspace(0, 2 * _np.pi, 100)
    _ax1.plot(_np.cos(_theta), _np.sin(_theta), color="#ddd", lw=1)
    _ax1.axhline(0, color="#eee", lw=0.5)
    _ax1.axvline(0, color="#eee", lw=0.5)

    for _i, (_a, _c) in enumerate(zip(_angles_rad, _colors)):
        _x, _y = _np.cos(_a), _np.sin(_a)
        _ax1.annotate("", xy=(_x, _y), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->,head_width=0.15",
                                       color=_c, lw=2.5))
        _ax1.text(_x * 1.15, _y * 1.15, f"v{_i+1}",
                  ha="center", va="center", fontsize=10, fontweight="bold",
                  color=_c)

    _ax1.set_title("2D: 3 vectors at 120°", fontsize=11)
    _ax1.text(0, -1.35, "cos 120° = −0.50", ha="center", fontsize=9,
              color="#666")
    _ax1.spines["top"].set_visible(False)
    _ax1.spines["right"].set_visible(False)
    _ax1.spines["bottom"].set_visible(False)
    _ax1.spines["left"].set_visible(False)
    _ax1.set_xticks([])
    _ax1.set_yticks([])

    # ── Right: 3D — tetrahedron (4 vectors at 109.47°) ──
    _ax2 = _fig.add_subplot(1, 2, 2, projection="3d")
    _tet = _np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float) / _np.sqrt(3)

    _colors3 = ["#1E88E5", "#E53935", "#43A047", "#FF9800"]
    for _i in range(4):
        _ax2.plot([0, _tet[_i, 0]], [0, _tet[_i, 1]], [0, _tet[_i, 2]],
                  color=_colors3[_i], lw=2.5)
        _ax2.scatter(*_tet[_i], color=_colors3[_i], s=60, zorder=5)
        _ax2.text(_tet[_i, 0] * 1.2, _tet[_i, 1] * 1.2,
                  _tet[_i, 2] * 1.2, f"v{_i+1}", fontsize=9,
                  fontweight="bold", color=_colors3[_i])

    # Draw edges (faint)
    for _i in range(4):
        for _j in range(_i + 1, 4):
            _ax2.plot([_tet[_i, 0], _tet[_j, 0]],
                      [_tet[_i, 1], _tet[_j, 1]],
                      [_tet[_i, 2], _tet[_j, 2]],
                      color="#ccc", lw=0.8, ls="--")

    _ax2.set_title("3D: 4 vectors at 109.5°", fontsize=11)
    _ax2.text2D(0.5, 0.02, "cos 109.5° = −0.33", ha="center", fontsize=9,
                color="#666", transform=_ax2.transAxes)
    _ax2.set_xlim(-1, 1)
    _ax2.set_ylim(-1, 1)
    _ax2.set_zlim(-1, 1)
    _ax2.set_xticks([])
    _ax2.set_yticks([])
    _ax2.set_zticks([])
    _ax2.xaxis.pane.fill = False
    _ax2.yaxis.pane.fill = False
    _ax2.zaxis.pane.fill = False
    _ax2.xaxis.pane.set_edgecolor("w")
    _ax2.yaxis.pane.set_edgecolor("w")
    _ax2.zaxis.pane.set_edgecolor("w")

    _plt.tight_layout()
    _plt.close(_fig)

    mo.vstack([
        _fig,
        mo.md("""
        In **2D** you can fit 3 equi-angled vectors, each 120° apart
        (cos = −0.50). In **3D** you can fit 4 — the vertices of a
        tetrahedron, at 109.5° (cos = −0.33). In general, N dimensions
        fit N+1 perfectly equi-angled vectors at cos θ = −1/N.

        But "exactly equi-angled" is a severe constraint. What if we
        relax it a little — allowing vectors to be *nearly* orthogonal
        instead of exactly equi-angled? Then we can pack **far more**
        directions, and high dimensionality is the reason.
        """),
    ])


@app.cell
def _(mo):
    packing_dim_slider = mo.ui.slider(
        start=2, stop=10000, step=1, value=7168,
        label="Dimensions (N):",
        full_width=True,
    )
    packing_zoom_slider = mo.ui.slider(
        start=-2, stop=2, step=0.1, value=0.0,
        label="Zoom (log₁₀ degrees):",
        full_width=True,
    )
    mo.hstack([packing_dim_slider, packing_zoom_slider], widths=[0.5, 0.5])
    return (packing_dim_slider, packing_zoom_slider,)


@app.cell
def _(packing_dim_slider, packing_zoom_slider, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import math as _math

    _N = packing_dim_slider.value

    _welch_limit = 1.0 / _math.sqrt(_N) if _N > 1 else 1.0
    _welch_limit_deg = _math.degrees(_math.asin(min(_welch_limit, 1.0)))
    # Zoom slider scales the x-axis range; plot size stays fixed
    _x_view = max(_welch_limit_deg * 8 * 10**packing_zoom_slider.value, 0.001)
    _deltas = _np.linspace(0.0001, _x_view, 800)
    _epsilons = _np.sin(_np.radians(_deltas))

    # ── Welch bound (tight upper bound on M for eps < 1/sqrt(N)) ──
    # M <= N(1-eps^2) / (1 - N*eps^2)
    _log_welch = _np.full_like(_epsilons, _np.nan)
    for _i, _e in enumerate(_epsilons):
        if _e < _welch_limit * 0.999:
            _denom = 1.0 - _N * _e**2
            if _denom > 0:
                _m = _N * (1.0 - _e**2) / _denom
                if _m > 1:
                    _log_welch[_i] = _math.log10(_m)

    # ── Random coding lower bound (the exponential curve) ──
    # M >= exp(N*eps^2/4)  — from union bound on random unit vectors
    # This is trivial near eps=0 but grows exponentially past 1/sqrt(N).
    _log_rc = _np.array([
        _N * e**2 / (4 * _math.log(10)) for e in _epsilons
    ])
    # Floor at N (orthogonal vectors always exist)
    _log_rc = _np.maximum(_log_rc, _math.log10(_N))

    # ── Quantization noise as angle (numerical, per-vector scaling) ──
    # All formats: subnormals present, no NaN, no infinities.

    def _build_format_values(_ebits, _mbits):
        """Non-negative representable values for ExMy (no NaN/Inf)."""
        _bias = 2**(_ebits - 1) - 1
        _max_e = (1 << _ebits) - 1
        _vals = set()
        for _m in range(1 << _mbits):  # subnormals (e=0)
            _vals.add(2.0**(1 - _bias) * (_m / (1 << _mbits)))
        for _e in range(1, _max_e + 1):  # normals
            for _m in range(1 << _mbits):
                _vals.add(2.0**(_e - _bias) * (1.0 + _m / (1 << _mbits)))
        return _np.array(sorted(_vals))

    def _quantize(_x, _pos_vals):
        """Round-to-nearest, clip to ±max."""
        _s = _np.sign(_x)
        _a = _np.clip(_np.abs(_x), 0, _pos_vals[-1])
        _idx = _np.searchsorted(_pos_vals, _a)
        _idx = _np.clip(_idx, 0, len(_pos_vals) - 1)
        _lo = _np.clip(_idx - 1, 0, len(_pos_vals) - 1)
        _d_lo = _np.abs(_a - _pos_vals[_lo])
        _d_hi = _np.abs(_a - _pos_vals[_idx])
        _best = _np.where(_d_lo <= _d_hi, _lo, _idx)
        return _pos_vals[_best] * _s

    def _dot_noise_deg(_ebits, _mbits, _dim, _n_trials=500):
        """Monte Carlo: RMS cosine error from quantizing both operands."""
        _rng = _np.random.default_rng(42)
        _pv = _build_format_values(_ebits, _mbits)
        _errs = []
        for _ in range(_n_trials):
            _a = _rng.standard_normal(_dim)
            _b = _rng.standard_normal(_dim)
            _a = _a / _np.linalg.norm(_a)
            _b = _b / _np.linalg.norm(_b)
            _exact = float(_np.dot(_a, _b))
            # Per-vector absmax scaling: map peak to max representable
            _sa = _pv[-1] / _np.abs(_a).max()
            _sb = _pv[-1] / _np.abs(_b).max()
            _aq = _quantize(_a * _sa, _pv) / _sa
            _bq = _quantize(_b * _sb, _pv) / _sb
            _approx = float(_np.dot(_aq, _bq))
            _errs.append(_approx - _exact)
        _sigma_cos = float(_np.std(_errs))
        return _math.degrees(_math.asin(min(_sigma_cos, 1.0)))

    # (name, ebits, mbits, color)
    _quant_formats = [
        ("E1M6 (8b)", 1, 6, "#90A4AE"),
        ("E2M5 (8b)", 2, 5, "#78909C"),
        ("E4M3 (8b)", 4, 3, "#546E7A"),
        ("E5M2 (8b)", 5, 2, "#455A64"),
        ("E2M3 (6b)", 2, 3, "#37474F"),
        ("E2M1 (4b)", 2, 1, "#263238"),
    ]
    _noise_lines = []
    for _fname, _eb, _mb, _color in _quant_formats:
        _deg = _dot_noise_deg(_eb, _mb, _N) if _N >= 4 else 90.0
        _noise_lines.append((_fname, _deg, _color))

    # ── Where does DeepSeek vocab cross each curve? ──
    _vocab_log = _math.log10(129280)
    _ds_cross_welch = None
    _ds_cross_rc = None
    for _i in range(len(_deltas)):
        if _ds_cross_welch is None and not _np.isnan(_log_welch[_i]) \
                and _log_welch[_i] >= _vocab_log:
            _ds_cross_welch = float(_deltas[_i])
        if _ds_cross_rc is None and _log_rc[_i] >= _vocab_log:
            _ds_cross_rc = float(_deltas[_i])

    # ── Plot ──
    _fig, _ax = _plt.subplots(figsize=(10, 5.5))

    # Random coding lower bound — the exponential curve
    _ax.plot(_deltas, _log_rc, color="#43A047", lw=2,
             label="Lower bound: M ≥ exp(Nε²/4)")
    # Shade above lower bound
    _ax.fill_between(_deltas, _log_rc, alpha=0.04, color="#43A047")

    # Welch bound
    _valid = ~_np.isnan(_log_welch)
    if _valid.any():
        _ax.plot(_deltas[_valid], _log_welch[_valid], color="#FF9800",
                 lw=2.5, label="Welch upper bound")
        _ax.fill_between(_deltas[_valid], _math.log10(_N),
                         _log_welch[_valid], alpha=0.06, color="#FF9800")

    # Vertical line at 1/sqrt(N)
    _ax.axvline(_welch_limit_deg, color="#FF9800", lw=1.5, ls=":",
                alpha=0.5)
    _ax.text(_welch_limit_deg, _ax.get_ylim()[0] if _ax.get_ylim()[0] > 0
             else 0.5, f" 1/√N", fontsize=8, color="#FF9800",
             va="bottom", ha="left")

    # M = N reference
    _ax.axhline(_math.log10(_N), color="#999", lw=1, ls="--", alpha=0.5,
                label=f"M = N = {_N:,}")

    # DeepSeek-V3 vocab
    _ax.axhline(_vocab_log, color="#1E88E5", lw=1.5, ls="--",
                alpha=0.7, label="DeepSeek-V3 vocab (129K)")
    if _ds_cross_welch:
        _ax.plot(_ds_cross_welch, _vocab_log, "o", color="#FF9800",
                 ms=7, zorder=5)
    if _ds_cross_rc:
        _ax.plot(_ds_cross_rc, _vocab_log, "o", color="#43A047",
                 ms=7, zorder=5)
        _ax.annotate(f"{_ds_cross_rc:.1f}°",
                     (_ds_cross_rc, _vocab_log),
                     textcoords="offset points", xytext=(8, -12),
                     fontsize=9, color="#43A047")

    # Quantization noise floors
    _noise_y_top = _log_rc.max() * 0.97
    for _qi, (_fname, _ndeg, _ncol) in enumerate(_noise_lines):
        if _ndeg < _x_view:
            _ax.axvline(_ndeg, color=_ncol, lw=1.5, ls=":", alpha=0.6)
            _y_pos = _noise_y_top - _qi * (_log_rc.max() * 0.055)
            _ax.text(_ndeg + _x_view * 0.008, _y_pos,
                     f"{_fname}\n({_ndeg:.3f}°)", fontsize=7,
                     color=_ncol, va="top", fontweight="bold")

    _ax.set_xlabel("Angle deviation from 90° (degrees)", fontsize=10)
    _ax.set_ylabel("Number of vectors M  (log₁₀)", fontsize=10)
    _ax.set_title(
        f"How many nearly-orthogonal directions in {_N:,} dimensions?",
        fontsize=11)
    _ax.legend(fontsize=9, loc="upper left")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.set_ylim(bottom=max(_math.log10(_N) - 0.5, 0))
    _ax.set_xlim(0, _x_view)
    _plt.tight_layout()
    _plt.close(_fig)

    # ── Interpretive text ──
    _ds_text = ""
    if _ds_cross_welch:
        _ds_eps = _math.sin(_math.radians(_ds_cross_welch))
        _ds_frac = _ds_eps / _welch_limit
        _ds_text = f"""
        DeepSeek-V3's vocabulary (129K tokens) crosses the Welch bound
        at **{_ds_cross_welch:.2f}°** — that's **{_ds_frac:.0%}** of the
        way to the 1/√N limit."""
    if _ds_cross_rc:
        _ds_text += f""" The random coding bound guarantees the same
        count is achievable by **{_ds_cross_rc:.1f}°**."""

    mo.vstack([
        _fig,
        mo.md(f"""
        Two regimes, separated by the vertical line at 1/√N =
        {_welch_limit:.4f} ({_welch_limit_deg:.2f}°):

        **Left of 1/√N** — the [Welch bound](https://en.wikipedia.org/wiki/Welch_bounds)
        gives a tight algebraic ceiling:
        M ≤ N(1−ε²)/(1−Nε²). It starts at M = N and diverges at the
        boundary.

        **Right of 1/√N** — the number of packable directions grows
        **exponentially** with N. The green curve is a lower bound from
        the probabilistic method: **M ≥ exp(Nε²/4)**. The N is in the
        exponent — that's where the room comes from. For N = {_N:,} at
        just {_welch_limit_deg * 7:.1f}° from orthogonal, you can
        already pack 10^{int(_N * (_welch_limit * 7)**2 / (4 * _math.log(10)))}
        directions. The best upper bound in this regime is
        [Kabatiansky–Levenshtein (1978)](https://en.wikipedia.org/wiki/Kabatiansky%E2%80%93Levenshtein_bound).
        {_ds_text}

        The vertical dotted lines show dot-product noise for different
        number formats (per-vector absmax scaling, both operands quantized).
        E1M6 ≈ symmetric INT8; E2M5 is popular for inference; E5M2 for
        training (gradient dynamic range). All 8-bit and 6-bit formats
        land far left of the bounds. Even E2M1 (4-bit!) barely dents
        the geometry — **low-precision arithmetic preserves the structure**.

        Try N = 50 to see the bounds at GloVe scale.
        """),
    ])


@app.cell
def _(mo):
    mo.md("""
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


# ── §9  Scaling up: what does 7168 dimensions look like? ─────────────


@app.cell
def _(mo):
    mo.md("""
    ## Scaling up: what does 7,168 dimensions look like?

    Everything above used GloVe at 50 dimensions. Real LLMs are much
    bigger — DeepSeek-V3 embeds tokens in **7,168 dimensions**, split
    across 128 attention heads of 56 dimensions each.

    A natural question: does the geometry behave like a 7,168-dimensional
    sphere, or like 128 independent 56-dimensional spheres? The answer
    matters — it tells you what "dimension" means for concentration of
    measure, distance collapse, and ultimately for how the hardware
    processes these vectors.

    We sampled 5,000 token embeddings from DeepSeek-V3's actual embedding
    table and compared the pairwise cosine distribution against two random
    models:
    - **Random S^7167**: unit vectors on the full 7,168-dim sphere
    - **128 × S^55**: 128 independent 56-dim unit vectors concatenated
      and normalized (the "independent heads" model)
    """)
    return


@app.cell
def _(mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import os as _os

    # Load precomputed histograms (4 KB file)
    _data_path = _os.path.join(_os.path.dirname(__file__) or ".",
                               "data", "deepseek_v3_cosine_stats.npz")
    _data = _np.load(_data_path)
    _bins = _data['bins']
    _bin_centers = (_bins[:-1] + _bins[1:]) / 2
    _hist_real = _data['hist_real']
    _hist_A = _data['hist_A']
    _hist_B = _data['hist_B']

    _real_std = float(_data['real_std'])
    _real_max = float(_data['real_max'])
    _A_std = float(_data['A_std'])
    _B_std = float(_data['B_std'])

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(12, 4))

    # Left: the two random models overlaid
    _width = _bins[1] - _bins[0]
    _ax1.bar(_bin_centers, _hist_A, width=_width, color="#1E88E5", alpha=0.6,
             label=f"Random S$^{{7167}}$ (σ={_A_std:.4f})")
    _ax1.bar(_bin_centers, _hist_B, width=_width, color="#FF9800", alpha=0.4,
             label=f"128 × S$^{{55}}$ concat (σ={_B_std:.4f})")
    _ax1.set_xlabel("Cosine similarity", fontsize=10)
    _ax1.set_ylabel("Density", fontsize=10)
    _ax1.set_title("Two random models — indistinguishable", fontsize=11)
    _ax1.set_xlim(-0.15, 0.15)
    _ax1.legend(fontsize=8)
    _ax1.spines["top"].set_visible(False)
    _ax1.spines["right"].set_visible(False)

    # Right: real vs random
    _ax2.bar(_bin_centers, _hist_A, width=_width, color="#1E88E5", alpha=0.5,
             label=f"Random S$^{{7167}}$ (σ={_A_std:.4f})")
    _ax2.bar(_bin_centers, _hist_real, width=_width, color="#E53935", alpha=0.5,
             label=f"DeepSeek-V3 (σ={_real_std:.4f})")
    _ax2.set_xlabel("Cosine similarity", fontsize=10)
    _ax2.set_title("Real embeddings vs random", fontsize=11)
    _ax2.set_xlim(-0.15, 0.6)
    _ax2.legend(fontsize=8)
    _ax2.spines["top"].set_visible(False)
    _ax2.spines["right"].set_visible(False)

    _plt.tight_layout()
    _plt.close(_fig)

    mo.vstack([
        _fig,
        mo.md(f"""
        **Left**: the two random models are indistinguishable. Concatenating
        128 independent 56-dim spheres and normalizing gives the same
        cosine distribution as a single 7,168-dim sphere (both σ={_A_std:.4f}).
        The multi-head structure doesn't exist in the input embeddings —
        the head decomposition only happens later, when attention computes
        separate Q, K, V projections.

        **Right**: DeepSeek-V3's real embeddings are **{_real_std/_A_std:.1f}×
        wider** (σ={_real_std:.4f}) with a long right tail reaching
        {_real_max:.2f}. That extra width is the learned structure — tokens
        with related meanings have genuinely higher cosine similarity.

        The takeaway: when someone says "7,168 dimensions," ask what they
        mean. The *ambient* dimension is 7,168. The concentration-of-measure
        predictions for that dimension are a useful baseline. But the real
        data has structure that no random model captures — and once attention
        slices it into 128 heads of 56 dimensions each, the effective
        geometry changes again.

        Simple models are a starting point, not the answer. But without
        them you can't see what the data is doing differently.
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
       quantization gets interesting.
    6. **"Dimension" needs thought**: 7,168 ambient dimensions, 128 heads
       of 56 dimensions, and the intrinsic dimensionality of the learned
       structure are three different numbers. Simple random models are
       useful baselines, not the full story.

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
