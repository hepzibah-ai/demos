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
    # The Dot Product

    Every similarity score, every projection, every attention weight in an LLM
    reduces to the same operation: **multiply element-wise, then sum**.
    That's the dot product.

    Cosine similarity *is* a dot product (on unit vectors). Gram–Schmidt
    *is* repeated dot products. Attention *is* a matrix of dot products.

    This notebook cracks open the operation so you can see what's actually
    happening inside.

    Why *this* operation? It comes from Pythagoras. The law of cosines
    generalizes $a^2 + b^2 = c^2$ to any triangle:

    $$c^2 = a^2 + b^2 - 2ab\cos\theta$$

    Rearrange for $\cos\theta$ and you get the dot product. That's why
    multiply-and-sum measures geometric similarity — it's not a convention,
    it's Pythagoras in disguise. For the full story, see 3Blue1Brown's
    [Dot products and duality](https://www.3blue1brown.com/lessons/dot-products).
    """)
    return


@app.cell
def _():
    import gensim.downloader as _api
    vectors = _api.load("glove-wiki-gigaword-50")
    return (vectors,)


@app.cell
def _(mo):
    word_a = mo.ui.text(value="king", label="Word A:", full_width=False)
    word_b = mo.ui.text(value="queen", label="Word B:", full_width=False)
    mo.vstack([
        mo.md("## Dot product exposed"),
        mo.md(
            """
            Pick two words. Below you'll see all 50 component-wise products
            — what cosine similarity computes and hides from you.

            Type words, then **click outside** to update.
            """
        ),
        mo.hstack([word_a, word_b], gap=1, justify="start"),
    ])
    return word_a, word_b


@app.cell
def _(mo, vectors, word_a, word_b):
    import numpy as _np
    import matplotlib.pyplot as _plt

    _wa = word_a.value.strip().lower()
    _wb = word_b.value.strip().lower()

    if _wa not in vectors:
        _out = mo.md(f"**`{_wa}`** is not in the vocabulary. Try another word.")
    elif _wb not in vectors:
        _out = mo.md(f"**`{_wb}`** is not in the vocabulary. Try another word.")
    else:
        _va = vectors[_wa].astype(float)
        _vb = vectors[_wb].astype(float)
        _products = _va * _vb
        _dot = float(_np.sum(_products))
        _cos = float(_np.dot(_va, _vb) / (
            _np.linalg.norm(_va) * _np.linalg.norm(_vb)))

        # Count positive and negative contributions
        _pos_sum = float(_np.sum(_products[_products > 0]))
        _neg_sum = float(_np.sum(_products[_products < 0]))

        # Bar chart of component-wise products
        _fig, _ax = _plt.subplots(figsize=(10, 3.5))
        _colors = ["#43A047" if p >= 0 else "#E53935" for p in _products]
        _ax.bar(range(50), _products, color=_colors, width=0.8)
        _ax.axhline(0, color="#333", lw=0.5)
        _ax.set_xlabel("Dimension", fontsize=10)
        _ax.set_ylabel("a[i] × b[i]", fontsize=10)
        _ax.set_title(
            f"'{_wa}' · '{_wb}' — 50 component-wise products",
            fontsize=11)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _plt.tight_layout()
        _plt.close(_fig)

        _text = mo.md(
            f"""
            ### Result: **{_dot:.2f}**

            | | Value |
            |---|---:|
            | Green bars (positive) | +{_pos_sum:.2f} |
            | Red bars (negative) | {_neg_sum:.2f} |
            | **Sum (dot product)** | **{_dot:.2f}** |
            | Cosine similarity | {_cos:.3f} |

            The dot product is the sum of those 50 bars. Positive bars
            = dimensions where both words agree in sign and magnitude.
            Negative bars = dimensions where they disagree.

            **Cosine similarity** divides by both norms, collapsing this
            to a single number between −1 and 1. Useful — but it hides
            the partial cancellation happening underneath.
            """
        )

        _out = mo.vstack([_fig, _text])
    _out
    return


@app.cell
def _(mo, vectors, word_a, word_b):
    import numpy as _np
    import matplotlib.pyplot as _plt

    _wa = word_a.value.strip().lower()
    _wb = word_b.value.strip().lower()

    if _wa not in vectors or _wb not in vectors:
        _out = mo.md("")
    else:
        _va = vectors[_wa].astype(float)
        _vb = vectors[_wb].astype(float)

        # Cumulative sum of component-wise products
        _products = _va * _vb
        _cumsum = _np.cumsum(_products)
        _dot = float(_cumsum[-1])

        _fig2, _ax2 = _plt.subplots(figsize=(10, 3))
        _ax2.fill_between(range(50), _cumsum,
                          where=_cumsum >= 0, color="#43A047", alpha=0.3)
        _ax2.fill_between(range(50), _cumsum,
                          where=_cumsum < 0, color="#E53935", alpha=0.3)
        _ax2.plot(range(50), _cumsum, color="#333", lw=1.5)
        _ax2.axhline(_dot, color="#1E88E5", lw=1, ls="--", alpha=0.7)
        _ax2.axhline(0, color="#999", lw=0.5)
        _ax2.set_xlabel("Dimension", fontsize=10)
        _ax2.set_ylabel("Running sum", fontsize=10)
        _ax2.set_title(
            f"Cumulative dot product — the path to {_dot:.2f}",
            fontsize=11)
        _ax2.spines["top"].set_visible(False)
        _ax2.spines["right"].set_visible(False)
        _plt.tight_layout()
        _plt.close(_fig2)

        _out = mo.vstack([
            _fig2,
            mo.md(
                f"""
                The running sum wanders up and down as each dimension either
                reinforces or cancels the total. The final value ({_dot:.2f})
                is the dashed blue line.

                Try "good" and "bad" — they're similar (cosine ≈ 0.8) but
                you'll see how many dimensions actively *disagree*. Or try
                "cat" and "democracy" for near-zero similarity with wild swings.
                """
            ),
        ])
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Gram–Schmidt revisited: it's all dot products

    In the [embedding notebook](/embedding), you saw Gram–Schmidt build
    an orthogonal coordinate system from two word vectors.
    Now that we've cracked open the dot product, let's name exactly
    what each step does.
    """)
    return


@app.cell
def _(mo, vectors, word_a, word_b):
    import numpy as _np

    _wa = word_a.value.strip().lower()
    _wb = word_b.value.strip().lower()

    if _wa not in vectors or _wb not in vectors:
        _out = mo.md("")
    else:
        _va = vectors[_wa].astype(float)
        _vb = vectors[_wb].astype(float)

        # Step 1: normalize a
        _norm_a = float(_np.linalg.norm(_va))
        _e1 = _va / _norm_a

        # Step 2: project b onto e1
        _proj_coeff = float(_np.dot(_vb, _e1))
        _proj = _proj_coeff * _e1
        _resid = _vb - _proj
        _norm_resid = float(_np.linalg.norm(_resid))
        _e2 = _resid / _norm_resid

        # Verify orthogonality
        _check = float(_np.dot(_e1, _e2))

        _out = mo.md(
            f"""
            ### The three operations

            Starting from **{_wa}** (a) and **{_wb}** (b):

            **1. Normalize** — make a unit vector:

            $\\quad e_1 = a \\;/\\; \\|a\\|$

            $\\quad\\|a\\| = \\sqrt{{a \\cdot a}} = \\sqrt{{{float(_np.dot(_va, _va)):.2f}}} = {_norm_a:.3f}$

            &nbsp;&nbsp;&nbsp;&nbsp;→ That square root requires **one dot product** (a with itself).

            **2. Project** — find how much b points along e₁:

            $\\quad \\text{{proj}} = (b \\cdot e_1) \\, e_1$

            $\\quad b \\cdot e_1 = {_proj_coeff:.3f}$

            &nbsp;&nbsp;&nbsp;&nbsp;→ **One dot product** to get the coefficient.

            **3. Subtract and normalize** — the leftover is the new direction:

            $\\quad r = b - \\text{{proj}}$, $\\quad e_2 = r \\;/\\; \\|r\\|$

            $\\quad \\|r\\| = {_norm_resid:.3f}$

            &nbsp;&nbsp;&nbsp;&nbsp;→ **One more dot product** (r with itself) for the norm.

            ---

            **Three dot products** to produce one new orthogonal basis vector.
            For a third word, it takes two projections + one norm = **five** dot products total.
            For D words, it's O(D²) dot products.

            Verification: $e_1 \\cdot e_2 = {_check:.1e}$ (≈ 0, as expected).
            """
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## RMSNorm: making unit vectors, efficiently

    DeepSeek (and most modern LLMs) applies **RMSNorm** before every
    attention and feed-forward layer. What is it?

    $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{D} \sum_i x_i^2}} \cdot \gamma$$

    Ignoring the learnable scale $\gamma$: this divides each vector by
    the root-mean-square of its components — which is just the norm
    divided by $\sqrt{D}$. The result is a vector with
    $\text{RMS} = 1$, or equivalently $\|x\| = \sqrt{D}$.

    Why does this matter? When all vectors have the same norm,
    **the dot product** ***is*** **the cosine similarity** (up to a constant).
    No extra division needed downstream.
    """)
    return


@app.cell
def _(mo, vectors, word_a):
    import numpy as _np

    _wa = word_a.value.strip().lower()

    if _wa not in vectors:
        _out = mo.md("")
    else:
        _va = vectors[_wa].astype(float)
        _D = len(_va)

        _norm = float(_np.linalg.norm(_va))
        _rms = float(_np.sqrt(_np.mean(_va ** 2)))

        # RMSNorm result
        _normed = _va / _rms
        _normed_norm = float(_np.linalg.norm(_normed))
        _normed_rms = float(_np.sqrt(_np.mean(_normed ** 2)))

        # Unit vector result
        _unit = _va / _norm
        _unit_norm = float(_np.linalg.norm(_unit))
        _unit_rms = float(_np.sqrt(_np.mean(_unit ** 2)))

        _out = mo.md(
            f"""
            ### Example: "{_wa}" ({_D} dimensions)

            | | Original | Unit vector (÷ ‖x‖) | RMSNorm (÷ RMS) |
            |---|---:|---:|---:|
            | ‖x‖ (L2 norm) | {_norm:.3f} | {_unit_norm:.3f} | {_normed_norm:.3f} |
            | RMS | {_rms:.3f} | {_unit_rms:.3f} | {_normed_rms:.3f} |

            The unit vector has ‖x‖ = 1. The RMSNorm vector has RMS = 1
            (so ‖x‖ = √{_D} = {_np.sqrt(_D):.3f}).

            Both preserve direction — they differ only in scale. The key
            insight: after RMSNorm, $a \\cdot b = D \\cdot \\cos(\\theta)$
            for any two vectors. The network never has to normalize again
            for comparisons.

            **Hardware payoff**: one division per vector, instead of one
            per *pair*. At DeepSeek scale (7168 dimensions, 128 heads,
            61 layers), that's enormous.
            """
        )
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Operation count: how the dot product scales

    One dot product on D-dimensional vectors costs:

    - **D multiplies** (one per component)
    - **D − 1 adds** (summing the products)

    Or roughly: **2D operations** per dot product, one **multiply-accumulate
    (MAC)** per component.
    """)
    return


@app.cell
def _(mo):
    dim_slider = mo.ui.slider(
        start=64, stop=16384, step=64, value=7168,
        label="Embedding dimension (D):",
        full_width=True,
    )
    ctx_slider = mo.ui.slider(
        start=128, stop=131072, step=128, value=4096,
        label="Context length (tokens):",
        full_width=True,
    )
    mo.vstack([
        mo.md("### How many operations in one forward pass?"),
        mo.md(
            """
            Adjust the sliders to see how multiply-accumulate operations
            scale. Default values are DeepSeek-V3 (7168-dim embeddings,
            128 attention heads, 61 layers).
            """
        ),
        dim_slider,
        ctx_slider,
    ])
    return ctx_slider, dim_slider


@app.cell
def _(ctx_slider, dim_slider, mo):
    _D = dim_slider.value
    _T = ctx_slider.value
    _heads = 128
    _layers = 61
    _d_head = _D // _heads if _heads > 0 else _D

    # Per head: Q·K^T is T×T dot products of d_head-dim vectors
    # then attn·V is T dot products of d_head-dim vectors per output position
    _qk_per_head = _T * _T * _d_head       # MACs for Q·K^T
    _av_per_head = _T * _T * _d_head       # MACs for attn·V
    _attn_per_layer = (_qk_per_head + _av_per_head) * _heads
    _attn_total = _attn_per_layer * _layers

    # Format with units
    def _fmt(n):
        if n >= 1e15:
            return f"{n/1e15:.1f} peta-MACs"
        if n >= 1e12:
            return f"{n/1e12:.1f} tera-MACs"
        if n >= 1e9:
            return f"{n/1e9:.1f} giga-MACs"
        if n >= 1e6:
            return f"{n/1e6:.1f} mega-MACs"
        return f"{n:,.0f} MACs"

    _single_dot = _D
    _one_head_qk = _T * _T * _d_head
    _one_layer = _attn_per_layer

    mo.md(
        f"""
        | What | MACs |
        |------|------|
        | One dot product ({_D}-dim) | **{_D:,}** |
        | One head, Q·K^T ({_T}×{_T} @ {_d_head}-dim) | **{_fmt(_one_head_qk)}** |
        | One layer ({_heads} heads, Q·K^T + attn·V) | **{_fmt(_one_layer)}** |
        | Full model ({_layers} layers, attention only) | **{_fmt(_attn_total)}** |

        That's just the attention dot products — feed-forward layers, which
        are also matrix multiplies (= batches of dot products), roughly
        double the total.

        Context length enters as **T²** — double the context, quadruple
        the work. This is why long-context inference is expensive, and why
        every multiply-accumulate must be as cheap as possible.

        **Next notebook**: what happens when those dimensions go from 50 to
        7,168 — the geometry of high-dimensional space.
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
