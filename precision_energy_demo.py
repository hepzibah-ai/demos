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


# ── §0  Title ──


@app.cell
def _(mo):
    mo.md(
        """
        # Precision and Energy

        You've seen that [high-dimensional geometry](../high-dimensions)
        preserves structure even under quantization — the noise from fp8 or
        4-bit arithmetic is a tiny fraction of a degree. Now the question
        is: **why go low-precision at all?**

        The answer is energy. Every multiply-accumulate (MAC) burns
        femtojoules. Halving the bit-width doesn't just halve the silicon
        — it can quarter the energy per operation. This notebook takes you
        from number formats through gate-level costs to system-level
        efficiency, building the case for purpose-built inference hardware.

        ---
        """
    )


# ── §1  Quantize the dot product ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §1 — Quantize the dot product

        ### Anatomy of a low-precision float

        Every format in the **ExMy** family packs a sign bit, E exponent
        bits, and M mantissa bits into a small word:

        ```
        ┌───┬───────────┬─────────────┐
        │ s │ e₃e₂e₁e₀  │ m₂ m₁ m₀    │   ← E4M3: 1+4+3 = 8 bits
        └───┴───────────┴─────────────┘
        value = (-1)^s  ×  1.m₂m₁m₀  ×  2^(e - bias)
        ```

        The leading **1.** before the mantissa is *implied* — you get an
        extra bit of precision for free. **Subnormals** are the exception:
        when the exponent field is all zeros, the implied bit becomes **0.**
        instead of 1., which is the only way to represent **zero** (and
        values near it). This creates an awkward side effect: both
        `00000000` and `10000000` are valid zeros (+0 and −0) — two
        bit patterns for the same value.

        The tradeoff: more exponent bits → wider dynamic range (useful
        when you don't know the data scale, e.g. training gradients).
        More mantissa bits → finer resolution (useful when you *do*
        know the scale, e.g. inference with calibrated weights).

        Pick two words below. We'll compute their cosine similarity at
        full precision, then quantize **both vectors** to each format
        before recomputing. The result barely moves — until 4-bit.
        """
    )


@app.cell
def _(mo):
    word1_input = mo.ui.text(value="king", label="Word 1:", full_width=False)
    word2_input = mo.ui.text(value="queen", label="Word 2:", full_width=False)
    mo.hstack([word1_input, word2_input], gap=1)
    return (word1_input, word2_input)


@app.cell
def _(mo):
    """Load GloVe model and define quantizer helpers (shared across notebook)."""
    import numpy as np
    import gensim.downloader as api

    glove_model = api.load("glove-wiki-gigaword-50")

    def build_format_values(ebits, mbits):
        """Non-negative representable values for ExMy (no NaN/Inf, with subnormals)."""
        bias = 2**(ebits - 1) - 1
        max_e = (1 << ebits) - 1
        vals = set()
        for m in range(1 << mbits):
            vals.add(2.0**(1 - bias) * (m / (1 << mbits)))
        for e in range(1, max_e + 1):
            for m in range(1 << mbits):
                vals.add(2.0**(e - bias) * (1.0 + m / (1 << mbits)))
        return np.array(sorted(vals))

    def quantize(x, pv):
        """Round-to-nearest, clip to ±max."""
        s = np.sign(x)
        a = np.clip(np.abs(x), 0, pv[-1])
        idx = np.searchsorted(pv, a)
        idx = np.clip(idx, 0, len(pv) - 1)
        lo = np.clip(idx - 1, 0, len(pv) - 1)
        d_lo = np.abs(a - pv[lo])
        d_hi = np.abs(a - pv[idx])
        best = np.where(d_lo <= d_hi, lo, idx)
        return pv[best] * s

    def quantize_vec(v, pv):
        """Per-vector absmax scaling then quantize."""
        scale = pv[-1] / np.abs(v).max()
        return quantize(v * scale, pv) / scale

    return (build_format_values, glove_model, quantize, quantize_vec)


@app.cell
def _(word1_input, word2_input, build_format_values, quantize_vec,
      glove_model, mo):
    import numpy as _np

    # Format definitions
    _formats = [
        ("fp32",     None, None),
        ("E5M2 (8b)", 5, 2),
        ("E4M3 (8b)", 4, 3),
        ("E2M5 (8b)", 2, 5),
        ("E1M6 (8b)", 1, 6),
        ("E2M3 (6b)", 2, 3),
        ("E2M1 (4b)", 2, 1),
    ]

    _word1 = word1_input.value.strip().lower()
    _word2 = word2_input.value.strip().lower()

    if _word1 not in glove_model or _word2 not in glove_model:
        _missing = [w for w in [_word1, _word2] if w not in glove_model]
        _out = mo.md(f"**Word not in vocabulary:** {', '.join(_missing)}")
    else:
        _v1 = glove_model[_word1].astype(_np.float64)
        _v2 = glove_model[_word2].astype(_np.float64)

        _exact = float(_np.dot(_v1, _v2) / (_np.linalg.norm(_v1) * _np.linalg.norm(_v2)))

        _rows = []
        for _fname, _eb, _mb in _formats:
            if _eb is None:
                _cos = _exact
                _bits = 32
                _err = 0.0
            else:
                _pv = build_format_values(_eb, _mb)
                _q1 = quantize_vec(_v1, _pv)
                _q2 = quantize_vec(_v2, _pv)
                _cos = float(_np.dot(_q1, _q2) / (_np.linalg.norm(_q1) * _np.linalg.norm(_q2)))
                _bits = 1 + _eb + _mb
                _err = abs(_cos - _exact)
            _rows.append(f"| {_fname} | {_bits} | {_cos:.6f} | {_err:.6f} |")

        _out = mo.md(f"""
        **"{_word1}"** vs **"{_word2}"** — GloVe-50d, per-vector absmax scaling:

        | Format | Bits | Cosine similarity | Error vs fp32 |
        |--------|------|-------------------|---------------|
        {chr(10).join(_rows)}

        The 8-bit formats all agree to ~4 decimal places. Even 6-bit (E2M3)
        is close. Only at 4-bit (E2M1) does the error become visible —
        and it's still small.
        """)
    _out


# ── §2  The ExMy family and scaling ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §2 — The ExMy family

        Every low-precision float is **E**xponent bits + **M**antissa bits.
        More exponent → wider dynamic range. More mantissa → finer
        resolution. For a fixed bit budget, it's a tradeoff.

        The slider below sweeps the **scale factor**: how many σ of the
        data we map to the format's max representable value. Clip too
        tight (2σ) and you lose outliers. Clip too loose (5σ) and you
        waste codes on empty space.
        """
    )


@app.cell
def _(mo):
    scale_slider = mo.ui.slider(
        start=2.0, stop=5.5, step=0.1, value=3.5,
        label="Scale factor (σ):",
        full_width=True,
    )
    scale_slider
    return (scale_slider,)


@app.cell
def _(scale_slider, build_format_values, quantize, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import io as _io

    _rng = _np.random.default_rng(42)
    _x = _rng.standard_normal(256)
    _sigma = _x.std()
    _sf = scale_slider.value

    _fmt_list = [
        ("E1M6 (8b)", 1, 6, "#1565C0"),
        ("E2M5 (8b)", 2, 5, "#2E7D32"),
        ("E4M3 (8b)", 4, 3, "#E65100"),
        ("E5M2 (8b)", 5, 2, "#6A1B9A"),
        ("E2M3 (6b)", 2, 3, "#00838F"),
        ("E2M1 (4b)", 2, 1, "#C62828"),
    ]

    # Sweep scale factors for the heatmap
    _scale_range = _np.arange(2.0, 5.6, 0.1)
    _results = {}
    for _fname, _eb, _mb, _col in _fmt_list:
        _pv = build_format_values(_eb, _mb)
        _errs = []
        for _s in _scale_range:
            _scale = _pv[-1] / (_s * _sigma)
            _q = quantize(_x * _scale, _pv) / _scale
            _errs.append(float(_np.sqrt(_np.mean((_q - _x)**2))))
        _results[_fname] = (_errs, _col)

    # Current scale factor column
    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(12, 4.5),
                                        gridspec_kw={"width_ratios": [2, 1]},
                                        constrained_layout=True)

    # Left: RMS error vs scale factor for all formats
    for _fname, _eb, _mb, _col in _fmt_list:
        _errs, _ = _results[_fname]
        _ax1.plot(_scale_range, _errs, color=_col, lw=2, label=_fname)

    _ax1.axvline(_sf, color="#999", ls="--", lw=1.5, label=f"current ({_sf:.1f}σ)")
    _ax1.set_xlabel("Scale factor (×σ mapped to max representable)")
    _ax1.set_ylabel("RMS quantization error")
    _ax1.set_title("Quantization error vs scaling")
    _ax1.legend(fontsize=8, ncol=2)
    _ax1.spines["top"].set_visible(False)
    _ax1.spines["right"].set_visible(False)

    # Right: bar chart at current scale factor
    _names = []
    _vals = []
    _cols = []
    for _fname, _eb, _mb, _col in _fmt_list:
        _pv = build_format_values(_eb, _mb)
        _scale = _pv[-1] / (_sf * _sigma)
        _q = quantize(_x * _scale, _pv) / _scale
        _err = float(_np.sqrt(_np.mean((_q - _x)**2)))
        _names.append(_fname)
        _vals.append(_err)
        _cols.append(_col)

    _ax2.barh(_names, _vals, color=_cols)
    _ax2.set_xlabel("RMS error")
    _ax2.set_title(f"At {_sf:.1f}σ scaling")
    _ax2.invert_yaxis()
    _ax2.spines["top"].set_visible(False)
    _ax2.spines["right"].set_visible(False)

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150)
    _plt.close(_fig)
    _buf.seek(0)

    # Find best format at current scale
    _best_idx = _np.argmin(_vals)
    _best_name = _names[_best_idx]
    _best_val = _vals[_best_idx]

    mo.vstack([
        mo.image(_buf.read(), width=900),
        mo.md(f"""
        At **{_sf:.1f}σ** scaling, **{_best_name}** has the lowest error
        ({_best_val:.5f}). More mantissa bits (E2M5, E1M6) consistently
        beat more exponent bits (E5M2, E4M3) when the data distribution
        is known and scaling is tight — which is exactly the inference
        situation.

        E5M2 exists for **training**, where gradients span a huge dynamic
        range and you can't pre-calibrate the scale.
        """),
    ])


# ── §3  Distribution meets number format ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §3 — Distribution meets format

        Where are the representable values relative to where the data
        actually lives? Float formats concentrate codes near zero
        (log-spaced); integer formats spread them uniformly. The match
        between format and distribution determines quantization quality.
        """
    )


@app.cell
def _(mo):
    fmt_dropdown = mo.ui.dropdown(
        options=["E2M1 (4b)", "E2M3 (6b)", "E4M3 (8b)", "E2M5 (8b)",
                 "E1M6 (8b)", "E5M2 (8b)"],
        value="E2M1 (4b)",
        label="Format:",
    )
    fmt_dropdown
    return (fmt_dropdown,)


@app.cell
def _(fmt_dropdown, build_format_values, glove_model, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import io as _io

    # Gather GloVe coordinates
    _words = list(glove_model.key_to_index.keys())[:5000]
    _vecs = _np.array([glove_model[w] for w in _words])
    _coords = _vecs.flatten()
    _sigma = _coords.std()

    _fmt_map = {
        "E1M6 (8b)": (1, 6), "E2M5 (8b)": (2, 5),
        "E4M3 (8b)": (4, 3), "E5M2 (8b)": (5, 2),
        "E2M3 (6b)": (2, 3), "E2M1 (4b)": (2, 1),
    }
    _eb, _mb = _fmt_map[fmt_dropdown.value]
    _pv = build_format_values(_eb, _mb)
    _bits = 1 + _eb + _mb

    # Scale: 3.5σ → max representable
    _scale = _pv[-1] / (3.5 * _sigma)
    _scaled_coords = _coords * _scale
    _abs_scaled = _np.abs(_scaled_coords)
    _clipped = _np.mean(_abs_scaled > _pv[-1])

    # ── Code density: 1/(gap between adjacent codes) ──
    _gaps = _np.diff(_pv)
    _midpoints = (_pv[:-1] + _pv[1:]) / 2
    _density = 1.0 / _gaps  # codes per unit
    # Normalize so area ≈ 1 (like a distribution)
    _density_norm = _density / (_density * _gaps).sum()

    # Data histogram for overlay (on same linear scale)
    _hist_vals, _hist_edges = _np.histogram(_abs_scaled, bins=80, density=True)
    _hist_centers = (_hist_edges[:-1] + _hist_edges[1:]) / 2

    # ── Two-panel figure ──
    _fig, (_ax1, _ax2) = _plt.subplots(2, 1, figsize=(10, 8),
                                        constrained_layout=True,
                                        gridspec_kw={"height_ratios": [1, 0.8]})

    # Top: linear picket fence + code density + data distribution
    # Picket fence
    for _v in _pv[1:]:
        _ax1.axvline(_v, color="#43A047", alpha=0.25, lw=0.5, ymin=0, ymax=0.15)
    # Code density curve
    _ax1.plot(_midpoints, _density_norm, color="#43A047", lw=2,
              label="Code density (1/gap, normalized)")
    # Data distribution
    _ax1.fill_between(_hist_centers, _hist_vals, alpha=0.15, color="#E53935")
    _ax1.plot(_hist_centers, _hist_vals, color="#E53935", lw=1.5,
              label="|GloVe coordinates| (scaled)", alpha=0.8)
    # Clipping boundary
    _ax1.axvline(_pv[-1], color="#43A047", lw=2, ls="--", alpha=0.7)

    _ax1.set_xlabel("|x|", fontsize=10)
    _ax1.set_ylabel("Density", fontsize=10)
    _ax1.set_title(f"{fmt_dropdown.value} — code density vs data distribution "
                   f"(linear scale)", fontsize=11)
    _ax1.legend(fontsize=9)
    _ax1.set_xlim(0, _pv[-1] * 1.15)
    _ax1.spines["top"].set_visible(False)
    _ax1.spines["right"].set_visible(False)

    # Bottom: log-scale survival plot (same as before)
    _thresholds = _np.logspace(-3, _np.log10(max(_abs_scaled.max(),
                               _pv[-1]) * 1.2), 400)
    _surv = _np.array([_np.mean(_abs_scaled > t) for t in _thresholds])
    _gauss = _np.abs(_np.random.default_rng(0).normal(0, _abs_scaled.std(),
                                                       size=len(_abs_scaled)))
    _surv_gauss = _np.array([_np.mean(_gauss > t) for t in _thresholds])

    _ax2.loglog(_thresholds, _surv_gauss, color="#1E88E5", lw=2,
                alpha=0.6, label="Matched Gaussian")
    _ax2.loglog(_thresholds, _surv, color="#E53935", lw=2,
                label="GloVe (scaled)")
    for _v in _pv[1:]:
        _ax2.axvline(_v, color="#43A047", alpha=0.15, lw=0.5)
    _ax2.axvline(_pv[-1], color="#43A047", lw=2, ls="--",
                 label=f"Max representable ({_pv[-1]:.1f})", alpha=0.8)

    # Code density + data density on twin axis (log scale)
    _ax2r = _ax2.twinx()
    _ax2r.loglog(_midpoints, _density_norm, color="#43A047", lw=1.5,
                 alpha=0.6, label="Code density")
    # Data density (KDE-like: histogram on log-spaced bins, normalized)
    _log_bins = _np.logspace(_np.log10(max(_abs_scaled[_abs_scaled > 0].min(), 1e-4)),
                              _np.log10(_abs_scaled.max() * 1.1), 80)
    _data_hist, _data_edges = _np.histogram(_abs_scaled, bins=_log_bins, density=True)
    _data_centers = (_data_edges[:-1] + _data_edges[1:]) / 2
    _ax2r.loglog(_data_centers, _data_hist, color="#E53935", lw=1.5,
                 alpha=0.5, ls="--", label="Data density")
    _ax2r.set_ylabel("Density (log)", fontsize=9, color="#43A047")
    _ax2r.tick_params(axis="y", colors="#43A047")
    _ax2r.spines["right"].set_color("#43A047")

    _ax2.set_xlabel("|x| (log scale)", fontsize=10)
    _ax2.set_ylabel("P(|X| > x)", fontsize=10)
    _ax2.set_title("Survival plot + code density (log scale)", fontsize=11)
    # Combined legend
    _h1, _l1 = _ax2.get_legend_handles_labels()
    _h2, _l2 = _ax2r.get_legend_handles_labels()
    _ax2.legend(_h1 + _h2, _l1 + _l2, fontsize=8, loc="lower left")
    _ax2.spines["top"].set_visible(False)

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150)
    _plt.close(_fig)
    _buf.seek(0)

    # Count how many codes land in occupied range
    _data_max = _abs_scaled.max()
    _codes_used = _np.sum(_pv <= _data_max)
    _codes_total = len(_pv)

    mo.vstack([
        mo.image(_buf.read(), width=900),
        mo.md(f"""
        **Top (linear)**: the green picket fence shows every representable
        positive value; the green curve is **code density** — 1/(gap
        between adjacent codes), normalized like a distribution. For
        float formats with ≥2 exponent bits, code density peaks near
        zero and falls off — it *looks like* a data distribution, which
        is why floats match Gaussian-ish data naturally.
        {"With just 1 exponent bit, the density is nearly flat — essentially a symmetric integer." if _eb < 2 else ""}

        **Bottom (log)**: the survival plot from notebook 4. The red
        curve sits above the blue (Gaussian) in the tails — Zipf-law
        heavy tails. Values beyond the dashed line ({_clipped:.2%} of
        data at 3.5σ scaling) get clipped.

        The {_codes_total} positive codes span {_pv[1]:.4f} to
        {_pv[-1]:.1f}; **{_codes_used}** fall within the data range.
        Try switching formats to see how the code density shape changes.
        """),
    ])


# ── §4  What does a MAC cost? ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §4 — What does a MAC cost?

        A multiply-accumulate (MAC) is the atomic operation of neural
        network inference: take two numbers, multiply them, add the result
        to a running sum. Every layer of every model is millions of MACs.

        At the gate level, an **E4M3** (fp8) MAC is:
        """
    )


@app.cell
def _(mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import io as _io

    # Component energy breakdown from h0-pe-8b area-power-estimate.md
    # 5nm @ 0.75V, with activity factors and glitch factors
    _components = [
        ("Multiplier\n(4×4 Dadda)", 7.1, "#E53935"),
        ("Accumulator\n(group adders)", 11.2, "#1E88E5"),
        ("Local acc\n+ carry", 5.5, "#1565C0"),
        ("Field\nextraction", 5.4, "#7B1FA2"),
        ("Exponent\nlogic", 3.8, "#00838F"),
        ("Barrel\nshifter", 3.7, "#2E7D32"),
        ("Bank file\n(amortized)", 2.5, "#F57F17"),
        ("Latches\n(glitch ctrl)", 2.6, "#6D4C41"),
        ("Sign\nconditioning", 1.9, "#455A64"),
        ("Counter\nupdate", 0.7, "#78909C"),
    ]

    _names = [c[0] for c in _components]
    _energies = [c[1] for c in _components]
    _colors = [c[2] for c in _components]
    _total = sum(_energies)

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(12, 5),
                                        constrained_layout=True)

    # Left: horizontal bar chart
    _y = range(len(_names))
    _ax1.barh(_y, _energies, color=_colors)
    _ax1.set_yticks(_y)
    _ax1.set_yticklabels(_names, fontsize=9)
    _ax1.set_xlabel("Energy per cycle (fJ) @ 0.75V, 5nm")
    _ax1.set_title(f"MAC energy breakdown — total {_total:.0f} fJ/MAC")
    _ax1.invert_yaxis()
    _ax1.spines["top"].set_visible(False)
    _ax1.spines["right"].set_visible(False)
    for _i, _e in enumerate(_energies):
        _ax1.text(_e + 0.3, _i, f"{_e:.1f}", va="center", fontsize=8)

    # Right: voltage and process scaling
    _scenarios = [
        ("5nm\n0.75V", 1.0),
        ("5nm\n0.4V", (0.4/0.75)**2),
        ("22nm\n0.4V", (0.4/0.75)**2 * 5),
    ]
    _scaled = [_total * s[1] for s in _scenarios]
    _labels = [s[0] for s in _scenarios]
    _bars = _ax2.bar(range(len(_scenarios)), _scaled,
                     color=["#1E88E5", "#43A047", "#FF9800"])
    _ax2.set_xticks(range(len(_scenarios)))
    _ax2.set_xticklabels(_labels, fontsize=9)
    _ax2.set_ylabel("fJ / MAC")
    _ax2.set_title("Voltage and process scaling")
    _ax2.spines["top"].set_visible(False)
    _ax2.spines["right"].set_visible(False)
    for _i, _v in enumerate(_scaled):
        _ax2.text(_i, _v + 1, f"{_v:.0f}", ha="center", fontsize=10,
                  fontweight="bold")

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150)
    _plt.close(_fig)
    _buf.seek(0)

    _v_scale = (0.4/0.75)**2
    _p_scale = 5  # 22nm vs 5nm area/capacitance ratio

    mo.vstack([
        mo.image(_buf.read(), width=900),
        mo.md(f"""
        **Left**: energy per MAC cycle at 5nm, 0.75V. The multiplier
        itself (a 4×4 Dadda tree: 16 AND gates, 4 half-adders, 8 full-adders)
        is only {_energies[0]:.0f} fJ — the accumulator and supporting
        logic cost more. The total is **{_total:.0f} fJ/MAC**.

        **Right**: scaling. Voltage scaling from 0.75V to 0.4V gives
        ×{_v_scale:.2f} (energy ∝ V²). Moving from 5nm to 22nm multiplies
        capacitance by ~{_p_scale}×, giving **{_scaled[2]:.0f} fJ/MAC** —
        still under 100 fJ. That's {_scaled[2]/2:.0f} fJ per operation
        (MAC = 2 ops), or about **{1e15/(_scaled[2]/2)/1e12:.0f} TOPS/W**
        for the core alone.

        These are real cell energies from 5nm Liberty timing files.
        """),
    ])


@app.cell
def _(mo):
    mo.md(
        """
        ### The table everyone cites — and what's changed

        In 2014, Mark Horowitz presented this table at ISSCC. It became
        the standard reference for "what does compute cost?" It's roughly
        right for 45nm general-purpose logic at 0.9V — but it's two
        decades old, and it gets misapplied to modern inference hardware
        constantly.

        Here it is alongside our 5nm numbers at 0.4V. These aren't
        apples-to-apples — that's the point. The algorithms have changed
        (fp32 → fp8), the architectures have changed (general-purpose →
        purpose-built), and the technology has changed (45nm/0.9V →
        5nm/0.4V). The right question has shifted from "what does a
        32-bit multiply cost?" to "what does an 8-bit MAC cost, including
        everything around it?"

        | Operation | Horowitz 2014 (45nm, 0.9V) | H0 PE (5nm, 0.4V) | Notes |
        |---|---|---|---|
        | 8-bit int add | 30 fJ | ~2 fJ | FA-based |
        | 8-bit int multiply | 200 fJ | 2.0 fJ | E4M3 mantissa: 4×4 Dadda tree |
        | 32-bit int add | 100 fJ | — | not useful for inference |
        | 32-bit FP multiply | 3,700 fJ | — | not useful for inference |
        | **E4M3 multiply** | — | **4.6 fJ** | mult + field extract + exp logic |
        | **E4M3 MAC (full)** | — | **12.8 fJ** | full datapath incl. accumulator |
        | 32-bit SRAM read (8KB) | 5,000 fJ | ~64 fJ | 2 fJ/bit, low-energy sense amp |
        | 32-bit DRAM read | 640,000 fJ | 160,000 fJ | LPDDR/HBM, sustained sequential ⚠ |

        ⚠ DRAM energy is for sustained sequential reads (5 pJ/bit),
        not random access. HBM is broadly similar per bit moved.

        The story: Horowitz showed that **data movement dominates
        compute**. That's still true — but the ratio has shifted. With
        8-bit MACs at ~13 fJ and SRAM at ~64 fJ, compute is now cheap
        enough that the entire design challenge is keeping the datapath
        fed.

        Traditionally, the multiplier was the energy bottleneck — its
        gate count grows as the **square** of the bit-width (an N×N
        multiplier needs ~N² AND gates). But that same quadratic
        scaling works in your favour at low precision: a 4×4 multiplier
        (E4M3 mantissa) has 16 AND gates; a 2×2 (E2M1) has 4. At these
        sizes the multiplier is no longer the dominant cost — the
        accumulator, shifter, and data movement are.

        *Source: M. Horowitz,
        ["Computing's Energy Problem (and what we can do about it),"](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf)
        ISSCC 2014.*
        """
    )


# ── §5  Where do the joules go? ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §5 — Where do the joules go?

        The MAC core is cheap. But you have to **feed** it — fetch
        operands from memory, deliver results through the network.
        Data movement dominates.
        """
    )


@app.cell
def _(mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import io as _io

    # System energy budget at 22nm, 0.4V
    # Source: h0-pe-8b/docs/area-power-estimate.md §4
    # Core E4M3: 64 fJ/MAC = 32 fJ/OP
    # System overhead (CRAM + NoC + clock + ctrl): ~10 fJ/OP conservative
    #   NoC energy amortized over ~128 MACs per transfer, controller similar
    # E2M1 core: 7.3 fJ/cycle (h0-pe-4b postscale), overhead ~5 fJ/OP (less data to move)

    _labels = ["E4M3 (8b)", "E2M1 (4b)"]
    _core =     [32, 7.3]
    _overhead = [10, 5]
    _totals =   [c + o for c, o in zip(_core, _overhead)]
    _tops_w =   [1e15 / (t * 1e-15) / 1e12 for t in _totals]

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(12, 5),
                                        constrained_layout=True)

    # Left: stacked bar — fJ/OP breakdown
    _x = range(len(_labels))
    _ax1.bar(_x, _core, color="#1E88E5", label="MAC core")
    _ax1.bar(_x, _overhead, bottom=_core, color="#FF9800",
             label="System overhead\n(CRAM + NoC + ctrl)")
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels(_labels, fontsize=10)
    _ax1.set_ylabel("fJ / OP")
    _ax1.set_title("Energy per operation @ 22nm, 0.4V")
    _ax1.legend(fontsize=9, loc="upper left")
    _ax1.spines["top"].set_visible(False)
    _ax1.spines["right"].set_visible(False)
    for _i in range(len(_labels)):
        _ax1.text(_i, _totals[_i] + 1, f"{_totals[_i]:.0f} fJ/OP",
                  ha="center", fontsize=10, fontweight="bold")

    # Right: TOPS/W (derived)
    _ax2.barh(range(len(_labels)), _tops_w,
              color=["#1E88E5", "#43A047"])
    _ax2.set_yticks(range(len(_labels)))
    _ax2.set_yticklabels(_labels, fontsize=10)
    _ax2.set_xlabel("TOPS/W (system)")
    _ax2.set_title("Efficiency (higher is better)")
    _ax2.invert_yaxis()
    _ax2.spines["top"].set_visible(False)
    _ax2.spines["right"].set_visible(False)
    for _i, _v in enumerate(_tops_w):
        _ax2.text(_v + 0.5, _i, f"{_v:.0f}", va="center", fontsize=10,
                  fontweight="bold")

    # Reference line: Boqueria/SpeedAI MLPerf
    _ax2.axvline(20, color="#999", ls="--", lw=1.5, alpha=0.6)
    _ax2.text(21, 0.5, "Boqueria MLPerf\n~20 TOPS/W",
              fontsize=8, color="#666", va="center")

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150)
    _plt.close(_fig)
    _buf.seek(0)

    mo.vstack([
        mo.image(_buf.read(), width=900),
        mo.md(f"""
        **Left**: energy per operation, split into MAC core (blue) and
        system overhead (orange). The core energy comes from cell-level
        analysis (§4). The overhead — CRAM bitline, NoC wire energy,
        clock, control — is conservatively estimated at ~10 fJ/OP for
        E4M3: NoC transfer energy is amortized over ~128 MACs, and
        controller overhead is similar.

        **Right**: the resulting system efficiency. E4M3 at
        **{_tops_w[0]:.0f} TOPS/W** is comfortably above the
        Boqueria/Speed AI
        [MLPerf result](https://mlcommons.org/benchmarks/inference-datacenter/)
        (~20 TOPS/W). E2M1 at **{_tops_w[1]:.0f} TOPS/W** benefits from
        both a smaller multiplier (quadratic in bit-width) and less data
        to move.

        All numbers are for 22nm @ 0.4V.
        """),
    ])


# ── §6  Why custom silicon ──


@app.cell
def _(mo):
    mo.md(
        """
        ## §6 — Why custom silicon

        Three properties of inference make it uniquely suited to
        purpose-built hardware:

        1. **The operation is (almost) uniform.** A transformer
           layer is overwhelmingly MACs — but not *entirely*. There are
           nonlinearities (SiLU, GELU), normalization (RMSNorm), and
           softmax. A naive accelerator that only does matrix-vector
           products hits **Amdahl's Law**: the 5% of runtime that isn't
           GEMV becomes the bottleneck. The trick is that these
           "non-MAC" operations can themselves be decomposed into MACs
           — SiLU is a polynomial approximation, softmax is
           exponentiation via multiply-and-shift, RMSNorm is a
           reciprocal square root. Build the MAC well enough, and it
           handles everything.

        2. **The precision is low.** We've seen that 8-bit — even 4-bit
           — preserves the geometry. Low precision means small
           multipliers, narrow data paths, less memory bandwidth. Every
           halving of bit-width can quarter the energy.

        3. **The volume is enormous.** A single GPT-4 inference is
           trillions of MACs. At this scale, even femtojoule savings per
           operation compound into watts at the system level. A 10×
           efficiency advantage means 10× the throughput per rack, or
           10× less cooling.

        This is why the industry is building inference-specific silicon:
        not because GPUs can't do inference, but because they waste
        energy on flexibility the workload doesn't need. The dot product
        — the same operation you explored in
        [notebook 3](../dot-product) — is the only primitive that
        matters, and it deserves hardware built around it.

        ---

        **Next**: [PCA](../pca) — discovering structure in the embedding
        space.

        **References**:
        [Dettmers et al. 2022 "LLM.int8()"](https://arxiv.org/abs/2208.07339) •
        [Dettmers blog "The case for 4-bit"](https://timdettmers.com/2022/08/15/which-gpu-for-deep-learning/) •
        [MLCommons MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) •
        [OCP Microscaling Formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
        """
    )


# ── boilerplate ──

@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
