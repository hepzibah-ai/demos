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

        Pick two words below. We'll compute their cosine similarity at
        full precision, then quantize **both vectors** to each format
        before recomputing. The result barely moves — until 4-bit.
        """
    )


@app.cell
def _(mo):
    _w1 = mo.ui.text(value="king", label="Word 1:", full_width=False)
    _w2 = mo.ui.text(value="queen", label="Word 2:", full_width=False)
    mo.hstack([_w1, _w2], gap=1)
    return (_w1, _w2)


@app.cell
def _(_w1, _w2, mo):
    import numpy as _np
    import gensim.downloader as _api

    _model = _api.load("glove-wiki-gigaword-50")

    # ── Quantizer helpers (shared across sections) ──

    def _build_format_values(_ebits, _mbits):
        """Non-negative representable values for ExMy (no NaN/Inf, with subnormals)."""
        _bias = 2**(_ebits - 1) - 1
        _max_e = (1 << _ebits) - 1
        _vals = set()
        for _m in range(1 << _mbits):
            _vals.add(2.0**(1 - _bias) * (_m / (1 << _mbits)))
        for _e in range(1, _max_e + 1):
            for _m in range(1 << _mbits):
                _vals.add(2.0**(_e - _bias) * (1.0 + _m / (1 << _mbits)))
        return _np.array(sorted(_vals))

    def _quantize(_x, _pv):
        """Round-to-nearest, clip to ±max."""
        _s = _np.sign(_x)
        _a = _np.clip(_np.abs(_x), 0, _pv[-1])
        _idx = _np.searchsorted(_pv, _a)
        _idx = _np.clip(_idx, 0, len(_pv) - 1)
        _lo = _np.clip(_idx - 1, 0, len(_pv) - 1)
        _d_lo = _np.abs(_a - _pv[_lo])
        _d_hi = _np.abs(_a - _pv[_idx])
        _best = _np.where(_d_lo <= _d_hi, _lo, _idx)
        return _pv[_best] * _s

    def _quantize_vec(_v, _pv):
        """Per-vector absmax scaling then quantize."""
        _scale = _pv[-1] / _np.abs(_v).max()
        return _quantize(_v * _scale, _pv) / _scale

    # Format definitions used throughout notebook
    _formats = [
        ("fp32",     None, None),
        ("E5M2 (8b)", 5, 2),
        ("E4M3 (8b)", 4, 3),
        ("E2M5 (8b)", 2, 5),
        ("E1M6 (8b)", 1, 6),
        ("E2M3 (6b)", 2, 3),
        ("E2M1 (4b)", 2, 1),
    ]

    _word1 = _w1.value.strip().lower()
    _word2 = _w2.value.strip().lower()

    if _word1 not in _model or _word2 not in _model:
        _missing = [w for w in [_word1, _word2] if w not in _model]
        mo.md(f"**Word not in vocabulary:** {', '.join(_missing)}")
    else:
        _v1 = _model[_word1].astype(_np.float64)
        _v2 = _model[_word2].astype(_np.float64)

        _exact = float(_np.dot(_v1, _v2) / (_np.linalg.norm(_v1) * _np.linalg.norm(_v2)))

        _rows = []
        for _fname, _eb, _mb in _formats:
            if _eb is None:
                _cos = _exact
                _bits = 32
                _err = 0.0
            else:
                _pv = _build_format_values(_eb, _mb)
                _q1 = _quantize_vec(_v1, _pv)
                _q2 = _quantize_vec(_v2, _pv)
                _cos = float(_np.dot(_q1, _q2) / (_np.linalg.norm(_q1) * _np.linalg.norm(_q2)))
                _bits = 1 + _eb + _mb
                _err = abs(_cos - _exact)
            _rows.append(f"| {_fname} | {_bits} | {_cos:.6f} | {_err:.6f} |")

        mo.md(f"""
        **"{_word1}"** vs **"{_word2}"** — GloVe-50d, per-vector absmax scaling:

        | Format | Bits | Cosine similarity | Error vs fp32 |
        |--------|------|-------------------|---------------|
        {chr(10).join(_rows)}

        The 8-bit formats all agree to ~4 decimal places. Even 6-bit (E2M3)
        is close. Only at 4-bit (E2M1) does the error become visible —
        and it's still small.
        """)

    return (_build_format_values, _formats, _model, _quantize, _quantize_vec)


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
def _(scale_slider, _build_format_values, _quantize, mo):
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
        _pv = _build_format_values(_eb, _mb)
        _errs = []
        for _s in _scale_range:
            _scale = _pv[-1] / (_s * _sigma)
            _q = _quantize(_x * _scale, _pv) / _scale
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
        _pv = _build_format_values(_eb, _mb)
        _scale = _pv[-1] / (_sf * _sigma)
        _q = _quantize(_x * _scale, _pv) / _scale
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
        options=["E1M6 (8b)", "E2M5 (8b)", "E4M3 (8b)", "E5M2 (8b)",
                 "E2M3 (6b)", "E2M1 (4b)"],
        value="E4M3 (8b)",
        label="Format:",
    )
    fmt_dropdown
    return (fmt_dropdown,)


@app.cell
def _(fmt_dropdown, _build_format_values, _model, mo):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import io as _io

    # Gather GloVe coordinates
    _words = list(_model.key_to_index.keys())[:5000]
    _vecs = _np.array([_model[w] for w in _words])
    _coords = _vecs.flatten()
    _sigma = _coords.std()

    _fmt_map = {
        "E1M6 (8b)": (1, 6), "E2M5 (8b)": (2, 5),
        "E4M3 (8b)": (4, 3), "E5M2 (8b)": (5, 2),
        "E2M3 (6b)": (2, 3), "E2M1 (4b)": (2, 1),
    }
    _eb, _mb = _fmt_map[fmt_dropdown.value]
    _pv = _build_format_values(_eb, _mb)
    _bits = 1 + _eb + _mb

    # Scale: 3.5σ → max representable
    _scale = _pv[-1] / (3.5 * _sigma)
    _scaled_coords = _coords * _scale
    _abs_scaled = _np.abs(_scaled_coords)

    # Survival plot
    _thresholds = _np.logspace(-3, _np.log10(max(_abs_scaled.max(), _pv[-1]) * 1.2), 400)
    _surv = _np.array([_np.mean(_abs_scaled > t) for t in _thresholds])

    # Gaussian comparison
    _gauss = _np.abs(_np.random.default_rng(0).normal(0, _abs_scaled.std(),
                                                        size=len(_abs_scaled)))
    _surv_gauss = _np.array([_np.mean(_gauss > t) for t in _thresholds])

    _fig, _ax = _plt.subplots(figsize=(10, 5), constrained_layout=True)

    _ax.loglog(_thresholds, _surv_gauss, color="#1E88E5", lw=2,
               alpha=0.6, label="Matched Gaussian")
    _ax.loglog(_thresholds, _surv, color="#E53935", lw=2,
               label="GloVe (scaled)")

    # Rug plot of representable values
    for _v in _pv[1:]:  # skip zero
        _ax.axvline(_v, color="#43A047", alpha=0.15, lw=0.5)
    # Mark clipping boundary
    _ax.axvline(_pv[-1], color="#43A047", lw=2, ls="--",
                label=f"Max representable ({_pv[-1]:.1f})", alpha=0.8)

    _clipped = _np.mean(_abs_scaled > _pv[-1])
    _ax.set_xlabel("|x| (scaled)", fontsize=10)
    _ax.set_ylabel("P(|X| > x)", fontsize=10)
    _ax.set_title(f"{fmt_dropdown.value} — {2**_bits} codepoints, "
                  f"{_clipped:.2%} clipped at 3.5σ scaling", fontsize=11)
    _ax.legend(fontsize=9)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

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
        The green lines are the {_codes_total} positive representable values.
        **{_codes_used}** of them fall within the data range.
        {"Float formats concentrate codes near zero — dense where the data is densest." if _eb >= 2 else "With just 1 exponent bit, this format is nearly uniform — like a symmetric integer."}

        The red curve sits above the blue (Gaussian) in the tails — that's
        the Zipf-law heavy tails from notebook 4. Values beyond the dashed
        line ({_clipped:.2%} of data) get clipped.

        Try switching formats to see how the code density and clipping
        boundary change.
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

        At the gate level, an 8-bit MAC is:
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

        | Operation | Horowitz 2014 | H0 PE | | Notes |
        | | 45nm, 0.9V | 5nm, 0.4V | | |
        |---|---|---|---|---|
        | 8-bit int add | 30 fJ | ~2 fJ | | FA-based, ×0.284 voltage, ~5× process |
        | 8-bit int multiply | 200 fJ | 2.0 fJ | | 4×4 Dadda tree |
        | 32-bit int add | 100 fJ | — | | not a useful operation for inference |
        | 32-bit FP multiply | 3,700 fJ | — | | not a useful operation for inference |
        | **FP8 E4M3 multiply** | — | **4.6 fJ** | | mult + field extract + exponent logic |
        | **FP8 E4M3 MAC** | — | **12.8 fJ** | | full datapath including accumulator |
        | 32-bit SRAM read (8KB) | 5,000 fJ | ~20 fJ | | 5nm SRAM, 32-bit word |
        | 32-bit DRAM read | 640,000 fJ | 160,000 fJ | ⚠ | LPDDR/HBM, long sequential reads |

        ⚠ DRAM energy is per-access for sustained sequential reads,
        not random access. HBM is broadly similar per bit moved.

        The story: Horowitz showed that **data movement dominates
        compute**. That's still true — but the ratio has shifted. With
        8-bit MACs at 5 fJ and SRAM at 20 fJ, compute is now cheap
        enough that the entire design challenge is keeping the datapath
        fed. Every architectural decision we make is about minimizing
        data movement.

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
    # Core MAC: ~64 fJ/MAC = ~32 fJ/OP
    # CRAM fetch: estimated from h0 analysis
    # NoC: network-on-chip delivery
    # Other: clock, control, I/O amortized
    _budget = {
        "MAC core": 32,
        "CRAM fetch": 22,
        "NoC (data movement)": 15,
        "Clock + control": 8,
        "I/O (amortized)": 5,
    }
    _total = sum(_budget.values())

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(12, 5),
                                        constrained_layout=True,
                                        gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: pie chart
    _colors = ["#1E88E5", "#E53935", "#FF9800", "#78909C", "#BDBDBD"]
    _wedges, _texts, _autotexts = _ax1.pie(
        _budget.values(), labels=_budget.keys(),
        autopct=lambda p: f"{p:.0f}%\n({p*_total/100:.0f} fJ)",
        colors=_colors, startangle=90,
        textprops={"fontsize": 9},
    )
    for _t in _autotexts:
        _t.set_fontsize(8)
    _ax1.set_title(f"System energy budget\n{_total} fJ/OP @ 22nm, 0.4V",
                   fontsize=11)

    # Right: TOPS/W comparison
    _configs = [
        ("MAC core\nonly", 32, "#1E88E5"),
        ("+ fetch\n(CRAM)", 32 + 22, "#E53935"),
        ("+ NoC", 32 + 22 + 15, "#FF9800"),
        ("Full\nsystem", _total, "#455A64"),
        ("4-bit\n(E2M1)", 14, "#43A047"),  # h0-pe-4b postscale
    ]
    _tops_w = [1e15 / (c[1] * 1e-15) / 1e12 for c in _configs]

    _bars = _ax2.bar(range(len(_configs)), _tops_w,
                     color=[c[2] for c in _configs])
    _ax2.set_xticks(range(len(_configs)))
    _ax2.set_xticklabels([c[0] for c in _configs], fontsize=9)
    _ax2.set_ylabel("TOPS/W")
    _ax2.set_title("Efficiency at each level")
    _ax2.spines["top"].set_visible(False)
    _ax2.spines["right"].set_visible(False)
    for _i, _v in enumerate(_tops_w):
        _ax2.text(_i, _v + 0.3, f"{_v:.1f}", ha="center", fontsize=9,
                  fontweight="bold")

    # Reference line: Boqueria/SpeedAI MLPerf
    _ax2.axhline(20, color="#999", ls="--", lw=1.5, alpha=0.6)
    _ax2.text(len(_configs) - 0.5, 20.5, "Boqueria MLPerf\n(~20 TOPS/W)",
              fontsize=8, color="#666", ha="right")

    _buf = _io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=150)
    _plt.close(_fig)
    _buf.seek(0)

    _compute_frac = _budget["MAC core"] / _total * 100
    _move_frac = (_budget["CRAM fetch"] + _budget["NoC (data movement)"]) / _total * 100

    mo.vstack([
        mo.image(_buf.read(), width=900),
        mo.md(f"""
        The MAC core is only **{_compute_frac:.0f}%** of system energy.
        Data movement (CRAM fetch + NoC) is **{_move_frac:.0f}%**. This
        is the fundamental challenge: **moving a byte costs more than
        multiplying it**.

        The rightmost bar shows what 4-bit inference buys: the multiplier
        shrinks dramatically (2×2 instead of 4×4), and you fetch half the
        bits. The Boqueria/Speed AI
        [MLPerf result](https://mlcommons.org/benchmarks/inference-datacenter/)
        at ~20 TOPS/W is an existence proof that this level of efficiency
        is achievable in real silicon.

        Note: all numbers here are for 22nm @ 0.4V (our process target).
        At 5nm the core energy drops to ~13 fJ/MAC, but the system
        overhead ratios stay similar.
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

        1. **The operation is uniform.** It's MACs all the way down.
           Every layer, every token, every model. A GPU provides
           thousands of operations you'll never use; an inference
           accelerator provides one operation done as efficiently as
           physics allows.

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
        [notebook 3](../dot-product) — is the only operation that
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
