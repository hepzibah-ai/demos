import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from device_model import (
        device_summary_md, device_sliders, device_from_sliders,
    )
    from spice_runner import run_balancer

    return (
        Path,
        device_from_sliders,
        device_sliders,
        device_summary_md,
        mo,
        np,
        plt,
        run_balancer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Switched-Capacitor Mid-Point Balancer

    ## The series-string idea

    Processing elements (PEs) operate at 0.3–0.5 V and draw ~1 A each.
    Instead of giving each PE its own voltage regulator, we **stack PEs in
    series** so the same current flows through all of them:

    ```
        Vin (battery, 0.9–1.6 V)
         │
        ┌┴┐
        │ │  String supply (buck converter — see separate notebook)
        └┬┘
         ├─── Vstring = N × Vpe
         │
        ┌┴┐
        │ │  PE N  (top)       ← draws ~1 A
        └┬┘
         ├─── Vmid
         │   ↕ balancer (handles ±10% current mismatch)
        ┌┴┐
        │ │  PE 1  (bottom)    ← draws ~1 A
        └┬┘
         │
        GND
    ```

    **If the PEs draw the same current**, the mid-point balancer does no
    work — current simply flows through. The balancer only handles the
    **mismatch** (±10% of load current), so it is small and low-power.

    ## Efficiency targets

    Our overall target is **90% wall-to-battery efficiency**. Not all
    watts are equal — the string supply carries the full load current,
    while the balancer only handles the mismatch:

    | Block | Current | Efficiency target | Why |
    |-------|---------|-------------------|-----|
    | String supply | 100% of load | **95%** | Carries all the power; every % counts |
    | Balancer | ≤10% of load (worst case) | **50%** | Only 10% of current → 50% × 10% = 5% system loss |
    | **System** | | **~90%** | 95% × (1 − 0.05) ≈ 90% |

    The 95% string-supply target over a wide input range (0.9–1.6 V)
    strongly favors an **inductive buck converter** — a switched-capacitor
    converter can only produce fixed rational fractions of Vin, and the
    ratio loss at unfavorable voltages is too high. The string supply is
    covered in a [separate notebook](./pol_inductive.py).

    **This notebook focuses on the switched-capacitor balancer**, where
    the relaxed efficiency target (50%) and small current make SC a
    natural fit: simple, no magnetics, small die area.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Balancer topology

    The balancer uses a **flying-capacitor** charge pump to shuttle charge
    between the two string segments. We start with the simplest version:
    a **1:1 converter** from the battery into the bottom segment.

    - Phase 1 (S1, S2 on): Cfly charges from battery (Vstring → GND)
    - Phase 2 (S3, S4 on): Cfly discharges to bottom segment (Vmid → GND)

    All switches are **NMOS** (the partner's process). High-side switches
    need a bootstrap or charge-pump gate drive.

    ### Physical load model

    Each PE is modeled as:
    - **Capacitor** (Cpe) — represents ~0.5 mm² of well capacitance
      in 28nm at the PE operating voltage. This is the intrinsic
      decoupling the silicon gives you, per square millimeter.
    - **Current source** — represents the PE's DC power consumption.
    - **Resistor** (Rpe) — Norton-equivalent load of the PE's
      switched-capacitor compute activity. This damps the PE node
      and models the continuous charge shuffling of operation.

    The simulation starts balanced (both PEs at Vpe, both drawing
    the same current). Then the bottom PE's current steps down,
    creating a mismatch that the balancer must handle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reference: Flying Capacitor Converter

    The diagram shows both phases of a basic divide-by-2 flying capacitor
    converter. The balancer reuses this switch fabric.
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    _here = Path(__file__).resolve().parent
    mo.image(src=str(_here / "figures" / "flying_capacitor.png"))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Parameters
    """)
    return


@app.cell
def _(mo):
    ui_vpe = mo.ui.slider(
        start=0.3, stop=0.5, step=0.01, value=0.4,
        label="Voltage per PE (V)",
    )
    ui_istring = mo.ui.slider(
        start=10, stop=500, step=10, value=100,
        label="String current (mA)",
    )
    ui_imismatch = mo.ui.slider(
        start=1, stop=50, step=1, value=10,
        label="Mismatch step (mA)",
    )
    mo.vstack([
        mo.md("### Application"),
        ui_vpe,
        mo.hstack([ui_istring, ui_imismatch], justify="start"),
    ])
    return ui_imismatch, ui_istring, ui_vpe


@app.cell
def _(device_sliders, mo):
    dev_ui = device_sliders(mo)
    mo.vstack([
        mo.md("### Device Technology"),
        mo.hstack([dev_ui["lg"], dev_ui["tox"]], justify="start"),
        mo.hstack([dev_ui["vgs"], dev_ui["vth"]], justify="start"),
        mo.hstack([dev_ui["mobility_pct"], dev_ui["ron_sp"]], justify="start"),
    ])
    return (dev_ui,)


@app.cell
def _(mo):
    ui_fsw = mo.ui.slider(
        start=10, stop=500, step=10, value=100,
        label="Switching frequency (MHz)",
    )
    ui_cfly = mo.ui.slider(
        start=0.1, stop=10.0, step=0.1, value=1.0,
        label="Flying capacitor (nF)",
    )
    ui_cpe = mo.ui.slider(
        start=0.1, stop=5.0, step=0.1, value=0.5,
        label="PE load capacitance (nF)",
    )
    ui_rpe = mo.ui.slider(
        start=1.0, stop=20.0, step=0.5, value=4.0,
        label="PE Norton resistance (Ω)",
    )
    ui_w_sw = mo.ui.slider(
        start=10, stop=1000, step=10, value=300,
        label="Switch width (µm)",
    )
    mo.vstack([
        mo.md("### Balancer Design"),
        mo.hstack([ui_fsw, ui_cfly], justify="start"),
        mo.hstack([ui_cpe, ui_rpe], justify="start"),
        ui_w_sw,
    ])
    return ui_cfly, ui_cpe, ui_fsw, ui_rpe, ui_w_sw


@app.cell
def _(dev_ui, device_from_sliders, device_summary_md, mo):
    dev = device_from_sliders(dev_ui)
    mo.md(f"### Device Model\n\n{device_summary_md(dev)}")
    return


@app.cell(hide_code=True)
def _(dev_ui, mo, ui_cfly, ui_fsw, ui_vpe, ui_w_sw):
    # Ron × Cfly time constant
    _W = ui_w_sw.value * 1e-6
    _L = dev_ui["lg"].value * 1e-9
    _Ron_sp = dev_ui["ron_sp"].value * 1e-3 * 1e-6  # mΩ·mm² → Ω·m²
    _Ron = _Ron_sp / (_W * _L) if _W * _L > 0 else float('inf')
    _Cfly = ui_cfly.value * 1e-9
    _tau = _Ron * _Cfly
    _Thalf = 0.5 / (ui_fsw.value * 1e6)

    _Vgs = dev_ui["vgs"].value
    _Vth = dev_ui["vth"].value
    _Vstring = 2 * ui_vpe.value
    _Vgs_eff_hi = _Vgs - _Vstring  # worst-case high-side Vgs

    mo.md(f"""
    ### Design Point

    | Parameter | Value | Notes |
    |-----------|-------|-------|
    | Vstring | {_Vstring:.2f} V | 2 × Vpe (ideal string regulator) |
    | Switch Ron | {_Ron*1e3:.1f} mΩ | W={ui_w_sw.value}µm, L={dev_ui['lg'].value}nm |
    | Ron × Cfly | **{_tau*1e9:.2f} ns** | Target ≤ 1 ns for full charge transfer |
    | Half-period | {_Thalf*1e9:.1f} ns | {ui_fsw.value} MHz |
    | τ / T_half | {_tau/_Thalf*100:.0f}% | Want ≪ 100% |
    | Cfly charge/cycle | {_Cfly * ui_vpe.value * 1e12:.0f} pC | At Vpe = {ui_vpe.value}V |
    | Max Imis at 100mV error | {_Cfly * 0.1 * ui_fsw.value * 1e6 * 1e3:.0f} mA | Cfly × Verror × fsw |
    | | | |
    | Gate drive (Vgs) | {_Vgs:.1f} V | {"⚠️ Need bootstrap" if _Vgs < _Vstring + _Vth else "OK"} |
    | High-side Vgs,eff | {_Vgs_eff_hi:.2f} V | Vgs − Vstring; need > Vth ({_Vth:.2f}V) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SPICE Transient Simulation

    [ASU PTM 90nm](https://mec.umn.edu/ptm) BSIM4 NMOS model.
    The string supply is ideal (Vstring = 2 × Vpe). Each PE is
    modeled as C + I + R (capacitance, DC current, Norton-equivalent
    compute load). The simulation starts balanced, then the bottom
    PE's current steps down to create a mismatch.
    """)
    return


@app.cell
def _(
    dev_ui,
    np,
    plt,
    run_balancer,
    ui_cfly,
    ui_cpe,
    ui_fsw,
    ui_imismatch,
    ui_istring,
    ui_rpe,
    ui_vpe,
    ui_w_sw,
):
    _wf = run_balancer(
        Vpe=ui_vpe.value,
        Vgs=dev_ui["vgs"].value,
        W_um=ui_w_sw.value,
        L_nm=dev_ui["lg"].value,
        Cfly_nF=ui_cfly.value,
        Cpe_nF=ui_cpe.value,
        Rpe_ohm=ui_rpe.value,
        Istring_mA=ui_istring.value,
        Imismatch_mA=ui_imismatch.value,
        fsw_MHz=ui_fsw.value,
    )
    _t = _wf["t_ns"]
    _step = _wf["step_time_ns"]

    _fig, _axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    # PE voltages
    _axes[0].plot(_t, _wf["v_bot"] * 1e3, label="V_bot (bottom PE)",
                  color="#4e79a7", linewidth=1.5)
    _axes[0].plot(_t, _wf["v_top"] * 1e3, label="V_top (top PE)",
                  color="#e15759", linewidth=1.5)
    _axes[0].axhline(ui_vpe.value * 1e3, color="gray", ls="--", alpha=0.5,
                     label=f"Target {ui_vpe.value*1e3:.0f}mV")
    _axes[0].axvline(_step, color="k", ls=":", alpha=0.5,
                     label="Mismatch step")
    _axes[0].set_ylabel("PE voltage (mV)")
    _axes[0].legend(fontsize=8, loc="right")
    _axes[0].grid(True, alpha=0.3)

    # Flying cap voltage
    _axes[1].plot(_t, _wf["v_cfly"] * 1e3, color="#59a14f", linewidth=1.5)
    _axes[1].axvline(_step, color="k", ls=":", alpha=0.5)
    _axes[1].set_ylabel("Vcfly (mV)")
    _axes[1].grid(True, alpha=0.3)

    # String supply current (clipped for visibility)
    _i_clip = np.clip(_wf["i_string_mA"], -500, 500)
    _axes[2].plot(_t, _i_clip, color="#f28e2b", linewidth=0.8)
    _axes[2].axvline(_step, color="k", ls=":", alpha=0.5)
    _axes[2].set_ylabel("I_string (mA)")
    _axes[2].set_xlabel("Time (ns)")
    _axes[2].grid(True, alpha=0.3)

    _fig.suptitle(
        f"1:1 SC Balancer — Vstring={2*ui_vpe.value:.1f}V, "
        f"W={ui_w_sw.value}µm, Cfly={ui_cfly.value}nF, "
        f"{ui_fsw.value}MHz\n"
        f"Bottom PE steps from {ui_istring.value}mA to "
        f"{ui_istring.value - ui_imismatch.value}mA at t={_step:.0f}ns",
        fontsize=10,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, ui_cfly, ui_fsw, ui_vpe):
    _Verror = ui_vpe.value  # worst case: full Vpe
    _I_max = ui_cfly.value * 1e-9 * _Verror * ui_fsw.value * 1e6 * 1e3
    _I_at_100mV = ui_cfly.value * 1e-9 * 0.1 * ui_fsw.value * 1e6 * 1e3

    mo.md(f"""
    ## Why 1:1 isn't enough

    The 1:1 topology has a fundamental problem: it can only source
    current proportional to the **error voltage** on the PE it's
    trying to correct.

    $$I_{{bal}} = C_{{fly}} \\times V_{{error}} \\times f_{{sw}}$$

    At the current settings:
    - To deliver 10 mA, need **100 mV** of error ({ui_cfly.value} nF × 0.1V × {ui_fsw.value} MHz)
    - 100 mV is **{0.1/ui_vpe.value*100:.0f}%** of Vpe — that's a huge regulation error
    - Max current (Verror = Vpe = {ui_vpe.value}V): **{_I_max:.0f} mA**

    The SPICE waveforms above confirm this: the PE voltages drift
    significantly before the balancer can respond. The 1:1 topology
    charges Cfly from Vstring (= 2×Vpe) and dumps into Vmid (≈ Vpe),
    so half the transferred energy is wasted as ratio loss.

    **Solution: a 3:2 step-up** provides built-in overdrive, allowing
    the balancer to source current near zero error voltage. This is
    covered in the next section of this notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    *SPICE waveforms powered by [ngspice](https://ngspice.sourceforge.io/)
    with [ASU Predictive Technology Models](https://mec.umn.edu/ptm) (90nm bulk BSIM4).
    The PTM model is a reasonable proxy for device behavior — substitute
    your own model for quantitative design.*
    """)
    return


if __name__ == "__main__":
    app.run()
