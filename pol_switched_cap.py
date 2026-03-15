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
        device_summary_md, gate_drive_loss,
        device_sliders, device_from_sliders,
    )
    return (
        device_from_sliders,
        device_sliders,
        device_summary_md,
        gate_drive_loss,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Switched-Capacitor Series-String Power Supply

    ## The idea

    Processing elements (PEs) operate at 0.3–0.5 V and draw ~1 A each.
    Instead of giving each PE its own voltage regulator, we **stack PEs in
    series** so the same current flows through all of them:

    ```
        Vin (battery)
         │
        ┌┴┐
        │ │  String supply — regulates total string voltage
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

    This notebook explores the **switched-capacitor** approach for both the
    string supply and the balancer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gear shifting

    A switched-capacitor converter produces an output that is a **fixed
    rational fraction** of the input. With one flying capacitor you get
    two gears:

    | Gear | Vout/Vin | Switches | Flying cap |
    |------|----------|----------|------------|
    | 1:1 (bypass) | 1 | 2 | none |
    | 2:1 (step-down) | 1/2 | 4 | 1 |

    When a gear's ideal output exceeds the target, the excess drops across
    switch resistance. The resulting **ratio efficiency** is:

    $$\eta_{ratio} = \frac{V_{target}}{k \cdot V_{in}}$$

    where $k$ is the gear ratio (1 or 1/2). You want the gear whose ideal
    output is **closest to, but above**, the target.

    As the battery discharges from 1.6 V to 0.9 V, the optimal gear shifts:

    - **Full charge** (1.6 V): 2:1 → ideal 0.80 V ≈ target → near-perfect
    - **Mid-life** (1.2 V): both gears overshoot — pick the closer one
    - **End of life** (0.9 V): 1:1 → 0.9 V, regulate down (89% ratio eff.)

    All switches are **NMOS** (the partner's process). High-side switches
    need a bootstrap or charge-pump gate drive.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2:1 Flying Capacitor Converter

    The diagram shows both phases of a basic divide-by-2 flying capacitor
    converter. C1 is the flying cap; C2 and C3 are output and input filter
    caps (off-chip). Four switches (SW1–SW4) reconfigure C1 between series
    (charging) and parallel (discharging) with C2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="figures/flying_capacitor.png")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Parameters
    """)
    return


@app.cell
def _(mo):
    ui_vin_max = mo.ui.slider(
        start=1.0, stop=3.0, step=0.1, value=1.6,
        label="Vin max — full charge (V)",
    )
    ui_vin_min = mo.ui.slider(
        start=0.5, stop=2.0, step=0.1, value=0.9,
        label="Vin min — end of life (V)",
    )
    ui_n_stages = mo.ui.slider(
        start=2, stop=8, step=1, value=2,
        label="String length (series PEs)",
    )
    ui_vpe = mo.ui.slider(
        start=0.3, stop=0.5, step=0.01, value=0.4,
        label="Voltage per PE (V)",
    )
    ui_iload = mo.ui.slider(
        start=0.1, stop=5.0, step=0.1, value=1.0,
        label="Load current per PE (A)",
    )
    ui_mismatch = mo.ui.slider(
        start=1, stop=20, step=1, value=10,
        label="Current mismatch (± %)",
    )
    mo.vstack([
        mo.md("### Application"),
        mo.hstack([ui_vin_max, ui_vin_min], justify="start"),
        mo.hstack([ui_n_stages, ui_vpe], justify="start"),
        mo.hstack([ui_iload, ui_mismatch], justify="start"),
    ])
    return ui_iload, ui_mismatch, ui_n_stages, ui_vin_max, ui_vin_min, ui_vpe


@app.cell
def _(device_sliders, mo):
    dev_ui = device_sliders(mo)
    mo.vstack([
        mo.md("### Device Technology"),
        mo.hstack([dev_ui["lg"], dev_ui["mobility_pct"]], justify="start"),
        mo.hstack([dev_ui["vgs"], dev_ui["vth"]], justify="start"),
        dev_ui["ron_sp"],
    ])
    return (dev_ui,)


@app.cell
def _(mo):
    ui_fsw = mo.ui.slider(
        start=10, stop=500, step=10, value=100,
        label="Switching frequency (MHz)",
    )
    ui_cfly = mo.ui.slider(
        start=1, stop=100, step=1, value=10,
        label="Flying capacitor (nF)",
    )
    ui_sw_area = mo.ui.slider(
        start=0.01, stop=1.0, step=0.01, value=0.1,
        label="Total switch die area (mm²)",
    )
    ui_eta_target = mo.ui.slider(
        start=70, stop=99, step=1, value=85,
        label="Efficiency target (%)",
    )
    ui_lr_target = mo.ui.slider(
        start=1, stop=15, step=1, value=5,
        label="Load regulation target (%)",
    )
    mo.vstack([
        mo.md("### Converter Design"),
        mo.hstack([ui_fsw, ui_cfly], justify="start"),
        ui_sw_area,
        mo.md("### Pass/Fail Targets"),
        mo.hstack([ui_eta_target, ui_lr_target], justify="start"),
    ])
    return ui_cfly, ui_eta_target, ui_fsw, ui_lr_target, ui_sw_area


@app.cell
def _(dev_ui, device_from_sliders, device_summary_md, mo):
    dev = device_from_sliders(dev_ui)
    mo.md(f"### Device Model\n\n{device_summary_md(dev)}")
    return (dev,)


@app.cell
def _(gate_drive_loss, np):
    # Gear definitions: (name, ratio k where Vout_ideal = k * Vin,
    #                     n_switches, uses_flying_cap)
    GEARS = [
        ("1:1", 1.0, 2, False),
        ("2:1", 0.5, 4, True),
    ]

    def sc_analysis(dev, Vin, Vstring, Iload, fsw, Cfly, A_sw):
        """Analyze all gears at a single Vin. Returns dict of gear results."""
        results = {}
        Pload = Vstring * Iload

        for name, k, n_sw, uses_cap in GEARS:
            Vout_ideal = k * Vin
            if Vout_ideal < Vstring:
                continue  # can't reach target in this gear

            eta_ratio = Vstring / Vout_ideal

            # Switch Ron: divide total switch area equally among switches
            A_per_sw = A_sw / n_sw
            Ron_sw = dev.Ron_sp / A_per_sw

            # Output impedance
            if uses_cap:
                R_SSL = 1.0 / (Cfly * fsw)
                R_FSL = n_sw * Ron_sw / 2   # half conduct at once
                R_out = np.sqrt(R_SSL**2 + R_FSL**2)
            else:
                # Bypass: 2 series switches
                R_out = n_sw * Ron_sw

            P_cond = Iload**2 * R_out
            P_gate = gate_drive_loss(dev, A_sw, fsw)
            eta_total = Pload / (Pload + P_cond + P_gate) * eta_ratio

            results[name] = {
                "k": k,
                "Vout_ideal": Vout_ideal,
                "eta_ratio": eta_ratio,
                "n_sw": n_sw,
                "Ron_sw": Ron_sw,
                "R_out": R_out,
                "P_cond": P_cond,
                "P_gate": P_gate,
                "eta_total": eta_total,
                "load_reg": Iload * R_out / Vstring,
            }

        return results

    return GEARS, sc_analysis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gear Map

    Efficiency vs. input voltage for each gear. The **bold envelope** is the
    best gear at each voltage. Dashed line = efficiency target.

    Efficiency includes **ratio loss** (inherent to the gear) and **resistive
    loss** (switch Ron, capacitor charge-sharing). Gate drive loss is added
    on top.
    """)
    return


@app.cell
def _(
    GEARS,
    dev,
    np,
    plt,
    sc_analysis,
    ui_cfly,
    ui_eta_target,
    ui_fsw,
    ui_iload,
    ui_n_stages,
    ui_sw_area,
    ui_vin_max,
    ui_vin_min,
    ui_vpe,
):
    Vstring = ui_n_stages.value * ui_vpe.value
    Iload = ui_iload.value
    fsw = ui_fsw.value * 1e6
    Cfly = ui_cfly.value * 1e-9
    A_sw = ui_sw_area.value * 1e-6  # m²

    _vmin = max(ui_vin_min.value, Vstring + 0.01)
    _vmax = ui_vin_max.value
    if _vmin >= _vmax:
        _vmax = _vmin + 0.1
    Vin_sweep = np.linspace(_vmin, _vmax, 200)

    # Per-gear efficiency arrays
    gear_etas = {name: np.full_like(Vin_sweep, np.nan) for name, *_ in GEARS}
    eta_best = np.zeros_like(Vin_sweep)

    for i, Vin in enumerate(Vin_sweep):
        res = sc_analysis(dev, Vin, Vstring, Iload, fsw, Cfly, A_sw)
        for name in res:
            gear_etas[name][i] = res[name]["eta_total"] * 100
        if res:
            best = max(res, key=lambda g: res[g]["eta_total"])
            eta_best[i] = res[best]["eta_total"] * 100

    _colors = {"1:1": "#4e79a7", "2:1": "#e15759"}
    fig_gear, ax_gear = plt.subplots(figsize=(9, 5))

    for _gname, *_ in GEARS:
        _valid = ~np.isnan(gear_etas[_gname])
        if _valid.any():
            ax_gear.plot(
                Vin_sweep[_valid], gear_etas[_gname][_valid],
                color=_colors.get(_gname, "gray"), alpha=0.4, linewidth=1,
                label=f"{_gname} gear",
            )

    ax_gear.plot(Vin_sweep, eta_best, "k-", linewidth=2.5, label="Best gear")
    ax_gear.axhline(
        ui_eta_target.value, color="r", linestyle="--", linewidth=1,
        label=f"Target: {ui_eta_target.value}%",
    )
    ax_gear.set_xlabel("Input voltage (V)")
    ax_gear.set_ylabel("Efficiency (%)")
    ax_gear.set_ylim(40, 100)
    ax_gear.legend(loc="lower right")
    ax_gear.grid(True, alpha=0.3)
    ax_gear.set_title(
        f"String supply: {ui_n_stages.value}×{ui_vpe.value:.2f}V "
        f"= {Vstring:.2f}V, {Iload:.1f}A"
    )
    fig_gear
    return A_sw, Cfly, Iload, Vstring, fsw


@app.cell
def _(mo, ui_vin_max, ui_vin_min):
    ui_vin_point = mo.ui.slider(
        start=ui_vin_min.value,
        stop=ui_vin_max.value,
        step=0.01,
        value=round((ui_vin_min.value + ui_vin_max.value) / 2, 2),
        label="Operating point Vin (V)",
    )
    mo.vstack([mo.md("## Design Point Details"), ui_vin_point])
    return (ui_vin_point,)


@app.cell
def _(A_sw, Cfly, Iload, Vstring, dev, fsw, mo, sc_analysis, ui_vin_point):
    Vin_pt = ui_vin_point.value
    results = sc_analysis(dev, Vin_pt, Vstring, Iload, fsw, Cfly, A_sw)

    if not results:
        mo.md(
            f"**No gear can reach {Vstring:.2f}V from Vin = {Vin_pt:.2f}V.** "
            "Increase Vin or reduce string voltage."
        )
    else:
        best_name = max(results, key=lambda g: results[g]["eta_total"])
        b = results[best_name]

        mo.md(f"""
    ### At Vin = {Vin_pt:.2f} V — best gear: **{best_name}**

    | | Value |
    |---|---|
    | Ideal gear output | {b['Vout_ideal']:.3f} V |
    | Ratio efficiency | {b['eta_ratio']*100:.1f}% |
    | | |
    | Switches | {b['n_sw']} × {b['Ron_sw']*1e3:.1f} mΩ |
    | **R_out** | **{b['R_out']*1e3:.1f} mΩ** |
    | | |
    | Conduction loss | {b['P_cond']*1e3:.1f} mW |
    | Gate drive loss | {b['P_gate']*1e3:.1f} mW |
    | Ratio loss | {Vstring * Iload * (1/b['eta_ratio'] - 1)*1e3:.1f} mW |
    | | |
    | **Efficiency** | **{b['eta_total']*100:.1f}%** |
    | **Load regulation** | **{b['load_reg']*100:.1f}%** |
    """)
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Loss Breakdown at Operating Point
    """)
    return


@app.cell
def _(Iload, Vstring, plt, results):
    if results:
        _best = max(results, key=lambda g: results[g]["eta_total"])
        _b = results[_best]
        _Pload = Vstring * Iload
        _P_ratio = _Pload * (1 / _b["eta_ratio"] - 1)

        _losses = [_P_ratio * 1e3, _b["P_cond"] * 1e3, _b["P_gate"] * 1e3]
        _labels = [
            f"Ratio\n{_losses[0]:.1f} mW",
            f"Conduction\n{_losses[1]:.1f} mW",
            f"Gate drive\n{_losses[2]:.1f} mW",
        ]
        _colors = ["#59a14f", "#4e79a7", "#f28e2b"]

        _fig_pie, _ax_pie = plt.subplots(figsize=(4, 4))
        _ax_pie.pie(
            _losses, labels=_labels, colors=_colors,
            autopct="%.0f%%", startangle=90,
        )
        _ax_pie.set_title(f"Loss breakdown ({_best} gear)")
        _fig_pie
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mid-Point Balancer

    The balancer keeps each PE at its target voltage by handling the
    **mismatch current** (±10% of load).

    We reuse the same flying-capacitor topology, with additional switches
    to connect Cfly across either segment of the string.

    ### Two operating modes

    **Normal mode** (Vmid too low — top PE draws less):
    Standard 2:1 from battery into the bottom segment, same circuit as
    the string supply but scaled for the mismatch current only.

    **Pump mode** (Vmid too high — bottom PE draws less):
    - Phase 1: connect Cfly across bottom segment (Vmid → GND) — charges
      to Vmid
    - Phase 2: connect Cfly across top segment (Vstring → Vmid) — dumps
      charge, raising V_top

    The top segment briefly goes over-voltage, but the string supply has
    the bandwidth to compensate by backing off. Net effect: charge moves
    from bottom to top, Vmid drops back toward target.

    > *Note: a "top → bottom" pump mode (reverse of pump mode) is also
    > possible with the same switch fabric — worth exploring for
    > symmetry, but not needed for the first product.*

    ### Cap sizing concern

    In pump mode, the voltage perturbation on the top segment when Cfly
    dumps its charge is:

    $$\Delta V_{top} = \frac{C_{fly}}{C_{fly} + C_{decoupling}} \cdot (V_{mid} - V_{pe,target})$$

    If C_decoupling (off-chip) is small relative to Cfly, the perturbation
    is large — which either forces a small Cfly (limiting charge transfer
    per cycle, requiring higher pump frequency) or a large decoupling cap.
    """)
    return


@app.cell
def _(dev, np, mo, ui_iload, ui_mismatch, ui_sw_area, ui_vpe, ui_fsw, ui_cfly):
    Imis = ui_iload.value * (ui_mismatch.value / 100)
    Vpe = ui_vpe.value
    Pload_total = Vpe * ui_iload.value * 2  # both PEs

    # Charge transfer per cycle: Q = Cfly × ΔV
    # In pump mode, Cfly charges to Vmid ≈ Vpe + δ, then dumps to top segment.
    # Charge transferred per cycle ≈ Cfly × δ where δ is the overvoltage.
    # For steady state: Imis = Cfly × δ × fsw → δ = Imis / (Cfly × fsw)
    _Cfly = ui_cfly.value * 1e-9  # F
    _fsw = ui_fsw.value * 1e6     # Hz
    _delta_v = Imis / (_Cfly * _fsw) if _Cfly * _fsw > 0 else float('inf')

    # Balancer switch sizing (same Ron budget as before)
    _V_error_budget = 0.05 * Vpe
    _Ron_bal_max = _V_error_budget / Imis if Imis > 0 else 1e6
    _A_bal_min = dev.Ron_sp / _Ron_bal_max
    _A_bal_um2 = _A_bal_min * 1e12
    _sw_area_mm2 = ui_sw_area.value

    # SC balancer efficiency: ratio loss is small (pumping within Vpe range)
    # Main loss is conduction: I² × Ron × duty
    # For 4 balancer switches sharing the balancer area budget
    _n_bal_sw = 4
    _A_per_bal_sw = _A_bal_min  # minimum area per switch
    _Ron_bal_sw = dev.Ron_sp / _A_per_bal_sw if _A_per_bal_sw > 0 else float('inf')
    _P_bal_cond = Imis**2 * _Ron_bal_sw * 2  # 2 switches in path at a time

    mo.md(f"""
    ### Balancer Sizing (SC pump)

    | | Value |
    |---|---|
    | Mismatch current | ±{Imis:.2f} A ({ui_mismatch.value}% of {ui_iload.value:.1f} A) |
    | PE voltage | {Vpe:.2f} V |
    | | |
    | **Pump mode ΔV on top segment** | **{_delta_v*1e3:.1f} mV** |
    | (using string supply's Cfly = {ui_cfly.value} nF @ {ui_fsw.value} MHz) | |
    | | |
    | Balancer switches | {_n_bal_sw} × {_Ron_bal_sw*1e3:.1f} mΩ |
    | Min area per switch | {_A_bal_um2:.0f} µm² |
    | Conduction loss | {_P_bal_cond*1e3:.1f} mW |
    | As % of total load | {_P_bal_cond / Pload_total * 100:.2f}% |
    | | |
    | String supply switch area (comparison) | {_sw_area_mm2:.2f} mm² |
    """)
    return


# ===================================================================
# Switch waveforms
# ===================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Switch Waveforms (2:1 gear)

    Idealized Vds and Ids for each switch through two complete cycles.
    These show what the transistors actually see — the key information
    for device design.

    Refer to the flying-capacitor diagram above:
    - **Phase 1 (series)**: SW1, SW2 on; SW3, SW4 off
    - **Phase 2 (parallel)**: SW3, SW4 on; SW1, SW2 off
    """)
    return


@app.cell
def _(
    dev, np, plt,
    Vstring, Iload, fsw, Cfly, A_sw,
    ui_vin_point,
):
    _Vin = ui_vin_point.value
    _Vout = Vstring
    _T = 1.0 / fsw          # period (s)
    _dt = _T / 200           # time step
    _t = np.arange(0, 3 * _T, _dt)  # 3 cycles
    _n_sw = 4
    _Ron = dev.Ron_sp / (A_sw / _n_sw)

    # Cfly voltage: in steady state, Vcfly ≈ Vin - Vout for 2:1
    _Vcfly = _Vin - _Vout

    # Phase signal: 0 = phase1 (series, SW1+SW2 on), 1 = phase2 (parallel, SW3+SW4 on)
    _phase = ((_t % _T) >= _T / 2).astype(float)
    _ph1 = _phase == 0  # series phase
    _ph2 = _phase == 1  # parallel phase

    # --- SW1: between Vin and Cfly+ ---
    # On in phase 1: Vds ≈ Iload × Ron, Ids = Iload
    # Off in phase 2: Vds ≈ Vin - Vout (blocks Vcfly), Ids = 0
    _sw1_vds = np.where(_ph1, Iload * _Ron, _Vcfly)
    _sw1_ids = np.where(_ph1, Iload, 0)

    # --- SW2: between Cfly- and GND ---
    # On in phase 1: Vds ≈ Iload × Ron, Ids = Iload
    # Off in phase 2: Vds ≈ Vout (blocks output voltage), Ids = 0
    _sw2_vds = np.where(_ph1, Iload * _Ron, _Vout)
    _sw2_ids = np.where(_ph1, Iload, 0)

    # --- SW3: between Cfly+ and Vout ---
    # Off in phase 1: Vds ≈ Vin - Vout (blocks Vcfly), Ids = 0
    # On in phase 2: Vds ≈ Iload × Ron, Ids = Iload
    _sw3_vds = np.where(_ph2, Iload * _Ron, _Vcfly)
    _sw3_ids = np.where(_ph2, Iload, 0)

    # --- SW4: between Cfly- and Vout ---
    # Off in phase 1: Vds ≈ Vout, Ids = 0
    # On in phase 2: Vds ≈ Iload × Ron, Ids = Iload
    _sw4_vds = np.where(_ph2, Iload * _Ron, _Vout)
    _sw4_ids = np.where(_ph2, Iload, 0)

    _switches = [
        ("SW1\n(Vin → Cfly+)", _sw1_vds, _sw1_ids),
        ("SW2\n(Cfly− → GND)", _sw2_vds, _sw2_ids),
        ("SW3\n(Cfly+ → Vout)", _sw3_vds, _sw3_ids),
        ("SW4\n(Vout → Cfly−)", _sw4_vds, _sw4_ids),
    ]

    _fig_sw, _axes = plt.subplots(4, 2, figsize=(10, 8), sharex=True)
    _t_us = _t * 1e6  # convert to µs

    for _i, (_name, _vds, _ids) in enumerate(_switches):
        _axes[_i, 0].plot(_t_us, _vds * 1e3, color="#e15759", linewidth=1.5)
        _axes[_i, 0].set_ylabel("Vds (mV)")
        _axes[_i, 0].set_title(_name, fontsize=9, loc="left")
        _axes[_i, 0].grid(True, alpha=0.3)

        _axes[_i, 1].plot(_t_us, _ids * 1e3, color="#4e79a7", linewidth=1.5)
        _axes[_i, 1].set_ylabel("Ids (mA)")
        _axes[_i, 1].grid(True, alpha=0.3)

    _axes[-1, 0].set_xlabel("Time (µs)")
    _axes[-1, 1].set_xlabel("Time (µs)")
    _fig_sw.suptitle(
        f"Switch waveforms: 2:1 gear, Vin={_Vin:.2f}V, "
        f"Vout={_Vout:.2f}V, {Iload:.1f}A, {fsw/1e6:.0f}MHz",
        fontsize=10,
    )
    _fig_sw.tight_layout()
    _fig_sw

    return


@app.cell(hide_code=True)
def _(dev, mo, A_sw, Iload, Vstring, ui_vin_point):
    _Vin = ui_vin_point.value
    _Vout = Vstring
    _Vcfly = _Vin - _Vout
    _n_sw = 4
    _Ron = dev.Ron_sp / (A_sw / _n_sw)

    mo.md(f"""
    ### Switch Stress Summary

    | Switch | Blocks (off) | Carries (on) | On-state Vds |
    |--------|-------------|-------------|-------------|
    | SW1 (Vin → Cfly+) | {_Vcfly*1e3:.0f} mV | {Iload*1e3:.0f} mA | {Iload*_Ron*1e3:.1f} mV |
    | SW2 (Cfly− → GND) | {_Vout*1e3:.0f} mV | {Iload*1e3:.0f} mA | {Iload*_Ron*1e3:.1f} mV |
    | SW3 (Cfly+ → Vout) | {_Vcfly*1e3:.0f} mV | {Iload*1e3:.0f} mA | {Iload*_Ron*1e3:.1f} mV |
    | SW4 (Vout → Cfly−) | {_Vout*1e3:.0f} mV | {Iload*1e3:.0f} mA | {Iload*_Ron*1e3:.1f} mV |

    All off-state voltages are well below 1 V — no BV concern. The on-state
    Vds ({Iload*_Ron*1e3:.1f} mV) is the Ron × Iload drop across each switch.

    **Note**: SW1 and SW3 have their source above ground — they need
    **bootstrap gate drive** (gate must be > source + Vth to turn on).
    SW2 and SW4 are ground-referenced and can use direct gate drive.
    """)
    return


@app.cell(hide_code=True)
def _(mo, results, ui_eta_target, ui_lr_target, ui_vin_point):
    if not results:
        mo.md("**Cannot evaluate** — no gear reaches target at this Vin.")
    else:
        _best = max(results, key=lambda g: results[g]["eta_total"])
        _b = results[_best]

        _eta_ok = _b["eta_total"] * 100 >= ui_eta_target.value
        _lr_ok = _b["load_reg"] * 100 <= ui_lr_target.value

        def _badge(ok):
            return "PASS" if ok else "**FAIL**"

        mo.md(f"""
    ## Pass / Fail (at Vin = {ui_vin_point.value:.2f} V, gear {_best})

    | Criterion | Target | Actual | Result |
    |-----------|--------|--------|--------|
    | Efficiency | ≥ {ui_eta_target.value}% | {_b['eta_total']*100:.1f}% | {_badge(_eta_ok)} |
    | Load regulation | ≤ {ui_lr_target.value}% | {_b['load_reg']*100:.1f}% | {_badge(_lr_ok)} |

    *Check at both Vin min ({ui_vin_point.start:.2f}V) and max ({ui_vin_point.stop:.2f}V)
    for full-range validation.*
    """)
    return


if __name__ == "__main__":
    app.run()
