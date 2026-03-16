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
    from spice_runner import run_balancer_normal, run_balancer_pump
    return (
        Path,
        device_from_sliders,
        device_sliders,
        device_summary_md,
        gate_drive_loss,
        mo,
        np,
        plt,
        run_balancer_normal,
        run_balancer_pump,
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
    between the two string segments. Two operating modes:

    ### Normal mode (Vmid too low — bottom PE draws more)

    A 1:1 connection from the battery (or string supply output) into the
    bottom segment, delivering extra current to make up the shortfall.
    The battery voltage is always above Vpe, so no step-up is needed —
    the excess drops across switch resistance.

    - Phase 1: Cfly charges from Vin (through 2 switches)
    - Phase 2: Cfly discharges into bottom segment (through 2 switches)

    This is the simplest SC topology — just a pair of half-bridge switches
    acting as a 1:1 converter, sized for the mismatch current only.

    ### Pump mode (Vmid too high — top PE draws more)

    Moves charge from bottom segment to top segment:
    - Phase 1: Cfly charges across bottom segment (Vmid → GND)
    - Phase 2: Cfly dumps into top segment (Vstring → Vmid)

    The top segment briefly sees a voltage bump, but the string supply
    has the bandwidth to compensate. Net effect: charge moves bottom → top,
    Vmid drops back toward target.

    All switches are **NMOS** (the partner's process). High-side switches
    need a bootstrap or charge-pump gate drive.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reference: Flying Capacitor Converter

    The diagram shows both phases of a basic divide-by-2 flying capacitor
    converter. The balancer reuses this switch fabric — the same 4 switches
    can be reconfigured for normal mode (1:1 from battery) or pump mode
    (bottom → top charge shuttle).
    """)
    return


@app.cell(hide_code=True)
def _(mo, Path):
    _here = Path(__file__).resolve().parent
    mo.image(src=str(_here / "figures" / "flying_capacitor.png"))
    return


@app.cell
def _(mo):
    mo.md("## Parameters")
    return


@app.cell
def _(mo):
    ui_vin = mo.ui.slider(
        start=0.5, stop=3.0, step=0.1, value=1.2,
        label="Battery voltage (V)",
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
        mo.hstack([ui_vin, ui_vpe], justify="start"),
        mo.hstack([ui_iload, ui_mismatch], justify="start"),
    ])
    return ui_iload, ui_mismatch, ui_vin, ui_vpe


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
    ui_w_sw = mo.ui.slider(
        start=1, stop=100, step=1, value=10,
        label="Switch width (µm)",
    )
    mo.vstack([
        mo.md("### Balancer Design"),
        mo.hstack([ui_fsw, ui_cfly], justify="start"),
        ui_w_sw,
    ])
    return ui_cfly, ui_fsw, ui_w_sw


@app.cell
def _(dev_ui, device_from_sliders, device_summary_md, mo):
    dev = device_from_sliders(dev_ui)
    mo.md(f"### Device Model\n\n{device_summary_md(dev)}")
    return (dev,)


# ===================================================================
# Balancer analysis (analytical)
# ===================================================================
@app.cell
def _(gate_drive_loss):

    def balancer_analysis(dev, Vin, Vpe, Imis, fsw, Cfly, W_sw, L_sw):
        """Analyze both balancer modes.

        W_sw: switch width (m), L_sw: gate length (m).
        Switch area per device = W × L. Total area depends on mode.
        """
        results = {}

        # --- Normal mode (1:1, battery to bottom segment) ---
        _n_sw = 2
        _A_sw_total = _n_sw * W_sw * L_sw  # total gate area
        _Ron_sw = dev.Ron_sp / (W_sw * L_sw)
        _R_out = _n_sw * _Ron_sw

        _eta_ratio = Vpe / Vin if Vin > 0 else 0
        _P_cond = Imis**2 * _R_out
        _P_gate = gate_drive_loss(dev, _A_sw_total, fsw)
        _Pload = Vpe * Imis
        _eta = _Pload / (_Pload + _P_cond + _P_gate) * _eta_ratio if _Pload > 0 else 0

        results["normal"] = {
            "n_sw": _n_sw,
            "Ron_sw": _Ron_sw,
            "R_out": _R_out,
            "A_sw_total": _A_sw_total,
            "eta_ratio": _eta_ratio,
            "P_cond": _P_cond,
            "P_gate": _P_gate,
            "eta_total": _eta,
            "Vdrop_sw": Imis * _R_out,
        }

        # --- Pump mode (bottom → top charge shuttle) ---
        _n_sw_p = 4
        _A_sw_total_p = _n_sw_p * W_sw * L_sw
        _Ron_sw_p = dev.Ron_sp / (W_sw * L_sw)
        _R_path = 2 * _Ron_sw_p  # 2 switches in path at a time

        _Q_per_cycle = Cfly * Vpe
        _I_max_pump = _Q_per_cycle * fsw
        _delta_v = Imis / (Cfly * fsw) if Cfly * fsw > 0 else float('inf')

        _P_cond_p = Imis**2 * _R_path
        _P_gate_p = gate_drive_loss(dev, _A_sw_total_p, fsw)
        _eta_p = _Pload / (_Pload + _P_cond_p + _P_gate_p) if _Pload > 0 else 0

        results["pump"] = {
            "n_sw": _n_sw_p,
            "Ron_sw": _Ron_sw_p,
            "R_out": _R_path,
            "A_sw_total": _A_sw_total_p,
            "eta_ratio": 1.0,
            "P_cond": _P_cond_p,
            "P_gate": _P_gate_p,
            "eta_total": _eta_p,
            "Vdrop_sw": Imis * _R_path,
            "delta_v": _delta_v,
            "I_max_pump": _I_max_pump,
        }

        return results

    return (balancer_analysis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Balancer Analysis

    Both operating modes analyzed at the current slider settings.
    The balancer only carries the **mismatch current** — a fraction
    of the full PE load current.
    """)
    return


@app.cell
def _(
    balancer_analysis, dev, mo,
    ui_cfly, ui_fsw, ui_w_sw,
    ui_iload, ui_mismatch, ui_vin, ui_vpe,
):
    Vin = ui_vin.value
    Vpe = ui_vpe.value
    Imis = ui_iload.value * (ui_mismatch.value / 100)
    fsw = ui_fsw.value * 1e6
    Cfly = ui_cfly.value * 1e-9
    W_sw = ui_w_sw.value * 1e-6  # µm → m
    L_sw = dev.Lg  # gate length from device model
    Pload_total = Vpe * ui_iload.value * 2  # both PEs

    bal = balancer_analysis(dev, Vin, Vpe, Imis, fsw, Cfly, W_sw, L_sw)
    _n = bal["normal"]
    _p = bal["pump"]

    mo.md(f"""
    ### Operating Point

    | | Normal mode | Pump mode |
    |---|---|---|
    | **When** | Vmid too low | Vmid too high |
    | **Action** | Battery → bottom | Bottom → top |
    | Mismatch current | ±{Imis*1e3:.0f} mA | ±{Imis*1e3:.0f} mA |
    | Switches | {_n['n_sw']} × {ui_w_sw.value} µm | {_p['n_sw']} × {ui_w_sw.value} µm (2 active) |
    | Ron per switch | {_n['Ron_sw']*1e3:.1f} mΩ | {_p['Ron_sw']*1e3:.1f} mΩ |
    | Switch Vdrop | {_n['Vdrop_sw']*1e3:.1f} mV | {_p['Vdrop_sw']*1e3:.1f} mV |
    | Total switch area | {_n['A_sw_total']*1e12:.0f} µm² | {_p['A_sw_total']*1e12:.0f} µm² |
    | | | |
    | Ratio efficiency | {_n['eta_ratio']*100:.0f}% (Vpe/Vin) | 100% (no ratio loss) |
    | Conduction loss | {_n['P_cond']*1e3:.2f} mW | {_p['P_cond']*1e3:.2f} mW |
    | Gate drive loss | {_n['P_gate']*1e3:.2f} mW | {_p['P_gate']*1e3:.2f} mW |
    | **Efficiency** | **{_n['eta_total']*100:.1f}%** | **{_p['eta_total']*100:.1f}%** |
    | | | |
    | ΔV perturbation (top) | — | {_p['delta_v']*1e3:.1f} mV |
    | Max pump current | — | {_p['I_max_pump']*1e3:.0f} mA |

    **System impact**: balancer loss at worst-case mismatch =
    {max(_n['P_cond'] + _n['P_gate'], _p['P_cond'] + _p['P_gate'])*1e3:.1f} mW
    = **{max(_n['P_cond'] + _n['P_gate'], _p['P_cond'] + _p['P_gate']) / Pload_total * 100:.2f}%**
    of total load power ({Pload_total*1e3:.0f} mW).

    > *Normal mode ratio efficiency is low ({_n['eta_ratio']*100:.0f}%) because
    > the battery ({Vin:.1f} V) is much higher than Vpe ({Vpe:.2f} V).
    > This is acceptable — the wasted power is only
    > {Imis * (Vin - Vpe) * 1e3:.1f} mW (mismatch current × excess voltage).*
    """)
    return Cfly, Imis, L_sw, Pload_total, Vin, Vpe, W_sw, bal, fsw


@app.cell(hide_code=True)
def _(mo):
    mo.md("### Loss Breakdown by Mode")
    return


@app.cell
def _(Imis, Vin, Vpe, bal, plt):
    _fig, (_ax_n, _ax_p) = plt.subplots(1, 2, figsize=(8, 3.5))

    _n = bal["normal"]
    _p = bal["pump"]
    _P_ratio_n = Imis * (Vin - Vpe)

    # Normal mode
    _losses_n = [_P_ratio_n * 1e3, _n["P_cond"] * 1e3, _n["P_gate"] * 1e3]
    _labels_n = [
        f"Ratio\n{_losses_n[0]:.1f} mW",
        f"Ron\n{_losses_n[1]:.2f} mW",
        f"Gate\n{_losses_n[2]:.2f} mW",
    ]
    _colors = ["#59a14f", "#4e79a7", "#f28e2b"]
    _ax_n.pie(_losses_n, labels=_labels_n, colors=_colors,
              autopct="%.0f%%", startangle=90)
    _ax_n.set_title(f"Normal mode\n(η = {_n['eta_total']*100:.0f}%)", fontsize=10)

    # Pump mode
    _losses_p = [_p["P_cond"] * 1e3, _p["P_gate"] * 1e3]
    _labels_p = [
        f"Ron\n{_losses_p[0]:.2f} mW",
        f"Gate\n{_losses_p[1]:.2f} mW",
    ]
    _ax_p.pie(_losses_p, labels=_labels_p, colors=_colors[1:],
              autopct="%.0f%%", startangle=90)
    _ax_p.set_title(f"Pump mode\n(η = {_p['eta_total']*100:.0f}%)", fontsize=10)

    _fig.suptitle(f"Balancer losses at ±{Imis*1e3:.0f} mA mismatch", fontsize=11)
    _fig.tight_layout()
    _fig
    return


# ===================================================================
# SPICE switch waveforms — normal mode
# ===================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Switch Waveforms (SPICE)

    Transient simulation using the
    [ASU PTM 90nm](https://mec.umn.edu/ptm) BSIM4 NMOS model.
    These are real transistor waveforms — not idealized square waves.

    Adjust the **switch width** and **frequency** sliders above to see
    how the device behaves under different sizing.

    ### Normal mode (1:1 from battery)

    Four switches form a full bridge. Phase 1 (S1, S3 on) charges Cfly
    from the battery; phase 2 (S2, S4 on) discharges into the bottom
    PE segment.
    """)
    return


@app.cell
def _(Cfly, Vin, Vpe, dev_ui, np, plt, run_balancer_normal, ui_fsw, ui_w_sw):
    _wf_n = run_balancer_normal(
        Vbat=Vin,
        Vpe=Vpe,
        Vgs=dev_ui["vgs"].value,
        W_um=ui_w_sw.value,
        L_nm=dev_ui["lg"].value,
        Cfly_nF=Cfly * 1e9,
        fsw_MHz=ui_fsw.value,
    )
    _t = _wf_n["t_ns"]

    _fig, _axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)

    # Cfly voltage
    _axes[0].plot(_t, _wf_n["v_cfly"] * 1e3, color="#59a14f", linewidth=1.5)
    _axes[0].set_ylabel("Vcfly (mV)")
    _axes[0].set_title("Flying capacitor voltage", fontsize=9, loc="left")
    _axes[0].grid(True, alpha=0.3)

    # Node voltages
    _axes[1].plot(_t, _wf_n["v_a"] * 1e3, label="Cfly+", linewidth=1.5)
    _axes[1].plot(_t, _wf_n["v_b"] * 1e3, label="Cfly−", linewidth=1.5)
    _axes[1].set_ylabel("Node (mV)")
    _axes[1].set_title("Cfly node voltages", fontsize=9, loc="left")
    _axes[1].legend(fontsize=8)
    _axes[1].grid(True, alpha=0.3)

    # Currents
    _axes[2].plot(_t, _wf_n["i_bat_mA"], color="#e15759", label="Battery",
                  linewidth=1.5)
    _axes[2].plot(_t, _wf_n["i_load_mA"], color="#4e79a7", label="Load",
                  linewidth=1.5)
    _axes[2].set_ylabel("Current (mA)")
    _axes[2].set_xlabel("Time (ns)")
    _axes[2].set_title("Supply currents", fontsize=9, loc="left")
    _axes[2].legend(fontsize=8)
    _axes[2].grid(True, alpha=0.3)

    # Compute average currents from steady-state portion
    _mid = len(_t) // 2
    _i_bat_avg = np.mean(_wf_n["i_bat_mA"][_mid:])
    _i_load_avg = np.mean(_wf_n["i_load_mA"][_mid:])

    _fig.suptitle(
        f"Normal mode SPICE: W={ui_w_sw.value}µm, L={dev_ui['lg'].value}nm, "
        f"Vin={Vin:.1f}V, {ui_fsw.value}MHz  |  "
        f"Ibat,avg={_i_bat_avg:.2f}mA, Iload,avg={_i_load_avg:.2f}mA",
        fontsize=9,
    )
    _fig.tight_layout()
    _fig
    return


# ===================================================================
# SPICE switch waveforms — pump mode
# ===================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pump mode (bottom → top)

    Phase 1 (S1, S2 on): Cfly charges across the bottom PE segment
    (Vmid → GND). Phase 2 (S3, S4 on): Cfly dumps charge into the
    top segment (Vstring → Vmid).
    """)
    return


@app.cell
def _(Cfly, Vpe, dev_ui, np, plt, run_balancer_pump, ui_fsw, ui_w_sw):
    _wf_p = run_balancer_pump(
        Vpe=Vpe,
        Vgs=dev_ui["vgs"].value,
        W_um=ui_w_sw.value,
        L_nm=dev_ui["lg"].value,
        Cfly_nF=Cfly * 1e9,
        fsw_MHz=ui_fsw.value,
    )
    _t = _wf_p["t_ns"]

    _fig, _axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)

    # Cfly voltage
    _axes[0].plot(_t, _wf_p["v_cfly"] * 1e3, color="#59a14f", linewidth=1.5)
    _axes[0].set_ylabel("Vcfly (mV)")
    _axes[0].set_title("Flying capacitor voltage", fontsize=9, loc="left")
    _axes[0].grid(True, alpha=0.3)

    # Node voltages
    _axes[1].plot(_t, _wf_p["v_a"] * 1e3, label="Cfly+", linewidth=1.5)
    _axes[1].plot(_t, _wf_p["v_b"] * 1e3, label="Cfly−", linewidth=1.5)
    _axes[1].set_ylabel("Node (mV)")
    _axes[1].set_title("Cfly node voltages", fontsize=9, loc="left")
    _axes[1].legend(fontsize=8)
    _axes[1].grid(True, alpha=0.3)

    # Currents
    _axes[2].plot(_t, _wf_p["i_top_mA"], color="#e15759", label="Top segment",
                  linewidth=1.5)
    _axes[2].plot(_t, _wf_p["i_bot_mA"], color="#4e79a7", label="Bottom segment",
                  linewidth=1.5)
    _axes[2].set_ylabel("Current (mA)")
    _axes[2].set_xlabel("Time (ns)")
    _axes[2].set_title("Segment currents", fontsize=9, loc="left")
    _axes[2].legend(fontsize=8)
    _axes[2].grid(True, alpha=0.3)

    _mid = len(_t) // 2
    _i_top_avg = np.mean(_wf_p["i_top_mA"][_mid:])
    _i_bot_avg = np.mean(_wf_p["i_bot_mA"][_mid:])

    _fig.suptitle(
        f"Pump mode SPICE: W={ui_w_sw.value}µm, L={dev_ui['lg'].value}nm, "
        f"Vpe={Vpe:.2f}V, {ui_fsw.value}MHz  |  "
        f"Itop,avg={_i_top_avg:.2f}mA, Ibot,avg={_i_bot_avg:.2f}mA",
        fontsize=9,
    )
    _fig.tight_layout()
    _fig
    return


# ===================================================================
# Switch stress summary
# ===================================================================
@app.cell(hide_code=True)
def _(Imis, Vin, Vpe, bal, mo, ui_w_sw):
    _n = bal["normal"]
    _p = bal["pump"]

    mo.md(f"""
    ### Switch Stress Summary

    | | Normal mode ({_n['n_sw']} × {ui_w_sw.value} µm) | Pump mode ({_p['n_sw']} × {ui_w_sw.value} µm) |
    |---|---|---|
    | Max off-state Vds | {(Vin - Vpe)*1e3:.0f} mV | {Vpe*1e3:.0f} mV |
    | On-state current | {Imis*1e3:.0f} mA | {Imis*1e3:.0f} mA |
    | On-state Vds | {_n['Vdrop_sw']*1e3:.1f} mV | {_p['Vdrop_sw']*1e3:.1f} mV |
    | Ron per switch | {_n['Ron_sw']*1e3:.1f} mΩ | {_p['Ron_sw']*1e3:.1f} mΩ |

    All voltages well below 1 V — no breakdown concern.
    Normal-mode S1 needs **bootstrap gate drive** (source above ground).
    Pump-mode S1, S3 need bootstrap drive.
    """)
    return


# ===================================================================
# Pass / Fail
# ===================================================================
@app.cell(hide_code=True)
def _(Imis, Pload_total, Vin, Vpe, bal, mo):
    _n = bal["normal"]
    _p = bal["pump"]
    _worst_loss = max(
        _n["P_cond"] + _n["P_gate"] + Imis * (Vin - Vpe),
        _p["P_cond"] + _p["P_gate"],
    )
    _sys_impact_pct = _worst_loss / Pload_total * 100

    _bal_eta_ok = min(_n["eta_total"], _p["eta_total"]) * 100 >= 50
    _sys_ok = _sys_impact_pct <= 5

    def _badge(ok):
        return "PASS" if ok else "**FAIL**"

    mo.md(f"""
    ## Pass / Fail

    | Criterion | Target | Actual | Result |
    |-----------|--------|--------|--------|
    | Balancer efficiency (worst mode) | ≥ 50% | {min(_n['eta_total'], _p['eta_total'])*100:.1f}% | {_badge(_bal_eta_ok)} |
    | System loss from balancer | ≤ 5% of load | {_sys_impact_pct:.2f}% | {_badge(_sys_ok)} |

    *The 50% balancer target is deliberately relaxed — at 10% mismatch
    current, even 50% efficiency only costs ~5% at the system level.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    *Switch waveforms powered by [ngspice](https://ngspice.sourceforge.io/)
    with [ASU Predictive Technology Models](https://mec.umn.edu/ptm) (90nm bulk BSIM4).
    The PTM model is a reasonable proxy for device behavior — the partner
    should substitute their own model for quantitative design.*
    """)
    return


if __name__ == "__main__":
    app.run()
