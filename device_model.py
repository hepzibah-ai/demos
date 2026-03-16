"""
Shared device model for BEOL transistor point-of-load power supply analysis.

Silicon limits and channel-dominated MOSFET model for low-voltage (≤3V)
power switches. Defaults match the ASU PTM 90nm BSIM4 model used for
SPICE waveforms; adjust sliders for your own process.
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EPS_SI = 11.7 * 8.854e-12       # Si permittivity (F/m)
EPS_OX = 3.9 * 8.854e-12        # SiO2 permittivity (F/m)
MU_N_BULK = 480e-4              # Bulk Si electron mobility (m²/V·s)
V_SAT = 1.0e5                   # Saturation velocity (m/s)
E_C_SI = 3e5 * 1e2              # Critical field Si (V/m)


# ---------------------------------------------------------------------------
# Device model
# ---------------------------------------------------------------------------
@dataclass
class DeviceModel:
    """Computed device characteristics."""
    # User inputs
    Lg: float          # Gate length (m)
    Vgs: float         # Gate voltage (V)
    Ron_sp: float      # Actual specific on-resistance (Ω·m²), user-specified
    # Derived
    tox: float         # Oxide thickness (m)
    Cox: float         # Oxide capacitance (F/m²)
    mu_n: float        # Effective mobility (m²/V·s)
    Vov: float         # Overdrive voltage (V)
    Ron_sp_ideal: float  # Ideal channel Ron,sp per gate area (Ω·m²)
    overhead: float    # Ron_sp / Ron_sp_ideal
    Qg_density: float  # Gate charge density (C/m²)
    fT: float          # Intrinsic device fT (Hz)
    fT_johnson_3V: float  # Johnson limit fT at 3V (Hz)


def compute_device(
    Lg_nm: float = 90,
    mobility_pct: float = 100,
    Vgs: float = 0.9,
    Vth: float = 0.40,
    Ron_sp_mohm_mm2: float = 0.03,
    tox_nm: float = 2.05,
) -> DeviceModel:
    """Compute device characteristics.

    Defaults match the ASU PTM 90nm BSIM4 model used for SPICE waveforms.
    Ron_sp is specified directly in mΩ·mm² (the user/partner knows their
    device better than any formula). The ideal channel Ron,sp is computed
    as a reference.
    """
    Lg = Lg_nm * 1e-9
    mu_n = MU_N_BULK * (mobility_pct / 100)
    Vov = max(Vgs - Vth, 0.01)

    tox = tox_nm * 1e-9
    Cox = EPS_OX / tox

    # Ideal channel Ron,sp per unit GATE area (Ω·m²)
    # Rch = Lg / (µn Cox Vov W),  gate area = W × Lg
    # Ron,sp_gate = Rch × A_gate = Lg² / (µn Cox Vov)
    Ron_sp_ideal = Lg**2 / (mu_n * Cox * Vov)

    # User-specified actual Ron,sp: convert mΩ·mm² → Ω·m²
    Ron_sp = Ron_sp_mohm_mm2 * 1e-3 * 1e-6  # mΩ·mm² → Ω·m²

    overhead = Ron_sp / Ron_sp_ideal if Ron_sp_ideal > 0 else 0

    # Gate charge density
    Qg_density = Cox * Vgs

    # Intrinsic fT = mu_n * Vov / (2 pi Lg²)
    fT = mu_n * Vov / (2 * np.pi * Lg**2)

    # Johnson limit at 3V
    fT_johnson = (E_C_SI * V_SAT) / (2 * np.pi * 3.0)

    return DeviceModel(
        Lg=Lg, Vgs=Vgs, Ron_sp=Ron_sp,
        tox=tox, Cox=Cox, mu_n=mu_n, Vov=Vov,
        Ron_sp_ideal=Ron_sp_ideal, overhead=overhead,
        Qg_density=Qg_density, fT=fT, fT_johnson_3V=fT_johnson,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_mohm_mm2(Ron_sp_si: float) -> float:
    """Convert Ron,sp from Ω·m² to mΩ·mm²."""
    return Ron_sp_si * 1e3 * 1e6


def device_summary_md(dev: DeviceModel) -> str:
    """Return a markdown table summarizing the device model."""
    rsp = to_mohm_mm2(dev.Ron_sp)
    rsp_ideal = to_mohm_mm2(dev.Ron_sp_ideal)
    return f"""\
| Parameter | Value | Notes |
|-----------|-------|-------|
| Gate length | {dev.Lg*1e9:.0f} nm | |
| Oxide thickness | {dev.tox*1e9:.2f} nm | |
| Cox | {dev.Cox/1e-3:.1f} fF/µm² | |
| Effective mobility | {dev.mu_n*1e4:.0f} cm²/V·s | |
| Overdrive (Vgs−Vth) | {dev.Vov:.2f} V | |
| | | |
| **Ron,sp (your device)** | **{rsp:.3f} mΩ·mm²** | User-specified |
| Ron,sp (ideal channel) | {rsp_ideal:.4f} mΩ·mm² | Lg²/(µn Cox Vov), gate area only |
| Overhead factor | {dev.overhead:.0f}× | Layout, contacts, metal, parasitics |
| | | |
| Qg density | {dev.Qg_density/1e-3:.1f} fC/µm² | Cox × Vgs |
| fT (device) | {dev.fT/1e9:.1f} GHz | Intrinsic |
| fT (Johnson @ 3V) | {dev.fT_johnson_3V/1e9:.0f} GHz | Material limit |"""


def gate_drive_loss(dev: DeviceModel, total_gate_area: float, fsw: float) -> float:
    """Gate drive power loss (W).

    P_gate = Qg_total * Vgs * fsw = Cox * Vgs² * A_gate * fsw
    """
    return dev.Cox * dev.Vgs**2 * total_gate_area * fsw


# ---------------------------------------------------------------------------
# Marimo UI: device sliders
# ---------------------------------------------------------------------------

def device_sliders(mo):
    """Create standard device parameter sliders. Returns dict of UI elements."""
    return {
        "lg": mo.ui.slider(
            start=50, stop=200, step=10, value=90,
            label="Gate length (nm)",
        ),
        "mobility_pct": mo.ui.slider(
            start=10, stop=120, step=5, value=100,
            label="Mobility (% of bulk Si, ~480 cm²/V·s)",
        ),
        "vgs": mo.ui.slider(
            start=0.5, stop=1.2, step=0.05, value=0.9,
            label="Gate drive voltage (V)",
        ),
        "vth": mo.ui.slider(
            start=0.15, stop=0.5, step=0.01, value=0.40,
            label="Threshold voltage (V)",
        ),
        "tox": mo.ui.slider(
            start=1.0, stop=5.0, step=0.05, value=2.05,
            label="Oxide thickness (nm)",
        ),
        "ron_sp": mo.ui.slider(
            start=0.01, stop=20.0, step=0.01, value=0.03,
            label="Ron,sp — your device (mΩ·mm²)",
        ),
    }


def device_from_sliders(dev_ui: dict) -> DeviceModel:
    """Build DeviceModel from slider dict."""
    return compute_device(
        Lg_nm=dev_ui["lg"].value,
        mobility_pct=dev_ui["mobility_pct"].value,
        Vgs=dev_ui["vgs"].value,
        Vth=dev_ui["vth"].value,
        Ron_sp_mohm_mm2=dev_ui["ron_sp"].value,
        tox_nm=dev_ui["tox"].value,
    )
