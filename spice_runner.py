"""
Run ngspice simulations from Python and return waveform data.

Generates parametrized SPICE decks, invokes ngspice in batch mode,
and parses the wrdata output into numpy arrays.
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path

SPICE_DIR = Path(__file__).resolve().parent / "spice"
PTM_MODEL = SPICE_DIR / "ptm_90nm_bulk.pm"


def _parse_wrdata(path: str, n_signals: int) -> tuple:
    """Parse ngspice wrdata output (interleaved time/value pairs).

    Returns (time_array, dict of index -> value_array).
    """
    data = np.loadtxt(path)
    t = data[:, 0]
    values = {i: data[:, 2 * i + 1] for i in range(n_signals)}
    return t, values


def _run_ngspice(deck: str, workdir: str) -> str:
    """Write deck to file and run ngspice -b. Returns stdout."""
    deck_path = Path(workdir) / "deck.sp"
    deck_path.write_text(deck)
    result = subprocess.run(
        ["ngspice", "-b", str(deck_path)],
        capture_output=True, text=True, timeout=60,
        cwd=workdir,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ngspice failed (rc={result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def run_balancer(
    Vbat: float = 1.2,
    Vpe: float = 0.4,
    Vgs: float = 0.9,
    W_um: float = 300.0,
    L_nm: float = 90,
    Cfly_nF: float = 1.0,
    Cpe_nF: float = 0.5,
    Istring_mA: float = 100.0,
    Imismatch_mA: float = 10.0,
    fsw_MHz: float = 100.0,
    n_cycles: int = 60,
) -> dict:
    """Simulate 1:1 SC balancer with physical PE loads.

    Each PE is modeled as a capacitor (Cpe, representing ~0.5mm² of
    well capacitance in 28nm at Vpe) with initial voltage Vpe, plus a
    parallel current source representing PE power consumption.

    The simulation starts balanced (both PEs at Vpe, both drawing
    Istring). After 1/4 of the simulation, the bottom PE current steps
    down by Imismatch (bottom draws less → Vmid rises → balancer should
    act).

    Switch width default (300µm) is sized for Ron×Cfly ≈ 1ns.

    Returns dict with numpy arrays.
    """
    fsw = fsw_MHz * 1e6
    T = 1.0 / fsw
    total_time = n_cycles * T
    step_time = total_time / 4
    Thalf = T / 2
    Tdead = T * 0.05

    # Pre-compute all timing values to avoid ngspice .param brace issues
    clk1_pw = Thalf - Tdead       # pulse width
    clk2_delay = Thalf            # delay for second clock

    I_bot_after = Istring_mA - Imismatch_mA
    _csv_placeholder = "__OUTPUT_CSV__"

    deck = f"""\
* SC balancer with physical PE loads
* PTM 90nm BSIM4 NMOS switches
.include {PTM_MODEL}

* === Battery (ideal — represents string supply output) ===
Vbat vstring 0 DC {Vbat}

* === Top PE: C + I from vstring to vmid ===
Cpe_top vstring vmid {Cpe_nF}n IC={Vpe}
Ipe_top vmid vstring DC {Istring_mA}m

* === Bottom PE: C + I from vmid to GND ===
* Current steps down at t={step_time}: {Istring_mA}mA → {I_bot_after}mA
Cpe_bot vmid 0 {Cpe_nF}n IC={Vpe}
Ipe_bot 0 vmid PWL(0 {Istring_mA}m {step_time} {Istring_mA}m {step_time + 1e-12} {I_bot_after}m)

* === Flying capacitor ===
Cfly cfly_p cfly_n {Cfly_nF}n

* === Balancer switches (1:1 from battery to bottom segment) ===
* Phase 1 (clk1): S1+S2 charge Cfly from battery (vstring→GND)
* Phase 2 (clk2): S3+S4 discharge Cfly to bottom segment (vmid→GND)
Ms1  vstring clk1 cfly_p 0  nmos w={W_um}u l={L_nm}n
Ms2  cfly_n  clk1 0      0  nmos w={W_um}u l={L_nm}n
Ms3  vmid    clk2 cfly_p 0  nmos w={W_um}u l={L_nm}n
Ms4  cfly_n  clk2 0      0  nmos w={W_um}u l={L_nm}n

* === Gate drive clocks (non-overlapping) ===
Vclk1 clk1 0 PULSE(0 {Vgs} 0 0.1n 0.1n {clk1_pw} {T})
Vclk2 clk2 0 PULSE(0 {Vgs} {clk2_delay} 0.1n 0.1n {clk1_pw} {T})

* === Initial conditions ===
.ic v(vmid)={Vpe}

* === Analysis ===
.tran 0.1n {total_time} UIC

.control
run
wrdata {_csv_placeholder} v(vstring) v(vmid) v(cfly_p) v(cfly_n) i(Vbat)
quit
.endc

.end
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "output.csv"
        final_deck = deck.replace(_csv_placeholder, str(csv_path))
        _run_ngspice(final_deck, tmpdir)
        if not csv_path.exists():
            raise RuntimeError("ngspice produced no output file")
        t, sigs = _parse_wrdata(str(csv_path), 5)

    v_string = sigs[0]
    v_mid = sigs[1]
    return {
        "t_ns": t * 1e9,
        "v_string": v_string,
        "v_mid": v_mid,
        "v_top": v_string - v_mid,   # voltage across top PE
        "v_bot": v_mid,               # voltage across bottom PE (= Vmid)
        "v_cfly": sigs[2] - sigs[3],  # v(cfly_p) - v(cfly_n)
        "i_bat_mA": -sigs[4] * 1e3,
        "step_time_ns": step_time * 1e9,
    }
