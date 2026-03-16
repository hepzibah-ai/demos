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

    Returns (time_array, dict of signal_name -> value_array).
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
        capture_output=True, text=True, timeout=30,
        cwd=workdir,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ngspice failed (rc={result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def run_balancer_normal(
    Vbat: float = 1.2,
    Vpe: float = 0.4,
    Vgs: float = 0.9,
    W_um: float = 10.0,
    L_nm: float = 90,
    Cfly_nF: float = 10.0,
    fsw_MHz: float = 100.0,
    n_cycles: int = 15,
    skip_cycles: int = 5,
) -> dict:
    """Simulate balancer normal mode (1:1 battery → bottom segment).

    Returns dict with numpy arrays:
        t_ns, v_a, v_b, v_cfly, clk1, clk2, i_bat, i_load
    """
    fsw = fsw_MHz * 1e6
    T = 1.0 / fsw
    total_time = n_cycles * T
    start_save = skip_cycles * T

    deck = f"""\
* Balancer normal mode: 1:1 half-bridge, PTM 90nm
.include {PTM_MODEL}

.param Vbat={Vbat} Vpe={Vpe} Vgate={Vgs}
.param Cfly_val={Cfly_nF}n
.param fsw_val={fsw_MHz}e6
.param Tperiod={{1/fsw_val}}
.param Thalf={{Tperiod/2}}
.param Tdead={{Tperiod*0.05}}
.param W_sw={W_um}u
.param L_sw={L_nm}n

Vbat  vbat  0  DC {{Vbat}}
Vload vload 0  DC {{Vpe}}

Cfly  a  b  {{Cfly_val}}

Ms1  vbat  clk1  a  0  nmos  w={{W_sw}}  l={{L_sw}}
Ms2  a     clk2  vload  0  nmos  w={{W_sw}}  l={{L_sw}}
Ms3  b     clk1  0  0  nmos  w={{W_sw}}  l={{L_sw}}
Ms4  b     clk2  0  0  nmos  w={{W_sw}}  l={{L_sw}}

Vclk1 clk1 0 PULSE(0 {{Vgate}} 0 0.1n 0.1n {{Thalf - Tdead}} {{Tperiod}})
Vclk2 clk2 0 PULSE(0 {{Vgate}} {{Thalf}} 0.1n 0.1n {{Thalf - Tdead}} {{Tperiod}})

.tran 0.01n {total_time} {start_save}

.control
run
wrdata output.csv v(a) v(b) v(a,b) v(clk1) v(clk2) i(Vbat) i(Vload)
quit
.endc

.end
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_ngspice(deck, tmpdir)
        csv_path = Path(tmpdir) / "output.csv"
        if not csv_path.exists():
            raise RuntimeError("ngspice produced no output file")
        t, sigs = _parse_wrdata(str(csv_path), 7)

    return {
        "t_ns": t * 1e9,
        "v_a": sigs[0],
        "v_b": sigs[1],
        "v_cfly": sigs[2],
        "clk1": sigs[3],
        "clk2": sigs[4],
        "i_bat_mA": -sigs[5] * 1e3,   # flip sign: current out of Vbat
        "i_load_mA": -sigs[6] * 1e3,  # flip sign: current into load
    }


def run_balancer_pump(
    Vpe: float = 0.4,
    Vgs: float = 0.9,
    W_um: float = 10.0,
    L_nm: float = 90,
    Cfly_nF: float = 10.0,
    fsw_MHz: float = 100.0,
    n_cycles: int = 15,
    skip_cycles: int = 5,
) -> dict:
    """Simulate balancer pump mode (bottom → top charge shuttle).

    Phase 1: Cfly charges across bottom segment (Vmid=Vpe → GND)
    Phase 2: Cfly dumps into top segment (Vstring=2*Vpe → Vmid=Vpe)

    Returns dict with numpy arrays:
        t_ns, v_a, v_b, v_cfly, clk1, clk2, i_top, i_bot
    """
    Vstring = 2 * Vpe
    fsw = fsw_MHz * 1e6
    T = 1.0 / fsw
    total_time = n_cycles * T
    start_save = skip_cycles * T

    deck = f"""\
* Balancer pump mode: bottom→top shuttle, PTM 90nm
.include {PTM_MODEL}

.param Vpe={Vpe} Vstring={Vstring} Vgate={Vgs}
.param Cfly_val={Cfly_nF}n
.param fsw_val={fsw_MHz}e6
.param Tperiod={{1/fsw_val}}
.param Thalf={{Tperiod/2}}
.param Tdead={{Tperiod*0.05}}
.param W_sw={W_um}u
.param L_sw={L_nm}n

* String and mid-point rails (ideal — the buck converter holds these)
Vstring vstring 0  DC {{Vstring}}
Vmid    vmid    0  DC {{Vpe}}

Cfly  a  b  {{Cfly_val}}

* Phase 1 switches: charge Cfly across bottom segment (vmid → GND)
* S1: vmid to Cfly+
Ms1  vmid   clk1  a  0  nmos  w={{W_sw}}  l={{L_sw}}
* S2: Cfly- to GND
Ms2  b      clk1  0  0  nmos  w={{W_sw}}  l={{L_sw}}

* Phase 2 switches: dump Cfly into top segment (vstring → vmid)
* S3: vstring to Cfly+
Ms3  vstring clk2  a  0  nmos  w={{W_sw}}  l={{L_sw}}
* S4: Cfly- to vmid
Ms4  b       clk2  vmid  0  nmos  w={{W_sw}}  l={{L_sw}}

Vclk1 clk1 0 PULSE(0 {{Vgate}} 0 0.1n 0.1n {{Thalf - Tdead}} {{Tperiod}})
Vclk2 clk2 0 PULSE(0 {{Vgate}} {{Thalf}} 0.1n 0.1n {{Thalf - Tdead}} {{Tperiod}})

.tran 0.01n {total_time} {start_save}

.control
run
wrdata output.csv v(a) v(b) v(a,b) v(clk1) v(clk2) i(Vstring) i(Vmid)
quit
.endc

.end
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_ngspice(deck, tmpdir)
        csv_path = Path(tmpdir) / "output.csv"
        if not csv_path.exists():
            raise RuntimeError("ngspice produced no output file")
        t, sigs = _parse_wrdata(str(csv_path), 7)

    return {
        "t_ns": t * 1e9,
        "v_a": sigs[0],
        "v_b": sigs[1],
        "v_cfly": sigs[2],
        "clk1": sigs[3],
        "clk2": sigs[4],
        "i_top_mA": -sigs[5] * 1e3,
        "i_bot_mA": -sigs[6] * 1e3,
    }
