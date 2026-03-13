"""Quantization noise test for float formats on Gaussian data.

For each format (E5M2, E4M3, E2M3, E2M1 — all with subnormals, no NaN, no Inf),
quantize 256 Gaussian samples at various scale factors and report RMS error.
"""

import numpy as np

# ── Build representable values for each format ──

def build_format_values(ebits, mbits):
    """Return sorted array of all non-negative representable values.

    Convention: subnormals present, no NaN, no infinities.
    All exponent bit patterns (including max) are valid.
    """
    bias = 2**(ebits - 1) - 1
    max_exp = (1 << ebits) - 1  # all 1s — valid (no inf/nan)

    values = []

    # Exponent = 0: subnormals.  value = 2^(1-bias) * 0.mantissa
    for m in range(1 << mbits):
        frac = m / (1 << mbits)
        values.append(2.0**(1 - bias) * frac)

    # Exponent = 1 .. max_exp: normals.  value = 2^(e-bias) * 1.mantissa
    for e in range(1, max_exp + 1):
        for m in range(1 << mbits):
            frac = 1.0 + m / (1 << mbits)
            values.append(2.0**(e - bias) * frac)

    return np.array(sorted(set(values)))


def quantize(x, pos_values):
    """Round-to-nearest quantization using precomputed positive representable values.

    Clips to ±max representable.
    """
    signs = np.sign(x)
    ax = np.abs(x)
    max_val = pos_values[-1]
    ax = np.clip(ax, 0, max_val)
    # Find nearest representable value
    idx = np.searchsorted(pos_values, ax)
    # Compare with neighbors
    idx = np.clip(idx, 0, len(pos_values) - 1)
    # Check idx and idx-1, pick closer
    lo = np.clip(idx - 1, 0, len(pos_values) - 1)
    hi = idx
    d_lo = np.abs(ax - pos_values[lo])
    d_hi = np.abs(ax - pos_values[hi])
    best = np.where(d_lo <= d_hi, lo, hi)
    result = pos_values[best] * signs
    return result


# Formats: (name, ebits, mbits)
formats = [
    ("E1M6 (8b)", 1, 6),
    ("E2M5 (8b)", 2, 5),
    ("E4M3 (8b)", 4, 3),
    ("E5M2 (8b)", 5, 2),
    ("E2M3 (6b)", 2, 3),
    ("E2M1 (4b)", 2, 1),
]

# Generate Gaussian data
rng = np.random.default_rng(42)
x = rng.standard_normal(256)
sigma = x.std()

# Scale factors: ratio of max representable to data
# "scale=k" means we map k*sigma to max representable value
scale_factors = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, "peak"]

print(f"Data: 256 Gaussian samples, σ = {sigma:.4f}, peak = {np.abs(x).max():.4f} ({np.abs(x).max()/sigma:.2f}σ)")
print()

# Header
header = f"{'Format':<12}"
for sf in scale_factors:
    if sf == "peak":
        header += f"  {'peak':>8}"
    else:
        header += f"  {sf:>5.1f}σ  "
print(header)
print("-" * len(header))

for name, ebits, mbits in formats:
    pos_values = build_format_values(ebits, mbits)
    max_repr = pos_values[-1]
    n_values = len(pos_values) * 2 - 1  # +/- plus zero

    row = f"{name:<12}"
    errors = []
    for sf in scale_factors:
        if sf == "peak":
            # Scale so peak maps to max representable
            scale = max_repr / np.abs(x).max()
        else:
            # Scale so sf*sigma maps to max representable
            scale = max_repr / (sf * sigma)

        scaled = x * scale
        quantized = quantize(scaled, pos_values)
        # Unscale
        result = quantized / scale
        rms_err = np.sqrt(np.mean((result - x)**2))
        errors.append((sf, rms_err))
        row += f"  {rms_err:>8.5f}"

    # Find best
    best_sf, best_err = min(errors, key=lambda t: t[1])
    best_label = "peak" if best_sf == "peak" else f"{best_sf:.1f}σ"
    row += f"   best: {best_label} ({best_err:.5f})"
    print(row)

print()
print("RMS error (in units of the original data)")
print()

# Also show format details
print("Format details (no NaN, no Inf, with subnormals):")
for name, ebits, mbits in formats:
    pos_values = build_format_values(ebits, mbits)
    print(f"  {name}: {len(pos_values)*2-1} codepoints, "
          f"max = {pos_values[-1]}, "
          f"min_normal = {2.0**(1-(2**(ebits-1)-1))}, "
          f"min_subnormal = {pos_values[1] if len(pos_values) > 1 else 'N/A'}")
