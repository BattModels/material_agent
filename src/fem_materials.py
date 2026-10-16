"""Small refractive-index table for the coupler screening study.

Indices are representative room-temperature values in the telecom window and
are deliberately simple: one linear-in-wavelength fit per material between
1.3 and 1.6 um. Good enough for a screening study whose purpose is to explore
trends; swap in a proper dispersion model (e.g. Sellmeier via ``refractiveindex.info``)
when a real solver is wired in.

Pure and stdlib-only so it can be imported by the tools, the prompt and the
tests without pulling in anything heavy.
"""
from __future__ import annotations

# material -> (n at 1.30 um, n at 1.60 um, role hint, default core height in um)
# The role hint is advisory: the agent may still use any material as core or
# cladding as long as n_core > n_clad.
_TABLE: dict[str, tuple[float, float, str, float | None]] = {
    "Si":     (3.5036, 3.4700, "core", 0.220),
    "SiN":    (1.9940, 1.9890, "core", 0.400),
    "LiNbO3": (2.2200, 2.2050, "core", 0.600),
    "SU8":    (1.5750, 1.5700, "core", 1.000),
    "SiO2":   (1.4470, 1.4430, "cladding", None),
    "Air":    (1.0000, 1.0000, "cladding", None),
}

WAVELENGTH_MIN_UM = 1.30
WAVELENGTH_MAX_UM = 1.60


def known_materials() -> list[str]:
    return list(_TABLE.keys())


def default_core_height_um(material: str) -> float | None:
    """Typical platform thickness for a core material, or None for claddings."""
    return _TABLE[_canon(material)][3]


def _canon(material: str) -> str:
    """Case-insensitive lookup; raises a helpful error for unknown names."""
    for k in _TABLE:
        if k.lower() == str(material).strip().lower():
            return k
    raise KeyError(
        f"Unknown material '{material}'. Known materials: {known_materials()}. "
        f"Alternatively pass an explicit refractive index."
    )


def refractive_index(material: str, wavelength_um: float) -> float:
    """Linear interpolation of n between the tabulated 1.30 and 1.60 um points.

    Wavelengths outside [1.30, 1.60] are extrapolated linearly (the study grid
    stays inside the window; this just avoids a hard failure at the edges).
    """
    n_lo, n_hi, _, _ = _TABLE[_canon(material)]
    t = (float(wavelength_um) - WAVELENGTH_MIN_UM) / (WAVELENGTH_MAX_UM - WAVELENGTH_MIN_UM)
    return n_lo + t * (n_hi - n_lo)


def materials_table(wavelengths_um=(1.30, 1.40, 1.50, 1.55, 1.60)) -> str:
    """Render the material library as a plain-text table for the agent."""
    header = ["material", "role"] + [f"n@{w:.2f}um" for w in wavelengths_um] + ["default_core_height_um"]
    rows = [header]
    for m, (_, _, role, h) in _TABLE.items():
        rows.append(
            [m, role]
            + [f"{refractive_index(m, w):.4f}" for w in wavelengths_um]
            + ["" if h is None else f"{h:.3f}"]
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in rows)
