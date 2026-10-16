"""The coupled-mode calculator behind the ``run_coupled_mode_analysis`` tool.

CONTRACT (the only thing the rest of the code base depends on):

    compute_coupled_modes(params: dict) -> dict

``params`` carries one directional-coupler design:
    gap_um, wg1_width_um, wg2_width_um, core_height_um, wavelength_um,
    n_core, n_clad  (core_material / clad_material are carried but unused here)

The returned dict has every key in ``RESULT_KEYS`` as a float. Any exception
signals a failed calculation; the tool layer records it on the process row.

Implementation: FEMWell finite-element mode solving + coupled-mode theory,
following https://helgegehring.github.io/femwell/photonics/examples/coupled_mode_theory.html
(the repo copy is coupled_mode_theory.py). Meshing is INTERNAL: the simulation
window and mesh resolution are derived from the design (index contrast and core
size), so the agent never sees a mesh parameter.

Pipeline per design (a few seconds for SOI at core resolution 0.03 um):
  1. cross-section polygons: core_1, core_2, cladding above, box below
  2. gmsh triangular mesh via femwell.mesh.mesh_from_OrderedDict (temp file)
  3. P0 permittivity field; compute_modes three times, keeping TE-like modes only
     (te_fraction > 0.5; thick cores make TM nearly degenerate with TE):
       both cores      -> 2 highest TE modes = symmetric / antisymmetric supermodes
       core_1 alone    -> its fundamental TE mode
       core_2 alone    -> its fundamental TE mode
  4. overlap integrals + coupling coefficients of the isolated modes -> kappa
     matrix, detuning delta, beta_c = sqrt(kappa12*kappa21 + delta^2)
  5. results:
       coupling_length_um  = lambda / (2 (n_sym - n_asym))   [supermode beating,
                             the quantity the FEMWell example prints as
                             "maximum power transfer after ... um"; equals
                             pi / (2 beta_c) from CMT]
       max_power_transfer  = |kappa21|^2 / |beta_c|^2          [CMT; 1 for a
                             symmetric coupler, < 1 when the cores are detuned]
       kappa_12/21_per_um  = |kappa| magnitudes, 1/um
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import numpy as np


class DesignNotViable(ValueError):
    """The DESIGN cannot be solved: bad parameters, no index contrast, or no
    guided mode. A real, reportable property of the design -- the tool records it
    as a failed process and the agent should disposition it."""


class CalculationError(RuntimeError):
    """The design could not be EVALUATED: the solver crashed, the child process
    died, or the environment is broken. Says nothing about the design, so the
    tool must not record it as a failed design."""


class CalculationTimeout(CalculationError):
    """The solve exceeded its wall-clock limit."""

RESULT_KEYS = (
    "n_eff_wg1",              # fundamental mode of waveguide 1 in isolation
    "n_eff_wg2",              # fundamental mode of waveguide 2 in isolation
    "n_eff_symmetric",        # symmetric (even) supermode of the coupled pair
    "n_eff_antisymmetric",    # antisymmetric (odd) supermode of the coupled pair
    "kappa_12_per_um",        # |coupling coefficient| 1 -> 2, 1/um
    "kappa_21_per_um",        # |coupling coefficient| 2 -> 1, 1/um
    "coupling_length_um",     # propagation length for maximum power transfer, um
    "max_power_transfer",     # peak fraction of power transferred (0..1)
)

# --- internal meshing policy (not exposed to the agent) -------------------------
# Field decay length outside the core scales as lambda / sqrt(n_core^2 - n_clad^2);
# pad the window by PAD_FACTOR of that, clamped, so weakly confined (low-contrast)
# modes are not truncated while SOI keeps the example's 1 um padding.
PAD_FACTOR = 1.5
PAD_MIN_UM, PAD_MAX_UM = 1.0, 3.0
# Core mesh: ~7.5 elements across the smallest core dimension, clamped so a
# 0.22 um SOI core lands exactly on the example's 0.03 um, with a coarser cap
# for big low-contrast cores.
CORE_RES_DIVISOR = 7.5
CORE_RES_MIN_UM, CORE_RES_MAX_UM = 0.03, 0.08
CORE_RES_DISTANCE_UM = 1.0        # how far the fine core resolution extends
BACKGROUND_RES_UM = 0.2           # scaled by pad/PAD_MIN so big windows stay cheap
MIN_WINDOW_WIDTH_UM = 4.0         # the example's window; never go below it
# If the supermodes do not bracket the isolated modes the window is truncating a
# weakly confined mode: enlarge the padding by these factors and retry.
PAD_RETRY_SCALES = (1.0, 1.7, 2.9)
# Polarization-aware mode selection: thick/square cores make the TM family nearly
# degenerate with TE, so the two lowest modes of the coupled pair can be the
# TE-symmetric and TM-symmetric supermodes rather than the TE pair. Solve a few
# extra modes and keep the highest-n_eff TE-like ones (te_fraction > TE_MIN).
N_MODES_BOTH, N_MODES_ISOLATED = 4, 2
TE_MIN = 0.5


def _mesh_policy(p: dict, pad_scale: float = 1.0) -> dict:
    n_core, n_clad = float(p["n_core"]), float(p["n_clad"])
    lam = float(p["wavelength_um"])
    decay = lam / math.sqrt(n_core**2 - n_clad**2)
    pad = min(max(PAD_FACTOR * decay, PAD_MIN_UM), PAD_MAX_UM) * pad_scale
    smallest = min(float(p["wg1_width_um"]), float(p["wg2_width_um"]), float(p["core_height_um"]))
    core_res = min(max(smallest / CORE_RES_DIVISOR, CORE_RES_MIN_UM), CORE_RES_MAX_UM)
    span = float(p["wg1_width_um"]) + float(p["gap_um"]) + float(p["wg2_width_um"])
    return {
        "w_sim": max(MIN_WINDOW_WIDTH_UM, span + 2 * pad),
        # the example's cladding spans y = 0..pad with the core inside it; keep
        # at least half a pad of cladding above a thick core
        "h_clad": max(pad, float(p["core_height_um"]) + 0.5 * pad),
        "h_box": pad,
        "core_res": core_res,
        "background_res": BACKGROUND_RES_UM * (pad / PAD_MIN_UM),
    }


def _validate(p: dict) -> None:
    for k in ("gap_um", "wg1_width_um", "wg2_width_um", "core_height_um", "wavelength_um", "n_core", "n_clad"):
        if k not in p:
            raise DesignNotViable(f"missing design parameter '{k}'")
        if not float(p[k]) > 0:
            raise DesignNotViable(f"design parameter '{k}' must be positive (got {p[k]!r})")
    if not float(p["n_core"]) > float(p["n_clad"]):
        raise DesignNotViable(
            f"n_core ({p['n_core']}) must exceed n_clad ({p['n_clad']}) for a guided mode."
        )


def compute_coupled_modes(params: dict) -> dict:
    """FEMWell mode solve + coupled-mode theory for one coupler design.

    Retries with a larger simulation window when the supermodes fail to bracket
    the isolated modes (the signature of a truncated, weakly confined mode)."""
    _validate(params)
    last = None
    for scale in PAD_RETRY_SCALES:
        out, ok = _solve(params, scale)
        if ok:
            return out
        last = out
    raise DesignNotViable(
        "the mode is too weakly confined for the largest simulation window tried "
        f"(supermodes {last['n_eff_symmetric']:.5f}/{last['n_eff_antisymmetric']:.5f} do not "
        f"bracket the isolated modes {last['n_eff_wg1']:.5f}/{last['n_eff_wg2']:.5f}); "
        "increase the index contrast or the core size."
    )


def _solve(params: dict, pad_scale: float) -> tuple[dict, bool]:
    """One FEM pass. Returns (results, consistent) where `consistent` is the
    supermode-bracketing check: n_asym <= min(n_iso) and n_sym >= max(n_iso)."""
    # Heavy imports kept local so importing src.tools stays cheap and so a
    # missing femwell only fails the calculation, not the whole agent.
    from scipy.constants import epsilon_0, speed_of_light
    from shapely.geometry import Polygon
    from skfem import Basis, ElementTriP0
    from skfem.io import from_meshio
    from femwell.maxwell.waveguide import compute_modes
    from femwell.mesh import mesh_from_OrderedDict

    w1, w2, gap, h = (float(params[k]) for k in ("wg1_width_um", "wg2_width_um", "gap_um", "core_height_um"))
    lam, n_core, n_clad = float(params["wavelength_um"]), float(params["n_core"]), float(params["n_clad"])
    k0 = 2 * np.pi / lam
    mp = _mesh_policy(params, pad_scale)
    w_sim, h_clad, h_box = mp["w_sim"], mp["h_clad"], mp["h_box"]

    # 1. geometry (core_1 to the left of x=0, core_2 to the right; cores sit on y=0)
    polygons = OrderedDict(
        core_1=Polygon([(-w1 - gap / 2, 0), (-w1 - gap / 2, h), (-gap / 2, h), (-gap / 2, 0)]),
        core_2=Polygon([(w2 + gap / 2, 0), (w2 + gap / 2, h), (gap / 2, h), (gap / 2, 0)]),
        clad=Polygon([(-w_sim / 2, 0), (-w_sim / 2, h_clad), (w_sim / 2, h_clad), (w_sim / 2, 0)]),
        box=Polygon([(-w_sim / 2, 0), (-w_sim / 2, -h_box), (w_sim / 2, -h_box), (w_sim / 2, 0)]),
    )
    resolutions = dict(
        core_1={"resolution": mp["core_res"], "distance": CORE_RES_DISTANCE_UM},
        core_2={"resolution": mp["core_res"], "distance": CORE_RES_DISTANCE_UM},
    )

    # 2. mesh (gmsh needs a file; use a temp dir so nothing lands in the cwd)
    with tempfile.TemporaryDirectory() as tmp:
        mesh = from_meshio(
            mesh_from_OrderedDict(
                polygons, resolutions,
                filename=os.path.join(tmp, "mesh.msh"),
                default_resolution_max=mp["background_res"],
            )
        )
    basis0 = Basis(mesh, ElementTriP0(), intorder=4)

    # 3. permittivity fields and mode solves
    def eps_with(*cores):
        e = basis0.zeros() + n_clad**2
        e[basis0.get_dofs(elements=cores if len(cores) > 1 else cores[0])] = n_core**2
        return e

    eps_both, eps_1, eps_2 = eps_with("core_1", "core_2"), eps_with("core_1"), eps_with("core_2")
    def te_modes(eps, n_modes, need, what):
        ms = compute_modes(basis0, eps, wavelength=lam, mu_r=1, num_modes=n_modes)
        te = sorted((m for m in ms if float(m.te_fraction) > TE_MIN),
                    key=lambda m: -float(np.real(m.n_eff)))
        if len(te) < need:
            raise DesignNotViable(
                f"found only {len(te)} TE-like mode(s) among the {n_modes} lowest modes of {what} "
                f"(need {need}); the design is at or below cutoff for the TE family."
            )
        return te[:need]

    modes_both = te_modes(eps_both, N_MODES_BOTH, 2, "the coupled pair")
    modes_1 = te_modes(eps_1, N_MODES_ISOLATED, 1, "waveguide 1")
    modes_2 = te_modes(eps_2, N_MODES_ISOLATED, 1, "waveguide 2")

    n_sym, n_asym = (float(np.real(m.n_eff)) for m in modes_both)   # already sorted descending
    n1, n2 = float(np.real(modes_1[0].n_eff)), float(np.real(modes_2[0].n_eff))
    for label, n in (("waveguide 1", n1), ("waveguide 2", n2), ("antisymmetric supermode", n_asym)):
        if not n > n_clad:
            raise DesignNotViable(
                f"{label} has no guided mode at {lam} um (n_eff={n:.4f} <= n_clad={n_clad:.4f}); "
                f"the design is below cutoff."
            )

    # 4. coupled-mode theory on the isolated modes (as in the FEMWell example)
    epsilons = [eps_1, eps_2]
    modes = list(chain(modes_1, modes_2))
    n_iso = len(modes_1)
    overlap = np.array([[mi.calculate_overlap(mj) for mj in modes] for mi in modes], dtype=complex)
    coupling = np.array(
        [[k0 * speed_of_light * epsilon_0 * 0.5
          * mi.calculate_coupling_coefficient(mj, epsilons[(j // n_iso + 1) % 2] - n_clad**2)
          for j, mj in enumerate(modes)] for mi in modes],
        dtype=complex,
    )
    denom = 1 - overlap[0, 1] * overlap[1, 0] / (overlap[0, 0] * overlap[1, 1])
    kappas = np.array(
        [[(coupling[i, j] - overlap[i, (i + 1) % 2] * coupling[(i + 1) % 2, j] / overlap[(i + 1) % 2, (i + 1) % 2]) / denom
          for i in range(2)] for j in range(2)]
    )
    delta = 0.5 * (n1 * k0 + kappas[1, 1] - (n2 * k0 + kappas[0, 0]))
    beta_c = np.sqrt(kappas[0, 1] * kappas[1, 0] + delta**2)

    # 5. results
    # KNOWN ISSUE (see KNOWN_ISSUES.md #1): for two IDENTICAL cores delta must be 0 by
    # symmetry, but the mesh is not mirror-symmetric, so n1 != n2 by ~1e-4 and that fake
    # detuning enters beta_c. Harmless while kappa >> 1e-4, but kappa decays exponentially
    # with the gap, so beyond kappa ~ 5e-3 (L_c > ~300 um) BOTH numbers below are artefact:
    # a symmetric coupler at a 1.0 um gap reports 17.5 dB of loss where physics says 0 dB.
    # Fix agreed, not implemented: solve the isolated mode once when the cores are identical.
    coupling_length = lam / (2 * (n_sym - n_asym))
    # The min() also silently hides a raw value > 1, which would signal a real bug
    # (today's overshoot is benign: 1.000172 for the symmetric 0.2 um gap design).
    max_power = float(min(1.0, abs(kappas[1, 0] ** 2 / beta_c**2)))
    out = {
        "n_eff_wg1": n1,
        "n_eff_wg2": n2,
        "n_eff_symmetric": n_sym,
        "n_eff_antisymmetric": n_asym,
        "kappa_12_per_um": float(abs(kappas[0, 1])),
        "kappa_21_per_um": float(abs(kappas[1, 0])),
        "coupling_length_um": float(coupling_length),
        "max_power_transfer": max_power,
    }
    bad = [k for k, v in out.items() if not math.isfinite(v)]
    if bad:
        raise CalculationError(f"the solver produced non-finite result(s): {bad}")
    tol = 1e-4
    consistent = (n_asym <= min(n1, n2) + tol) and (n_sym >= max(n1, n2) - tol)
    return out, consistent


# ---------------------------------------------------------------------------
# child-process driver
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]


def solve_in_subprocess(params: dict, timeout_seconds=None) -> dict:
    """Run one design in a CHILD PROCESS and return its results.

    Why not in-process: ``gmsh.initialize()`` installs a SIGINT handler, which
    CPython allows only on the main thread -- and LangGraph executes tool calls on
    ThreadPoolExecutor workers, so every in-process solve died with "signal only
    works in main thread of the main interpreter". The child does its work on its
    own main thread, so the problem disappears; it also contains a solver crash or
    a hang, which would otherwise take the whole study down.

    Raises DesignNotViable (the design is bad), CalculationTimeout, or
    CalculationError (the environment is broken). Results are unaffected by
    running out-of-process: the solve is a pure function of the parameters.
    """
    if timeout_seconds is None:
        try:
            from src import var
            timeout_seconds = getattr(var, "FEM_SOLVE_TIMEOUT_SECONDS", 120)
        except Exception:
            timeout_seconds = 120
    timeout = float(timeout_seconds) if timeout_seconds and float(timeout_seconds) > 0 else None

    with tempfile.TemporaryDirectory() as tmp:
        p_in, p_out = Path(tmp) / "params.json", Path(tmp) / "result.json"
        p_in.write_text(json.dumps(params, default=float))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(_REPO_ROOT), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
        cmd = [sys.executable, "-m", "src.fem_calculator", str(p_in), str(p_out)]
        try:
            proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env, timeout=timeout,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            raise CalculationTimeout(
                f"the solve exceeded its {timeout:g} s limit and was cancelled. The design "
                f"may be pathological for the eigensolver; try a different geometry."
            ) from None

        if not p_out.exists():
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            raise CalculationError(
                f"the solver process exited with code {proc.returncode} without producing a "
                f"result{': ' + ' | '.join(tail) if tail else ''}."
            )
        payload = json.loads(p_out.read_text())

    if payload.get("ok"):
        return payload["result"]
    kind, message = payload.get("kind"), payload.get("message", "unknown failure")
    if kind == "not_viable":
        raise DesignNotViable(message)
    raise CalculationError(message)


if __name__ == "__main__":
    # Child entry point: read params JSON, write a result JSON envelope.
    #   python -m src.fem_calculator params.json result.json
    # Always exits 0 when it managed to write the envelope; the parent reads `ok`
    # and `kind` from it, so a bad design and a broken environment stay distinct.
    _params = json.loads(Path(sys.argv[1]).read_text())
    try:
        _payload = {"ok": True, "result": compute_coupled_modes(_params)}
    except DesignNotViable as _e:
        _payload = {"ok": False, "kind": "not_viable", "message": str(_e)}
    except BaseException as _e:  # noqa: BLE001 -- the parent classifies, we just report
        _payload = {"ok": False, "kind": "error", "message": f"{type(_e).__name__}: {_e}"}
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(_payload))
    print(json.dumps(_payload, indent=2))
