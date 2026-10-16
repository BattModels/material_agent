"""Regression + contract tests for the FEMWell coupled-mode calculator.

One real FEM solve (a few seconds) is shared across the module via a fixture.
Reference numbers come from
https://helgegehring.github.io/femwell/photonics/examples/coupled_mode_theory.html
"""
import math

import pytest

from src import fem_calculator as FC
from src.fem_calculator import CalculationError, CalculationTimeout, DesignNotViable
from src.fem_explog import RESULT_COLUMNS

DOC_DESIGN = dict(gap_um=0.4, wg1_width_um=0.45, wg2_width_um=0.46, core_height_um=0.22,
                  wavelength_um=1.55, n_core=3.4777, n_clad=1.444,
                  core_material="Si", clad_material="SiO2")


@pytest.fixture(scope="module")
def doc():
    return FC.compute_coupled_modes(DOC_DESIGN)


def test_contract_keys_match_the_explog_schema(doc):
    assert set(doc) == set(FC.RESULT_KEYS) == set(RESULT_COLUMNS)
    assert all(isinstance(v, float) and math.isfinite(v) for v in doc.values())


def test_reproduces_the_femwell_example(doc):
    assert doc["n_eff_symmetric"] == pytest.approx(2.3711, abs=2e-3)
    assert doc["n_eff_antisymmetric"] == pytest.approx(2.3492, abs=2e-3)
    assert doc["n_eff_wg1"] == pytest.approx(2.3495, abs=2e-3)
    assert doc["n_eff_wg2"] == pytest.approx(2.3707, abs=2e-3)
    assert doc["coupling_length_um"] == pytest.approx(35.31, abs=0.3)


def test_physical_consistency(doc):
    d = DOC_DESIGN
    for k in ("n_eff_wg1", "n_eff_wg2", "n_eff_symmetric", "n_eff_antisymmetric"):
        assert d["n_clad"] < doc[k] < d["n_core"]
    # supermodes bracket the isolated modes
    assert doc["n_eff_antisymmetric"] <= min(doc["n_eff_wg1"], doc["n_eff_wg2"]) + 1e-4
    assert doc["n_eff_symmetric"] >= max(doc["n_eff_wg1"], doc["n_eff_wg2"]) - 1e-4
    # reported L_c is the supermode beat half-period
    assert doc["coupling_length_um"] == pytest.approx(
        d["wavelength_um"] / (2 * (doc["n_eff_symmetric"] - doc["n_eff_antisymmetric"])), rel=1e-9)
    # a 10 nm mismatch detunes the cores: small power transfer, kappa below beta_c
    beta_c = math.pi / (2 * doc["coupling_length_um"])
    assert 0 < doc["max_power_transfer"] < 0.15
    assert 0 < doc["kappa_12_per_um"] < beta_c and 0 < doc["kappa_21_per_um"] < beta_c


@pytest.mark.parametrize("bad, key", [
    (dict(DOC_DESIGN, n_core=1.4, n_clad=1.5), "n_core"),
    (dict(DOC_DESIGN, gap_um=0), "gap_um"),
    ({k: v for k, v in DOC_DESIGN.items() if k != "wavelength_um"}, "wavelength_um"),
])
def test_invalid_designs_fail_fast_as_DesignNotViable(bad, key):
    # DesignNotViable, not a bare ValueError: the tool layer keys off the type to
    # decide whether to blame the DESIGN or the ENVIRONMENT.
    with pytest.raises(DesignNotViable) as ei:
        FC.compute_coupled_modes(bad)
    assert key in str(ei.value)


def test_below_cutoff_is_a_design_fault_not_an_environment_fault():
    tiny = dict(DOC_DESIGN, wg1_width_um=0.03, wg2_width_um=0.03,
                core_height_um=0.03, n_core=1.55, n_clad=1.444)
    with pytest.raises(DesignNotViable):
        FC.compute_coupled_modes(tiny)


def test_the_exception_hierarchy_keeps_the_two_causes_separable():
    assert issubclass(CalculationTimeout, CalculationError)
    assert not issubclass(DesignNotViable, CalculationError)
    assert not issubclass(CalculationError, DesignNotViable)


# --------------------------------------------------------------- subprocess hop
@pytest.mark.slow
def test_subprocess_result_is_identical_to_in_process(doc):
    out = FC.solve_in_subprocess(DOC_DESIGN)
    for k in RESULT_COLUMNS:
        assert out[k] == pytest.approx(doc[k], abs=1e-9)


@pytest.mark.slow
def test_subprocess_propagates_a_design_fault_as_DesignNotViable():
    with pytest.raises(DesignNotViable) as ei:
        FC.solve_in_subprocess(dict(DOC_DESIGN, n_core=1.4, n_clad=1.5))
    assert "n_core" in str(ei.value)


@pytest.mark.slow
def test_subprocess_timeout_is_reported_as_such():
    with pytest.raises(CalculationTimeout) as ei:
        FC.solve_in_subprocess(DOC_DESIGN, timeout_seconds=0.5)
    assert "0.5 s limit" in str(ei.value)


def test_a_child_that_dies_without_a_result_is_an_environment_error(monkeypatch):
    class Dead:
        returncode = -11            # e.g. SIGSEGV inside the C solver
        stdout, stderr = "", "Segmentation fault"
    monkeypatch.setattr(FC.subprocess, "run", lambda *a, **k: Dead())
    with pytest.raises(CalculationError) as ei:
        FC.solve_in_subprocess(DOC_DESIGN)
    msg = str(ei.value)
    assert "-11" in msg and "Segmentation fault" in msg
    assert not isinstance(ei.value, DesignNotViable)


def test_mesh_policy_matches_the_example_for_soi_and_pads_low_contrast():
    soi = FC._mesh_policy(DOC_DESIGN)
    assert (soi["w_sim"], soi["h_clad"], soi["h_box"], soi["core_res"], soi["background_res"]) == (4.0, 1.0, 1.0, 0.03, 0.2)
    su8 = FC._mesh_policy(dict(DOC_DESIGN, n_core=1.572, core_height_um=1.0, wg1_width_um=1.0, wg2_width_um=1.0))
    assert su8["h_box"] == 3.0 and su8["w_sim"] > 4.0 and su8["core_res"] == 0.08
    assert FC._mesh_policy(DOC_DESIGN, pad_scale=2.0)["h_box"] == 2.0
