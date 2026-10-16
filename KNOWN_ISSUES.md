# Known issues — `fem_demo` branch

Open problems that are understood and diagnosed but deliberately **not fixed yet**.
Each entry says what is wrong, the evidence, the effect on results, and the agreed fix.

---

## 1. Mesh asymmetry fakes a detuning, wrecking weakly-coupled symmetric designs

**Status:** open, diagnosed 2026-10-04. Fix agreed, not implemented.
**Where:** `src/fem_calculator.py`, `_solve` — the three `compute_modes` calls and the
CMT block that follows.

### What is wrong

`_solve` meshes the two cores independently and solves each isolated waveguide on its
own. For two *geometrically identical* waveguides that must give identical effective
indices, but the mesh is not mirror-symmetric, so they differ by ~1e-4:

```
sym Si gap0.2:  n1=2.349458  n2=2.349411   (n1-n2) = +4.7e-05
```

The CMT block reads that difference as a genuine detuning
`delta = 0.5*(n1*k0 + K22 - (n2*k0 + K11))` and propagates it into
`beta_c = sqrt(K12*K21 + delta**2)`, hence into both reported quantities:

```
max_power_transfer = |K21^2 / beta_c^2|       ->  1 / (1 + (delta/kappa)^2)
coupling_length_um = lambda / (2*(n_sym - n_asym))
```

`delta_noise` is roughly constant (~2e-4 to 8e-4) while `kappa` decays **exponentially**
with the gap. Once `kappa` falls to the noise level the result is pure artefact.

### Evidence

Symmetric coupler, 0.45 um wide, Si/SiO2, 1.55 um. Physics demands
`max_power_transfer = 1.000` and `excess_loss = 0 dB` on **every** row:

| gap um | kappa 1/um | P_max | loss dB | verdict |
|---|---|---|---|---|
| 0.10 | 0.14113 | 0.9997 | 0.001 | fine |
| 0.20 | 0.06605 | 1.0000 | 0.000 | fine |
| 0.30 | 0.02938 | 0.9992 | 0.004 | fine |
| 0.40 | 0.01300 | 0.9995 | 0.002 | fine |
| 0.50 | 0.00577 | 0.9987 | 0.006 | fine |
| 0.70 | 0.00115 | 0.8192 | 0.866 | **wrong** |
| 1.00 | 0.00010 | 0.0176 | 17.538 | **badly wrong** |

The arithmetic closes exactly: at gap 1.0, `delta ~ 7.7e-4` against `kappa = 1.0e-4`
gives `1/(1 + (7.7)^2) = 0.017`, matching the reported 0.0176. The coupling length is
contaminated the same way — it reads 2015 um where `pi/(2*kappa)` implies ~15700 um.

**Breakdown threshold:** roughly `kappa < 5e-3`, i.e. coupling lengths beyond ~300 um.
A real width mismatch is safe: 10 nm gives a detuning ~100x the noise floor.

### Why it matters

The 20 um target sits far inside the trustworthy region, so headline results are sound.
But `invoke.py`'s objective tells the agent to sweep the gap out to 1.0 um, straight
into the bad zone. It already did damage in `fem_test-2`: dc_0004 (gap 0.7) and dc_0005
(gap 1.0) are exactly symmetric, so their true loss is 0 dB, yet they reported 0.85 dB
and 17.4 dB and the agent **abandoned both** and concluded that large gaps are lossy.
That conclusion is false and is in its round-1 report.

### Agreed fix (three parts)

1. **Exploit the symmetry.** When the two cores are geometrically and materially
   identical, solve the isolated mode **once** and reuse it for both. Then `delta` is
   exactly 0 and `K12 == K21` by construction, so symmetric designs are exact at any
   gap. Also removes one of the three mode solves, so those designs get faster.
2. **Guard the asymmetric case.** Compare the computed detuning against the mesh noise
   floor (~1e-4 in effective index) and mark the result unreliable when they are
   comparable, instead of reporting a confident wrong number.
3. **Make the clamp honest.** `max_power = float(min(1.0, ...))` currently turns a raw
   1.5 — which would signal a real bug — into a perfect coupler silently. Clamp quietly
   for a small overshoot, flag or raise beyond a tolerance. (Today's overshoot is benign:
   the raw value for the symmetric 0.2 um gap design is 1.000172, from `K21/K12 = 1.00017`.)

**Regression test to add:** a symmetric gap sweep out to 1 um asserting 0 dB at every gap.

---

## 2. Eigensolver cost is unpredictable (3 s to 25 s per design)

**Status:** open, bounded but not explained.

Meshing is negligible (~0.17 s). All the time is in `compute_modes`. Same design,
repeated, is uniformly slow or uniformly fast, so it is **not** a warm-up effect and
not the `PAD_RETRY_SCALES` retry loop (traced: every case was a single pass):

| design | mesh | mode solves | total |
|---|---|---|---|
| symmetric 0.45/0.45, gap 0.4 | 0.16 s | 0.96 + 0.69 + 0.60 s | 2.9 s |
| asymmetric 0.45/0.46, gap 0.4 | 0.17 s | 11.9 + 5.7 + 6.7 s | 25.0 s |

Bounded in practice by `var.FEM_SOLVE_TIMEOUT_SECONDS` (120 s) now that each solve runs
in a child process. Worth understanding before any large sweep: a 10x spread in cost
makes the study budget hard to plan.

---

## 3. `src/workflow_audit` still parses OER tool names

**Status:** open, low priority.

The audit parser and notebooks key off the OER tool vocabulary (`submit_dft_job`, etc.)
and will not produce meaningful numbers for a FEM run. Offline only — it imports nothing
from the runtime and cannot affect a study.
