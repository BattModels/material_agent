#!/usr/bin/env python
"""Build the OER explog site from a frozen explog pickle.

    python build.py explog.pkl                    -> docs/index.html + docs/data.json
    python build.py explog.pkl --out site         -> same, into site/
    python build.py explog.pkl --single demo.html -> one self-contained file

The pickle must be a dict with "candidates_df" and "processes_df". Everything
else (site metadata, relaxed structures) is optional: if `study_obj` is absent
or unreadable, the site simply omits structures and site geometry.

Cluster paths in VASP_dir are reduced to the run directory name.
"""
import argparse
import json
import pickle
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent

CAND_COLS = {
    "candidate_id": "id",
    "Reduced Formula": "formula",
    "state": "state",
    "decision": "decision",
    "G(O) deviation": "gO_dev",
    "Overpotential_from_scaling": "eta_scaling",
    "idealOverPotential": "eta_ideal",
    "max_dG_U[1.2,2.0]_pH0": "pourbaix",
    "n_bulk_started": "bulk_s", "n_bulk_finalized": "bulk_f",
    "n_surface_started": "surf_s", "n_surface_finalized": "surf_f",
    "n_O_started": "O_s", "n_O_finalized": "O_f",
    "n_OH_started": "OH_s", "n_OH_finalized": "OH_f",
    "reason_or_hypothesis": "reason",
    "notes": "notes",
}

PROC_COLS = {
    "process_id": "id",
    "candidate_id": "cand",
    "job_type": "job",
    "status": "status",
    "termination_index": "term",
    "site_index": "site",
    "VASP_dir": "dir",
    "G(O)": "gO",
    "G(OH)": "gOH",
    "G(OOH) from scaling relation": "gOOH",
    "ideal overpotential": "eta_ideal",
    "overpotential from OH-OOH scaling relation": "eta_scaling",
    "processNote": "note",
}


def clean(v):
    """pandas NA / numpy scalars -> plain JSON values."""
    if v is None:
        return None
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def take(row, col):
    """Read a column if the frame has it, else None. Keeps old/new schemas working."""
    try:
        return clean(row[col])
    except (KeyError, IndexError):
        return None


def export_tables(explog, warn):
    cdf, pdf = explog["candidates_df"], explog["processes_df"]

    for name, df, cols in (("candidates_df", cdf, CAND_COLS), ("processes_df", pdf, PROC_COLS)):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            warn("%s is missing %d column(s): %s" % (name, len(missing), ", ".join(missing)))

    candidates = []
    has_disp = "disposition_record" in cdf.columns
    for _, row in cdf.iterrows():
        rec = {short: take(row, col) for col, short in CAND_COLS.items()}
        rounds = []
        if has_disp:
            for entry in (row["disposition_record"] or []):
                if isinstance(entry, dict):
                    rounds.append({
                        "summary": entry.get("Summary"),
                        "plan": entry.get("Future_plan"),
                        "decision": entry.get("Decision"),
                        "pids": entry.get("Summarized_process_id"),
                    })
        rec["rounds"] = rounds
        candidates.append(rec)

    processes = []
    for _, row in pdf.iterrows():
        rec = {short: take(row, col) for col, short in PROC_COLS.items()}
        rec["dir"] = Path(str(rec["dir"])).name if rec["dir"] else None   # strip cluster path
        processes.append(rec)

    return candidates, processes


def norm_neighbors(v, limit=8):
    out = []
    try:
        for item in v:
            try:
                el, dist = item[0], float(item[1])
            except (TypeError, IndexError, ValueError):
                continue
            out.append([str(el), round(dist, 2)])
    except TypeError:
        return None
    out.sort(key=lambda t: t[1])
    return out[:limit] or None


def pack_atoms(atoms):
    try:
        return {
            "s": list(atoms.get_chemical_symbols()),
            "p": [round(float(v), 2) for xyz in atoms.get_positions() for v in xyz],
            "c": [round(float(v), 2) for r in atoms.get_cell() for v in r],
            "t": [int(t) for t in atoms.get_tags()],
        }
    except Exception:
        return None


def export_sites_and_structs(explog, want_structs, warn):
    """Walk study_obj for per-site geometry. Degrades to empty if unavailable."""
    cdf = explog["candidates_df"]
    sites, structs = [], {}
    if "study_obj" not in cdf.columns:
        warn("no study_obj column - site geometry and structures omitted")
        return sites, structs

    failed = 0
    for _, row in cdf.iterrows():
        cand = take(row, "candidate_id")
        study = row["study_obj"]
        if study is None:
            continue
        try:
            terms = getattr(study, "terminations", None) or {}
            surfaces = getattr(study, "oer_surface_studies", None) or {}
        except Exception:
            failed += 1
            continue

        for tk, ss in surfaces.items():
            slab = terms.get(tk)
            miller = getattr(slab, "miller_index", None)
            miller = [int(m) for m in miller] if miller is not None else None
            shift = clean(getattr(slab, "shift", None))

            surf_e = surf_f = None
            try:
                surf_e = round(float(ss.relaxed_surface_energy), 2)
            except Exception:
                pass
            base = getattr(ss, "base_surface_relaxed", None)
            try:
                surf_f = base.get_chemical_formula()
            except Exception:
                pass
            if want_structs:
                pk = pack_atoms(base)
                if pk:
                    structs["%s|%s|surface" % (cand, tk)] = pk

            df = getattr(ss, "add_sites_df_relaxed", None)
            byidx = {}
            if df is not None:
                try:
                    for _, r in df.iterrows():
                        byidx[int(r["Site index"])] = r
                except Exception:
                    pass

            for sk, st in (getattr(ss, "relaxed_surface_add_sites_studies", None) or {}).items():
                r = byidx.get(int(sk))
                get = lambda col: clean(r[col]) if (r is not None and col in r) else None
                stype = get("site type") or getattr(st, "site_type", None)
                el = get("ad site element")
                if el is None:
                    try:
                        el = base.get_chemical_symbols()[st.site_index]
                    except Exception:
                        pass
                sites.append({
                    "cand": cand, "term": int(tk), "site": int(sk),
                    "type": str(stype) if stype else None,
                    "el": str(el) if el else None,
                    "nbrs": norm_neighbors(get("ad site neighboring elements")),
                    "coord": get("reduced coordination"),
                    "miller": miller, "shift": round(shift, 3) if shift is not None else None,
                    "surfF": surf_f, "surfE": surf_e,
                })
                if want_structs:
                    for attr, tag in (("relaxed_O_atoms", "O"), ("relaxed_OH_atoms", "OH")):
                        pk = pack_atoms(getattr(st, attr, None))
                        if pk:
                            structs["%s|%s|%s|%s" % (cand, tk, sk, tag)] = pk

    if failed:
        warn("%d candidate(s) had an unreadable study_obj" % failed)
    return sites, structs


DESCRIPTION = ("Exploration log from an autonomous DFT screening campaign for "
               "oxygen evolution catalysts.")


def build_version():
    """Version shown on the page: the date this build was made, ISO 8601.

    Reads as "published on". The pickle it came from is recorded separately in
    meta.source, so a given page can still be traced back to its snapshot.
    """
    return date.today().isoformat()


def data_date(pkl):
    """The date the SNAPSHOT was frozen: the pickle's mtime.

    Distinct from `version`, which is when the page was built. Two pages built
    on different days from the same pickle share a data_date, and that is what
    lets a republished page be traced back to the study state it shows.
    """
    try:
        return date.fromtimestamp(pkl.stat().st_mtime).isoformat()
    except OSError:
        return None


def wrap_document(content, version=None):
    """Wrap the template body in a real HTML document.

    template.html is authored as a fragment. Served as-is a browser falls back
    to quirks mode and, with no <meta charset>, may mis-decode the Greek and
    maths characters (η, Δ, Å, ₂) unless the server happens to send UTF-8 in a
    header. Both matter once the file is published, so hoist <title> and the
    stylesheet into a proper <head>.
    """
    title = "OER-GNoME"
    m = re.search(r"<title>(.*?)</title>\s*", content, re.S)
    if m:
        title = m.group(1).strip()
        content = content[: m.start()] + content[m.end():]

    if version:
        title = "%s %s" % (title, version)

    styles = ""
    m = re.search(r"<style>.*?</style>\s*", content, re.S)
    if m:
        styles = m.group(0).strip()
        content = content[: m.start()] + content[m.end():]

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="description" content="%s">\n'
        "<title>%s</title>\n"
        "%s\n"
        "</head>\n"
        "<body>\n"
        "%s\n"
        "</body>\n"
        "</html>\n"
    ) % (DESCRIPTION, title, styles, content.strip())


def render(template, payload_json, data_src, version=None):
    """Inline the payload, or point the page at an external data.json."""
    safe = payload_json.replace("<", "\\u003c")     # never terminate the script tag early
    assert "</script" not in safe.lower()
    out = template.replace("__DATASRC__", data_src).replace("__DATA__", safe)
    assert "__DATA__" not in out and "__DATASRC__" not in out
    return wrap_document(out, version)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pickle", type=Path, help="frozen explog .pkl")
    ap.add_argument("--out", type=Path, default=Path("docs"),
                    help="output directory for index.html + data.json (default: docs)")
    ap.add_argument("--single", type=Path,
                    help="instead, write ONE self-contained html file at this path")
    ap.add_argument("--no-structures", action="store_true",
                    help="skip relaxed structures (smaller payload, no 3-D views)")
    ap.add_argument("--template", type=Path, default=HERE / "template.html")
    args = ap.parse_args()

    warnings = []
    warn = lambda m: (warnings.append(m), print("  warning: " + m))[1]

    if not args.pickle.exists():
        sys.exit("no such pickle: %s" % args.pickle)
    template = args.template.read_text(encoding="utf-8")

    print("reading %s (%.1f MB)" % (args.pickle.name, args.pickle.stat().st_size / 1e6))
    # stream from the file rather than read_bytes(): these pickles are 75 MB and
    # growing, and holding the raw bytes as well as the objects doubles the peak
    with args.pickle.open("rb") as fh:
        explog = pickle.load(fh)
    for k in ("candidates_df", "processes_df"):
        if k not in explog:
            sys.exit("pickle has no %r - keys present: %s" % (k, list(explog)))

    candidates, processes = export_tables(explog, warn)
    sites, structs = export_sites_and_structs(explog, not args.no_structures, warn)

    version = build_version()
    payload = {"meta": {"version": version, "source": args.pickle.name,
                        "data_date": data_date(args.pickle)},
               "candidates": candidates, "processes": processes,
               "sites": sites, "structs": structs}
    blob = json.dumps(payload, separators=(",", ":"), default=str)

    print("  candidates %4d   processes %5d   sites %4d   structures %4d"
          % (len(candidates), len(processes), len(sites), len(structs)))
    print("  version    %s" % version)
    print("  payload    %.2f MB" % (len(blob) / 1e6))

    if args.single:
        args.single.parent.mkdir(parents=True, exist_ok=True)
        args.single.write_text(render(template, blob, "data.json", version), encoding="utf-8")
        print("wrote %s (%.2f MB, self-contained)" % (args.single, args.single.stat().st_size / 1e6))
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "data.json").write_text(blob, encoding="utf-8")
        (args.out / "index.html").write_text(render(template, "", "data.json", version), encoding="utf-8")
        print("wrote %s/index.html (%.0f KB) + data.json (%.2f MB)"
              % (args.out, (args.out / "index.html").stat().st_size / 1e3, len(blob) / 1e6))
        print("preview locally:  python -m http.server -d %s" % args.out)

    if warnings:
        print("\n%d warning(s) - the site was still written." % len(warnings))


if __name__ == "__main__":
    main()
