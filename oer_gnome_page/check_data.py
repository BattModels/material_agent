#!/usr/bin/env python
"""Validate a built data.json against every assumption the page makes.

    python check_data.py docs/data.json

Exits non-zero if any check fails, so it can gate a deploy.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path


def load_payload(path):
    """Accept either a split build's data.json or a built single-file page.

    A --single build has no data.json -- the payload is inlined in a
    <script id="explog-data"> tag. `publish` always builds --single, so without
    this branch these checks could never run on the artifact that actually
    ships, which was the case until 2026-08-31.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".html", ".htm"):
        m = re.search(r'<script[^>]*id="explog-data"[^>]*>(.*?)</script>',
                      text, re.S)
        if not m:
            sys.exit('no <script id="explog-data"> block in %s' % path)
        text = m.group(1).strip()
        if not text:
            sys.exit("inline data block is empty in %s -- a split build's "
                     "index.html carries no data; check its data.json instead"
                     % path)
    return json.loads(text)


path = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/data.json")
d = load_payload(path)
CAND = {c["id"]: c for c in d["candidates"]}
PROC = d["processes"]
SITES = d.get("sites", [])
STRUCTS = d.get("structs", {})

fails = []
def check(name, ok, detail=""):
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        fails.append(name)

print("%s  (%.2f MB)" % (path, path.stat().st_size / 1e6))
print("candidates %d   processes %d   sites %d   structures %d"
      % (len(d["candidates"]), len(PROC), len(SITES), len(STRUCTS)))
print()

# ---------------------------------------------------------------- energetics
print("energetics")
full = [p for p in PROC if p.get("gO") is not None and p.get("gOH") is not None]
f4 = lambda x, y: max(y, x, 3.2 - x, 1.72 - y) - 1.23
fid = lambda x, y: max(y, x, (4.92 - (x + y)) / 2) - 1.23

bad = [p for p in full if p.get("eta_scaling") is not None
       and abs(f4(p["gO"] - p["gOH"], p["gOH"]) - p["eta_scaling"]) > 0.006]
check("eta_scaling = max(y,x,3.2-x,1.72-y) - 1.23", not bad, "%d/%d off" % (len(bad), len(full)))

bad = [p for p in full if p.get("eta_ideal") is not None
       and abs(fid(p["gO"] - p["gOH"], p["gOH"]) - p["eta_ideal"]) > 0.006]
check("eta_ideal = max(y,x,(4.92-G(O))/2) - 1.23", not bad, "%d/%d off" % (len(bad), len(full)))

oo = [p["gOOH"] - p["gOH"] for p in full if p.get("gOOH") is not None]
check("G(OOH) = G(OH) + 3.2 exactly", oo and max(abs(v - 3.2) for v in oo) < 1e-6,
      "n=%d" % len(oo))

# FED ladders must reproduce their own eta
def ladder(p, mode):
    ooh = (4.92 + p["gO"]) / 2 if mode == "ideal" else p["gOOH"]
    lv = [0, p["gOH"], p["gO"], ooh, 4.92]
    return max(lv[i] - lv[i - 1] for i in range(1, 5))

for mode, fld in (("scaling", "eta_scaling"), ("ideal", "eta_ideal")):
    bad = [p for p in full if p.get(fld) is not None and p.get("gOOH") is not None
           and abs(ladder(p, mode) - 1.23 - p[fld]) > 0.006]
    check("%s FED largest step - 1.23 = stored eta" % mode, not bad,
          "%d off" % len(bad))
print()

# ---------------------------------------------------------------- site join
print("sites")
from_proc = {(p["cand"], p["term"], p["site"]) for p in PROC
             if p["job"] in ("O_adsorption", "OH_adsorption")
             and p["term"] is not None and p["site"] is not None}
from_meta = {(m["cand"], m["term"], m["site"]) for m in SITES}
check("site metadata joins 1:1 with processes",
      from_proc == from_meta,
      "proc %d, meta %d, sym-diff %d" % (len(from_proc), len(from_meta),
                                         len(from_proc ^ from_meta)))
check("every candidate id in sites exists", all(m["cand"] in CAND for m in SITES))
types = Counter(m["type"] for m in SITES)
print("       site types:", dict(types))
print()

# ---------------------------------------------------------------- structures
print("structures")
meta = {(m["cand"], m["term"], m["site"]): m for m in SITES}
is_lat = lambda m: m["type"] == "lattice O" or m["el"] == "N/A"

def key(c, t, s, kind, lat):
    if lat:
        if kind == "surface":
            return "%s|%s|%s|O" % (c, t, s)
        if kind == "O":
            return "%s|%s|surface" % (c, t)
    return "%s|%s|surface" % (c, t) if kind == "surface" else "%s|%s|%s|%s" % (c, t, s, kind)

# Which sites have a finished adsorbate calculation. For a lattice-O site the
# '*' key IS its relaxed-O structure, which only exists once that job lands, so
# in a snapshot frozen mid-study its absence is expected, not a defect. For an
# on-top site the '*' key is the base surface, whose absence always is one.
have_energy = {(p.get("cand"), p.get("term"), p.get("site"))
               for p in PROC if p.get("gO") is not None}

if not STRUCTS:
    print("  [SKIP] structure invariants - payload carries no structures "
          "(--no-structures build)")
else:
    missing = [k for k in meta if key(*k, "surface", is_lat(meta[k])) not in STRUCTS]
    pending = [k for k in missing if is_lat(meta[k]) and k not in have_energy]
    real = [k for k in missing if k not in pending]
    check("every site with a finished calculation resolves a '*' structure",
          not real, "%d missing" % len(real))
    if pending:
        print("       %d lattice-O site(s) still awaiting their adsorbate job "
              "- no structure yet, not a defect" % len(pending))

lat_ok = lat_bad = top_ok = top_bad = 0
for k, m in (meta.items() if STRUCTS else ()):
    lat = is_lat(m)
    a = STRUCTS.get(key(*k, "surface", lat))
    b = STRUCTS.get(key(*k, "O", lat))
    if not a or not b:
        continue
    good = len(b["s"]) - len(a["s"]) == 1        # O* must have exactly one more atom
    if lat:
        lat_ok, lat_bad = lat_ok + good, lat_bad + (not good)
    else:
        top_ok, top_bad = top_ok + good, top_bad + (not good)
# Everything below reads STRUCTS. With no structures exported these would all
# pass on empty sets, which reads as verification but checks nothing.
if STRUCTS:
    check("lattice-O sites: n(O*) - n(*) = +1 after swap",
          lat_bad == 0 and lat_ok > 0, "%d ok, %d bad" % (lat_ok, lat_bad))
    check("on-top sites:    n(O*) - n(*) = +1",
          top_bad == 0 and top_ok > 0, "%d ok, %d bad" % (top_ok, top_bad))

    t4 = Counter()
    for k, v in STRUCTS.items():
        if k.endswith("|O") or k.endswith("|OH"):
            t4[(k.rsplit("|", 1)[1], sum(1 for x in v["t"] if x == 4))] += 1
    print("       adsorbate (tag 4) atoms per structure:", dict(t4))
    elements = {s for v in STRUCTS.values() for s in v["s"]}
    # read the real tables out of the template rather than trusting a copy here
    tpl = (Path(__file__).parent / "template.html").read_text(encoding="utf-8")
    KNOWN = set(re.findall(r'([A-Z][a-z]?):"[0-9A-F]{6}"',
                           re.search(r"const CPK=\{.*?\};", tpl, re.S).group(0)))
    HASR = set(re.findall(r'([A-Z][a-z]?):[0-9.]+',
                          re.search(r"const RCOV=\{.*?\};", tpl, re.S).group(0)))
    check("all elements have a colour entry", elements <= KNOWN,
          "%d in table; unknown: %s" % (len(KNOWN), sorted(elements - KNOWN) or "none"))
    check("all elements have a radius entry", elements <= HASR,
          "%d in table; unknown: %s" % (len(HASR), sorted(elements - HASR) or "none"))
print()

# ---------------------------------------------------------------- plot window
print("volcano window  x[0,3]  y[-0.7,2.7]")
pts = [(p["gO"] - p["gOH"], p["gOH"]) for p in full if p.get("eta_scaling") is not None]
ins = [1 for x, y in pts if 0 <= x <= 3 and -0.7 <= y <= 2.7]
print("       %d of %d sites inside (%d clipped, reported in the caption)"
      % (len(ins), len(pts), len(pts) - len(ins)))
print()

print("%d check(s) failed" % len(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
