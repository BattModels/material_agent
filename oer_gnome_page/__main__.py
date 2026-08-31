#!/usr/bin/env python
"""OER-GNoME page — build, check, preview and publish.

Run from the directory that contains this package:

    python -m oer_gnome_page build   path/to/explog.pkl
    python -m oer_gnome_page check   OER-GNoME.html
    python -m oer_gnome_page serve   OER-GNoME.html
    python -m oer_gnome_page publish path/to/explog.pkl --push

`publish` uploads one self-contained index.html to the hosting repository as a
single commit with no parent, force-replacing whatever was there. The hosting
repo therefore holds only the current snapshot and never accumulates history.
The commit is assembled in a throwaway repository, so nothing is written into
this repository's object store.
"""
import argparse
import subprocess
import sys
import tempfile
import webbrowser
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent

SITE_REPO = "https://github.com/Matminator/OER-GNoME.git"
SITE_BRANCH = "main"


# ----------------------------------------------------------------- helpers
def sh(cmd, cwd=None, check=True, input_=None):
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                       input=input_, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit("failed: %s\n%s%s" % (" ".join(map(str, cmd)), r.stdout, r.stderr))
    return r


def py(script, *args, **kw):
    return sh([sys.executable, "-X", "utf8", HERE / script, *args], **kw)


def step(msg):
    print("\n=== %s ===" % msg)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def need(p, what):
    if not p.exists():
        sys.exit("no such %s: %s" % (what, p))
    return p


# ----------------------------------------------------------------- commands
def validate(target, quiet=False):
    """Run whichever checks apply to a built page. Returns a failure count.

    `quiet` shortens a PASS to one line; a failure always prints in full, since
    that is exactly when the detail matters.
    """
    page = target / "index.html" if target.is_dir() else target
    data = target / "data.json" if target.is_dir() else None
    bad = 0

    def report(r):
        # not `stdout or stderr`: a checker that prints and THEN crashes has
        # both, and dropping stderr hides the traceback behind partial output.
        out = ((r.stdout or "") + (r.stderr or "")).rstrip()
        failed = r.returncode != 0
        if quiet and not failed and out:
            print("  " + out.splitlines()[-1].strip())
        elif out:
            print(out)
        return failed

    bad += report(py("check_js.py", need(page, "page"), check=False))
    # check_data reads a split build's data.json, or the payload inlined in a
    # single-file page. Publishing always builds --single, so without the
    # second branch the data invariants never guarded a publish at all.
    probe = data if (data and data.exists()) else (
        page if page.suffix.lower() in (".html", ".htm") else None)
    if probe:
        bad += report(py("check_data.py", probe, check=False))
    return bad


def cmd_build(a):
    pkl = need(resolve(a.pickle), "pickle")
    if a.split:
        out = resolve(a.output or "docs")
        args = [pkl, "--out", out]
    else:
        out = resolve(a.output or "OER-GNoME.html")
        args = [pkl, "--single", out]
    if a.no_structures:
        args.append("--no-structures")

    print(py("build.py", *args).stdout.rstrip())

    # always validate: an unchecked build can be silently dead, and this costs
    # milliseconds. `check` stays available for pages built elsewhere.
    print("\n=== validating ===")
    bad = validate(out, quiet=True)
    if bad:
        sys.exit("\n%d check(s) failed — do not publish this build" % bad)
    print("\nopen it with:  python -m oer_gnome_page serve %s" % out.name)
    return 0


def cmd_check(a):
    return validate(need(resolve(a.target or "OER-GNoME.html"), "file or directory"))


def cmd_serve(a):
    target = need(resolve(a.target or "OER-GNoME.html"), "file or directory")
    if target.is_file():
        # a self-contained page opens straight from disk; no server needed
        print("opening %s" % target)
        webbrowser.open(target.as_uri())
        return 0
    url = "http://localhost:%d/" % a.port
    print("serving %s at %s   (ctrl-c to stop)" % (target, url))
    try:
        webbrowser.open(url)
    except Exception:
        pass
    subprocess.run([sys.executable, "-m", "http.server", str(a.port), "-d", str(target)])
    return 0


def cmd_publish(a):
    pkl = need(resolve(a.pickle), "pickle")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        stage.mkdir()

        step("1/5  lint the template")
        py("check_js.py", HERE / "template.html")
        print("  template parses")

        step("2/5  build the self-contained page")
        args = [pkl, "--single", stage / "index.html"]
        if a.no_structures:
            args.append("--no-structures")
        print(py("build.py", *args).stdout.rstrip())

        step("3/5  validate")
        bad = validate(stage / "index.html", quiet=True)
        if bad and not a.ignore_data_checks:
            sys.exit("\n%d check(s) failed - refusing to publish.\n"
                     "Inspect above, or re-run with --ignore-data-checks if the "
                     "failure is understood and acceptable." % bad)
        if bad:
            print("  %d check(s) failed, overridden by --ignore-data-checks" % bad)

        step("4/5  assemble the upload")
        (stage / ".nojekyll").write_text("", encoding="utf-8")
        readme = HERE / "repo_files" / "README.md"
        if readme.exists():
            (stage / "README.md").write_text(readme.read_text(encoding="utf-8"),
                                             encoding="utf-8")
        for f in sorted(stage.iterdir()):
            print("  %-12s %8.2f MB" % (f.name, f.stat().st_size / 1e6))

        # throwaway repo: keeps the workflow repo's object store untouched
        work = Path(tmp) / "repo"
        work.mkdir()
        sh(["git", "init", "-q", "."], cwd=work)
        sh(["git", "config", "user.email", "oer-gnome@local"], cwd=work)
        sh(["git", "config", "user.name", "OER-GNoME publisher"], cwd=work)
        for f in stage.iterdir():
            (work / f.name).write_bytes(f.read_bytes())
        sh(["git", "add", "-A"], cwd=work)
        msg = a.message or "OER-GNoME — %s (%s)" % (date.today().isoformat(), pkl.stem)
        sh(["git", "commit", "-qm", msg], cwd=work)

        step("5/5  publish")
        if not a.push:
            print("  built and staged, not pushed.")
            print("  re-run with --push to replace %s on %s" % (a.branch, a.repo))
            return 0

        print("  force-pushing to %s (%s)" % (a.repo, a.branch))
        r = sh(["git", "push", "--force", a.repo, "HEAD:refs/heads/%s" % a.branch],
               cwd=work, check=False)
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            sys.exit("push failed — check the repo exists and you have access")
        print("  done. GitHub Pages redeploys in about a minute.")
        print("  First time only: Settings -> Pages -> branch %s, folder / (root)" % a.branch)
    return 0


# ----------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(
        prog="python -m oer_gnome_page", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the page from an explog pickle")
    b.add_argument("pickle")
    b.add_argument("-o", "--output", help="output file, or directory with --split")
    b.add_argument("--split", action="store_true",
                   help="index.html + data.json instead of one self-contained file")
    b.add_argument("--no-structures", action="store_true",
                   help="skip the 3-D structures: much smaller and faster")
    b.set_defaults(fn=cmd_build)

    c = sub.add_parser("check", help="validate a built page")
    c.add_argument("target", nargs="?")
    c.set_defaults(fn=cmd_check)

    s = sub.add_parser("serve", help="open a built page, or serve a split build")
    s.add_argument("target", nargs="?")
    s.add_argument("-p", "--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    p = sub.add_parser("publish", help="build and force-replace the hosted page")
    p.add_argument("pickle")
    p.add_argument("--push", action="store_true", help="actually push (makes it public)")
    p.add_argument("-m", "--message")
    p.add_argument("--repo", default=SITE_REPO)
    p.add_argument("--branch", default=SITE_BRANCH)
    p.add_argument("--no-structures", action="store_true")
    p.add_argument("--ignore-data-checks", action="store_true",
                   help="publish even if the data invariants fail")
    p.set_defaults(fn=cmd_publish)

    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
