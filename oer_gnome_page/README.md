# oer_gnome_page

Builds and publishes the OER-GNoME database page from a frozen explog pickle.

Run from the directory that *contains* this one:

```bash
python -m oer_gnome_page build   <explog.pkl>            # -> OER-GNoME.html
python -m oer_gnome_page check   OER-GNoME.html          # does the page actually run?
python -m oer_gnome_page serve   OER-GNoME.html          # open it in a browser
python -m oer_gnome_page publish <explog.pkl> --push     # replace the live page
```

`publish` force-replaces https://github.com/Matminator/OER-GNoME with a single
parentless commit, so the hosting repo only ever holds the current snapshot.
The commit is built in a throwaway repository, so nothing is written into this
repository. Without `--push` it builds and validates but uploads nothing.

## Files

| | |
|---|---|
| `__main__.py` | the CLI |
| `build.py` | pickle -> page. Usable on its own: `python build.py <pkl> --single out.html` |
| `template.html` | the page itself: layout, CSS, JS, tab content. Edit this to change appearance |
| `check_js.py` | catches unterminated literals / unbalanced brackets — a JS syntax error yields a page that looks fine and is completely dead |
| `check_data.py` | 12 invariants: both overpotential formulas, the FED ladders, the site join, the lattice-O structure swap, element coverage |
| `repo_files/README.md` | shipped to the hosting repo as its landing page |

## Requirements

`build.py` unpickles `study_obj`, so the environment must be able to import
`gnome_dreams_oer_screening`, `pymatgen` and `ase`. Everything else is stdlib.

`--no-structures` still needs them (unpickling happens first) but produces a
much smaller page, and is a good deal faster.

## Editing the page

After changing `template.html`, always run `check` before publishing. A syntax
error silently kills the whole page rather than degrading it.
