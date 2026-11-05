# Working in this repo

- Driver runs on `venv2`: `module load Python/3.11.3-GCCcore-12.3.0` BEFORE
  `source venv2/bin/activate`, or it fails on a missing libpython.
- `invoke.py ow` deletes its working directory, ONLY to be used on test workflows
- `.gitignore` is a single `*` — `git add -u` for tracked files, `-f` for new ones.
- `python -m oer_gnome_page publish --push` rebuilds the public OER-GNoME page
  (https://matminator.github.io/OER-GNoME/) from the live fork's explog; omit
  `--push` to dry-run.
