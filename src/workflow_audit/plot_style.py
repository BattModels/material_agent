"""Shared plotting conventions for the workflow_audit notebooks (oer_results.ipynb,
aq_gnome_fingerprints.ipynb) -- single source of truth so colours/colormaps stay
consistent across notebooks instead of being hand-picked separately in each one.

Usage (from a notebook sitting in this same directory):
    from plot_style import GRAY, MUTED, PALETTE, BLUE, ORANGE, PID_CMAP, OVERPOTENTIAL_CMAP
"""
import seaborn as sns

GRAY = "#b0aeab"                        # background/"everything" cloud
MUTED = "#898781"                       # muted/de-emphasised points (e.g. abandoned candidates)

PALETTE = sns.color_palette("deep")     # seaborn's default qualitative palette
BLUE, ORANGE = PALETTE[0], PALETTE[1]

PID_CMAP = "gist_ncar"                  # process_id colouring
OVERPOTENTIAL_CMAP = "jet"              # eta_scaling / eta_ideal colouring
DEVIATION_CMAP = "viridis"              # G(O)/G(OH) deviation-from-optimal colouring
