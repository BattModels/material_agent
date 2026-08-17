import json
from pathlib import Path

my_WORKING_DIRECTORY = None
my_SAVE_DIALOGUE = True
my_RESOURCE_DIRECTORY = {}
reflector_first_visit = True
original_objective = ""
startTime = ""
LLM_MODEL = ""
OTHER_GLOBAL_VARIABLES = {}
TOKEN_USAGE = []
TOTAL_TOKEN_USED = 0
GPU_AVAILABLE = False
path_to_data_directory = None

# Set by the write_report tool on a SUCCESSFUL report write, and consumed once by
# worker_agent_node AFTER the agent's stream this same turn, which uses them to
# compact the steps that report covers into a one-line digest (see src/past_steps.py):
# reportName -> the CANVAS key, reportId -> the artifact id, reportGist -> the worker's
# own one-sentence summary used as the digest label.
# Consume-and-clear, never read twice: these used to be cleared only inside the
# compaction block, so reportName latched and every later turn believed a fresh
# report had just landed.
reportName = ""
reportId = ""
reportGist = ""

# --- Study deadline windows -----------------------------------------------------
# Total study budget and the two deadline-relative windows for the queue floor,
# defined in DAYS (the tunable knobs); the *_SECONDS forms below are what the
# code reads. invoke.py's objective text derives its "maximum of N days" from
# STUDY_BUDGET_DAYS, so prose and gates cannot drift apart.
STUDY_BUDGET_DAYS = 30
# enforce_queue_floor=False on a plan step is HONORED only when remaining time
# < this many days; earlier it is coerced back to True (see worker_agent_node).
FLOOR_DISARM_WINDOW_DAYS = 2
# Path B (the "expand the study" refusal when no ready work exists) is disabled
# when remaining time < this many days -- the wait then simply proceeds.
# PATH_B_CUTOFF_DAYS doubles as the honest "too late to start anything new"
# threshold quoted to the BOSS in boss_prompt: inside this window a freshly
# submitted job could not finish, so concluding the study is defensible. The
# boss decides for itself -- there is deliberately NO hard gate on finishing --
# but it is given days remaining and the live queue counts so the decision is
# informed. (The 27-05 run concluded at 14.68/30 days having only ever been
# told ELAPSED time.)
PATH_B_CUTOFF_DAYS = 4

_SECONDS_PER_DAY = 86400
STUDY_BUDGET_SECONDS = STUDY_BUDGET_DAYS * _SECONDS_PER_DAY
FLOOR_DISARM_WINDOW_SECONDS = FLOOR_DISARM_WINDOW_DAYS * _SECONDS_PER_DAY
PATH_B_CUTOFF_SECONDS = PATH_B_CUTOFF_DAYS * _SECONDS_PER_DAY

# HPC queue policy: the hard floor on QUEUED (SLURM-pending) jobs. If fewer than
# this many jobs are queued, wait_for_update refuses to wait and tells the agent
# (with the real numbers) to refill. Queued -- not running -- jobs are the target:
# queued jobs are what feed nodes the instant they free up, and under fair-share
# a near-empty queue means the cluster is absorbing everything we submit.
QUEUE_MIN_PENDING = 25

# Soft refill goal shown to agents when the queue is below QUEUE_MIN_PENDING:
# the floor (hard, gates waiting) triggers the refill; this target is what the
# refill should AIM for -- fill the queue well beyond the floor with the most
# VALUABLE ready work (never padding with junk just to hit a number).
QUEUE_REFILL_TARGET = 100

# --- Disposition gate ---------------------------------------------------------
# Runtime slot mirroring the CURRENT task's `myStep.enforce_queue_floor` (the
# authoritative per-task default lives there). worker_agent_node copies
# plan[0].enforce_queue_floor here each turn (re-derived on resume); Gate 2 in
# wait_for_update reads it via getattr(var, "enforce_queue_floor", True). This
# initial value is just the pre-first-turn fallback, NOT a config knob.
enforce_queue_floor = True

# Runtime one-shot handback flag: wait_for_update raises this to True whenever it
# refuses on a queue-floor (Path A/B) or idle handback -- i.e. a situation the
# worker must return to the SUPERVISOR to resolve (submit ready work, or expand /
# wind down). NOT set for a Gate 1 disposition backlog (the worker clears that
# itself). supervisor_chain_node consumes-and-clears it while building its prompt,
# re-derives the path (classify_wait_handback) from live EXPLOG, and injects the
# matching directive. Process-global (same mechanism as enforce_queue_floor); a
# handback the supervisor never consumed is simply re-signalled next round.
wait_handback = False

# Runtime one-shot flag: worker_agent_node sets this when it coerced a plan
# step's enforce_queue_floor=False back to True (disarm requested earlier than
# the final FLOOR_DISARM_WINDOW_DAYS of the budget). supervisor_chain_node
# consumes-and-clears it and tells the supervisor directly, so the rule is
# learned from feedback at the moment it matters, not only from static prose.
floor_coerce_notice = False

# The fixed Decision vocabulary for a candidate disposition (single-valued).
# Two TERMINAL tags (no further compute) bracket three ACTIVE priority levels:
#   Abandon         -> terminal: stop, unpromising
#   Low priority    -> active: keep going, low priority
#   Medium priority -> active: keep going, medium priority (neutral default)
#   High priority   -> active: keep going, high priority
#   Sufficient      -> terminal: characterized, no further compute warranted
# A TERMINAL tag may only be set once the candidate is "fully settled" (no
# forgotten/ready work AND nothing in flight); a FAILED candidate may ONLY be
# Abandon. That gate is enforced in the disposition tool layer (src/tools.py),
# since find_forgotten_jobs lives in the parent. The active priority levels are
# assigned but currently DEFERRED -- not surfaced or used to drive planning (so
# they never discourage adding new candidates).
DISPOSITION_DECISIONS = ("Abandon", "Low priority", "Medium priority", "High priority", "Sufficient")
# Subsets/derived -- used by the terminal-tag gate (Part 2) and the resume
# reconciliation (Part 3); inert until then.
DISPOSITION_TERMINAL_DECISIONS = ("Abandon", "Sufficient")
DISPOSITION_ACTIVE_DECISIONS = ("Low priority", "Medium priority", "High priority")  # ascending
DISPOSITION_DEFAULT_ACTIVE = "Medium priority"   # neutral; legacy "Investigating" migrates here

# G(O) deviation (|G(O) - 2.46| eV) at/under which an O-adsorption site is
# "competitive" enough to warrant an OH job. Drives find_forgotten_jobs, hence
# which sites the wait-gate SURFACES to the worker as ready OH work -- that
# listing is how the value reaches the agent.
#
# DELIBERATELY NOT quoted in the worker prompt: requirement 13 keeps the
# qualitative wording ("far from the ideal value of 2.46 eV") so the number
# stays an operator knob rather than a figure the agent reasons about directly.
# (An earlier comment here claimed requirement 13 cited the value; it never did.)
# Set negative to disable forgotten-OH detection entirely.
GO_DEV_OH_THRESHOLD = 0.8

# --- Live visualisation write throttle -----------------------------------------
# LiveVisualizer._flush() rewrites BOTH live_data.js and live_visualization.html
# in full, on EVERY streamed agent event -- i.e. at least once per model call and
# once per tool call. Measured on the 27-05 run that is ~180 MB per event
# (80 MB js + 99 MB self-contained html), plus json.dumps of the whole state
# twice and a str.replace over the ~99 MB html. Unthrottled, that was MORE I/O
# than the checkpointer, for a dashboard nobody watches during an unattended run.
#
# Both files are pure VIEWS rebuilt from in-memory state, so skipping a write
# loses nothing -- the next flush emits the latest state. close() always forces
# a final write, so the end state is never missing.
LIVE_VIZ_ENABLED = True          # False disables the dashboard writes entirely
LIVE_VIZ_DATA_MIN_INTERVAL_S = 60      # live_data.js  (lean; polled by the page)
LIVE_VIZ_HTML_MIN_INTERVAL_S = 900     # live_visualization.html (heavy; ~99 MB)

# CANVAS.write() also re-dumps the ENTIRE canvas as a human-readable
# canvas.pickle.txt on every write -- 5 MB on the 27-05 run, x753
# write_my_canvas calls ~= 3.8 GB of pure inspection output. Nothing reads the
# file back (canvas.pickle is the real artefact), so it is safe to switch off.
CANVAS_TXT_DUMP_ENABLED = True

# --- Dialogue history log rotation (hist/his_<N>.txt) --------------------------
# Once the active his_<N>.txt file reaches this many bytes, close it and start
# the next (see src/history_log.py). Tune here.
HIST_ROTATE_BYTES = 1 * 1024**3  # ~1 GiB

# Runtime cache owned by src/history_log.py -- which his_<N>.txt is active and
# its current byte count. None until the first write_history() call in THIS
# process, which (re)computes both from disk. Do not set by hand.
hist_active_index = None
hist_active_bytes = None

# How many forgotten-work items the wait-tool's Gate-2 (Path A) message lists
# before collapsing the rest into "... and N more".
#
# find_forgotten_jobs returns the list BEST FIRST, so this cap is a BATCH SIZE,
# not a filter: the worker submits these, re-calls wait_for_update, and the next
# batch surfaces when capacity frees up. Before the list was sorted the same cap
# was a filter over registration order, and it hid 9 sites with G(O) deviation
# 0.055-0.294 eV inside "... and 42 more" for 88 consecutive refusals -- see
# _KIND_RANK in src/forgotten_jobs.py.
FORGOTTEN_JOBS_DISPLAY_CAP = 20

# Process ids that were already finalized before the first resume after the
# disposition-gate rollout. The wait-tool's Gate-1 coverage check treats these
# as exempt (the agent need not cite them to dispose a legacy candidate), but
# the agent is never blinded to them.
#
# Frozen snapshot in legacy_disposition_exempt_ids.json (generated once by
# dry_run_disposition.py from the production explog's finalized process ids).
#
# NOTE: that JSON is NOT committed -- the repo's '*' .gitignore keeps it
# untracked, so it lives ONLY in this working tree (the one we resume the 27-05
# run from), where it loads 488 ids. A fresh clone won't have it and falls back
# to the empty set below. That is CORRECT for anyone NOT resuming this exact run:
# leave it empty (or regenerate your own snapshot with dry_run_disposition.py).
# So a clone degrades to empty by design, not by accident.
#
# Loaded on EVERY resume; must NEVER be cleared for THIS run: a historical
# candidate is dispositioned with an EMPTY citation (its old units are exempt),
# so emptying this set later would make those old units "uncovered & not exempt"
# and re-block the candidate forever. The empty-set fallback keeps startup
# crash-proof if the file is absent.
try:
    # Coerce to int HERE (inside the try) so a wrong-shape file -- a dict, or a
    # list with a non-numeric element -- degrades to the empty set like a missing
    # file, instead of slipping through and crashing later at EXPLOG.init's int().
    LEGACY_DISPOSITION_EXEMPT_IDS: set[int] = {
        int(x) for x in json.loads(
            (Path(__file__).parent / "legacy_disposition_exempt_ids.json").read_text())
    }
except (OSError, ValueError, TypeError):
    LEGACY_DISPOSITION_EXEMPT_IDS = set()