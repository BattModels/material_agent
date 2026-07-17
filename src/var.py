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
QUEUE_MIN_PENDING = 15

# Soft refill goal shown to agents when the queue is below QUEUE_MIN_PENDING:
# the floor (hard, gates waiting) triggers the refill; this target is what the
# refill should AIM for -- fill the queue well beyond the floor with the most
# VALUABLE ready work (never padding with junk just to hit a number).
QUEUE_REFILL_TARGET = 50

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
# "competitive" enough to warrant an OH job. Used by the forgotten-OH reminder
# and cited by requirement-13 in the worker prompt so the two stay consistent.
# Set negative to disable forgotten-OH detection entirely.
GO_DEV_OH_THRESHOLD = 0.3

# --- Dialogue history log rotation (hist/his_<N>.txt) --------------------------
# Once the active his_<N>.txt file reaches this many bytes, close it and start
# the next (see src/history_log.py). Tune here.
HIST_ROTATE_BYTES = 1 * 1024**3  # ~1 GiB

# Runtime cache owned by src/history_log.py -- which his_<N>.txt is active and
# its current byte count. None until the first write_history() call in THIS
# process, which (re)computes both from disk. Do not set by hand.
hist_active_index = None
hist_active_bytes = None

# When the wait-tool's Gate-2 message lists more than this many forgotten jobs,
# there is plainly plenty of work to do, so it drops the "return to the
# supervisor / do a literature review" closer and just tells the worker to get on
# with it. At or below this count the closer is kept.
FORGOTTEN_CLOSER_SUPPRESS_ABOVE = 30

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