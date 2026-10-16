"""Process-global runtime state and tunable knobs.

Everything here is either (a) a tunable knob read by several modules, or (b) a
runtime slot written by one module and consumed by another (the "consume-and-
clear" one-shot flags). Nothing here is checkpointed; anything that must survive
a resume lives in the LangGraph state (see src/planNexe2.py) or in EXPLOG's own
pickle (see src/fem_explog.py).
"""

my_WORKING_DIRECTORY = None
my_SAVE_DIALOGUE = True
startTime = ""
OTHER_GLOBAL_VARIABLES = {}
TOKEN_USAGE = []

# Set by the write_report tool on a SUCCESSFUL report write, and consumed once by
# worker_agent_node AFTER the agent's stream this same turn, which uses them to
# compact the steps that report covers into a one-line digest (see src/past_steps.py):
# reportName -> the CANVAS key, reportId -> the artifact id, reportGist -> the worker's
# own one-sentence summary used as the digest label.
# Consume-and-clear, never read twice.
reportName = ""
reportId = ""
reportGist = ""

# One-shot handoff used on a mid-round resume: invoke.py stores the pre-crash
# result ids here and worker_agent_node consumes them instead of resetting the
# current-round id list (see CANVAS.check_required_tool_use).
resume_curr_round_result_ids = None

# --- Study time budget -----------------------------------------------------------
# invoke.py's objective text derives its "maximum of N hours" from this, so prose
# and the check_time tool cannot drift apart.
STUDY_BUDGET_HOURS = 1.5
STUDY_BUDGET_SECONDS = STUDY_BUDGET_HOURS * 3600
# Fraction of the budget at which check_time starts telling the agent to wrap up.
# NOTHING enforces the budget -- it is advisory prose only.
TIME_BUDGET_WARN_FRACTION = 0.8

# --- FEM solver ------------------------------------------------------------------
# Each coupled-mode analysis runs in a CHILD PROCESS (see src/fem_calculator.py):
# gmsh is a C singleton whose initialize() installs a SIGINT handler, which CPython
# forbids off the main thread -- and LangGraph runs tool calls on worker threads.
# A child also turns a solver crash or hang into one failed design instead of a
# dead study. Measured cost: ~4 s of interpreter + imports on top of a 3-25 s solve.
FEM_SOLVE_TIMEOUT_SECONDS = 120   # wall-clock limit per design; 0 disables the limit

# --- Run gate (pause / step-through control; see src/run_gate.py) -----------------
# status.txt is created with this mode. "num" starts the study in step-through
# mode: it runs RUN_GATE_INITIAL_STEPS gate crossings and then waits for you to
# append more to step_num.txt. Set "run" for an unattended run.
RUN_GATE_INITIAL_MODE = "num"
RUN_GATE_INITIAL_STEPS = "1"
RUN_GATE_POLL_SECONDS = 5        # how often a paused agent re-reads the files
RUN_GATE_HEARTBEAT_SECONDS = 15  # how often a paused agent reprints why it is waiting

# --- Design target ---------------------------------------------------------------
# The coupling length the study is aiming for, in um. EXPLOG derives
# coupling_length_deviation_um = |coupling_length_um - TARGET| on every result and
# invoke.py's objective text quotes the same number, so prose and metric cannot
# drift apart. Excess loss (-10 log10 max_power_transfer, dB) needs no target.
TARGET_COUPLING_LENGTH_UM = 20.0

# --- Disposition vocabulary ----------------------------------------------------
# The fixed Decision vocabulary for a candidate disposition (single-valued).
# Two TERMINAL tags (no further compute) bracket three ACTIVE priority levels:
#   Abandon         -> terminal: stop, unpromising
#   Low priority    -> active: keep going, low priority
#   Medium priority -> active: keep going, medium priority (neutral default)
#   High priority   -> active: keep going, high priority
#   Sufficient      -> terminal: characterized, no further compute warranted
# A FAILED candidate may ONLY be Abandon (enforced in the disposition tool layer).
DISPOSITION_DECISIONS = ("Abandon", "Low priority", "Medium priority", "High priority", "Sufficient")
DISPOSITION_TERMINAL_DECISIONS = ("Abandon", "Sufficient")
DISPOSITION_ACTIVE_DECISIONS = ("Low priority", "Medium priority", "High priority")  # ascending
DISPOSITION_DEFAULT_ACTIVE = "Medium priority"

# --- Dialogue history log rotation (hist/his_<N>.txt) --------------------------
# Once the active his_<N>.txt file reaches this many bytes, close it and start
# the next (see src/history_log.py).
HIST_ROTATE_BYTES = 1 * 1024**3  # ~1 GiB

# Runtime cache owned by src/history_log.py -- which his_<N>.txt is active and
# its current byte count. None until the first write_history() call in THIS
# process, which (re)computes both from disk. Do not set by hand.
hist_active_index = None
hist_active_bytes = None
