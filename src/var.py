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
reportName = ""

# HPC queue policy: if fewer than this many jobs are pending, tools recommend
# submitting more ready work so the queue does not drain (agents decide).
QUEUE_MIN_PENDING = 15

# --- Disposition gate ---------------------------------------------------------
# Runtime slot mirroring the CURRENT task's `myStep.enforce_queue_floor` (the
# authoritative per-task default lives there). worker_agent_node copies
# plan[0].enforce_queue_floor here each turn (re-derived on resume); Gate 2 in
# wait_for_update reads it via getattr(var, "enforce_queue_floor", True). This
# initial value is just the pre-first-turn fallback, NOT a config knob.
enforce_queue_floor = True

# The fixed Decision vocabulary for a candidate disposition. Three mutually
# exclusive lifecycle states (single-valued): a candidate is either being
# worked, deemed done-good, or dropped.
#   Abandon       -> stop, unpromising
#   Investigating -> actively working it; specifics live in Future_plan
#   Sufficient    -> characterized, no further compute warranted
DISPOSITION_DECISIONS = ("Abandon", "Investigating", "Sufficient")

# G(O) deviation (|G(O) - 2.46| eV) at/under which an O-adsorption site is
# "competitive" enough to warrant an OH job. Used by the forgotten-OH reminder
# and cited by requirement-13 in the worker prompt so the two stay consistent.
# Set negative to disable forgotten-OH detection entirely.
GO_DEV_OH_THRESHOLD = 0.3

# When the wait-tool's Gate-2 message lists more than this many forgotten jobs,
# there is plainly plenty of work to do, so it drops the "return to the
# supervisor / do a literature review" closer and just tells the worker to get on
# with it. At or below this count the closer is kept.
FORGOTTEN_CLOSER_SUPPRESS_ABOVE = 30

# Process ids that were already finalized before the first resume after the
# disposition-gate rollout. The wait-tool's Gate-1 coverage check treats these
# as exempt (the agent need not cite them to dispose a legacy candidate), but
# the agent is never blinded to them. Frozen snapshot -> lives in source so it
# survives resume. TODO(Step 7): fill from the production explog's finalized ids.
LEGACY_DISPOSITION_EXEMPT_IDS: set[int] = set()