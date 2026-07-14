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
All_Report_Names = []
tmp_report_names = []
# Per-turn read-once guard state. Maps canvas key -> number of times it has
# been read via read_my_canvas during the CURRENT supervisor/worker node
# turn. Reset to {} at the top of every node that streams an agent. Enforces
# the "read the same key exactly ONCE" rule and breaks degenerate re-read
# loops (see read_my_canvas).
READ_KEYS_THIS_TURN = {}