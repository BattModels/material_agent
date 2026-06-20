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