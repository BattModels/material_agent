"""workflow_audit — tools for auditing DREAMS agentic-workflow runs.

Each audit is a small, self-contained module so more can be added over time
(token accounting, timing, queue utilization) without entangling them.

Current audits:
    history_parser : parse his.txt into a per-tool-call table
                     (round, agent, tool, calc_type, outcome, time).

Nothing in the workflow runtime imports this package; import it explicitly
when auditing:  from src.workflow_audit import parse_history
"""
from .history_parser import parse_history  # noqa: F401
