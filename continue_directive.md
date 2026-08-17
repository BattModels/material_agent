# Operator directive — delivered to the supervisor as BOSS FEEDBACK
#
# Read by `python invoke.py continue [path]`. The whole file (minus surrounding
# whitespace) becomes state["boss_feedback"], so the supervisor sees it as:
#
#     "Your previous draft final answer has been reviewed and rejected by the
#      boss and received the following feedback: <this file>"
#
# It is ONE-SHOT: the supervisor round that reads it also clears it, so it is
# seen exactly once. Anything that must persist across rounds belongs in the
# objective instead -- see
# continue_objective.md, which replaces state["inputs"] wholesale and IS
# re-rendered into every prompt, every round. Keep this file to: what just
# happened, the facts that correct the record, and what to do FIRST.
#
# This leading '#' block is STRIPPED before delivery (invoke.read_operator_message),
# so it costs the agents nothing and can stay. The one rule: do not begin the
# message body below with a '#' heading, or it is eaten with the header.
# EDIT BEFORE USE.

OPERATOR UPDATE — the run was down for four days on an infrastructure fault, not on anything
you did. Two things have changed while you were stopped.

1. THE TIME BUDGET IS NOW 200 DAYS, not 30. This study is a continuing programme that runs
until the compute grant is spent; there is no deadline and no final report to work toward.
Your CANVAS and plan text still say "Day 24 of 30", "5.8 days remaining" and "wind down at
Day 26.0". All of that is now wrong. Correct it in your notes NOW rather than at reporting
time -- a stale figure left in the record is inherited by every later round, which is how the
0.30 eV threshold survived two corrections.

2. SAMPLE EACH CANDIDATE AT MORE ADSORPTION SITES. Across the study a candidate averages 2.91
O sites tried. 139 candidates have found a competitive O site, but 87 of them found exactly
one and stopped there. The deepest candidate in the study reached 12 sites and 4 terminations,
so the room exists -- this is a habit, not a limit. Aim for roughly 4 to 8 O sites per
candidate, more where your results justify it, and relax a further termination when a surface
runs out of distinct promising sites. Treat that range as a guide: some candidates genuinely
do not offer that many sites, and those are finished, not under-sampled.

The most under-sampled are also among the best you have. Verify these against the log rather
than taking the list on faith:

  9e74301d57  PrBiRh2O7          1 site,  1 surface,  best G(O) deviation 0.010
  cf03e0cb1d  LuBi3Ir2(RhO7)2    1 site,  1 surface,  0.020
  47c41df315  YCoTeO6            2 sites, 1 surface,  0.030
  65c2439d36  Al3Co(TeO6)2       2 sites, 1 surface,  0.030
  a21b6febda  FeIrRhO6           3 sites, 1 surface,  0.030
  0dbc4e8975  PtRh(SeO3)4        2 sites, 1 surface,  0.040

You can find the rest yourself: query_explog on the candidates table, sort n_O_started
ascending, filter on G(O) deviation or idealOverPotential to take the promising ones first.

Also: about 354 calculations finished while the run was down and need dispositioning, and
there is one hard ceiling on submission -- do not take the queue above ~800 queued jobs,
because SLURM refuses submissions once you hold 1000 active and they will simply fail.
