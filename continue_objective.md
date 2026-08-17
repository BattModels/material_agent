# REPLACEMENT USER PROMPT (state["inputs"]) for a resumed run
#
# This is the OBJECTIVE -- the user's own request -- not feedback about it.
# Write it in the user's voice ("Please conduct ..."), because that is how the
# agents see it: "The overall goal is: <this file>", in EVERY supervisor prompt
# and EVERY boss prompt, for the rest of the run.
#
# Contrast with continue_directive.md, which is delivered ONCE as boss_feedback
# ("your draft was reviewed and rejected ... feedback: ...") and is consumed by
# the supervisor round that reads it. Standing policy belongs HERE; "what just
# happened and what to do first" belongs THERE.
#
# Read by `python invoke.py continue`, which writes it into state["inputs"] via
# the same update_state call that injects the directive. OPTIONAL: delete or
# rename this file and the run resumes with its original objective untouched.
#
# WHY IT IS NEEDED AT ALL: on a resumed run state["inputs"] comes from the
# CHECKPOINT (invoke.py sets inputs = None), so editing the objective in
# invoke.py has no effect whatsoever. This file is the only way to change it.
#
# This leading '#' block is STRIPPED before delivery (invoke.read_operator_message),
# so it costs the agents nothing and can stay. The one rule: do not begin the
# message body below with a '#' heading, or it is eaten with the header.

Please conduct an acidic OER screening study to identify the best catalytic candidates
for the oxygen evolution reaction (OER) in the Google DeepMind GNoME database.
Please do an iterative multi-round screening, learning from each round and applying
insights to new candidates, surfaces/terminations, or active sites when sensible.
Please use literature searches to inform your per-round candidate selection and
hypothesis formation and note them down clearly.

Scope and coverage:
BROAD COVERAGE OF THE DATABASE IS A PRIMARY GOAL OF THIS STUDY, alongside depth on the
most promising systems. The filtered candidate pool is far larger than this study can
exhaust, so examining a large number of candidates -- prioritised by relevance rather
than swept indiscriminately -- is itself a deliverable, not a means to one. Keep widening
the study for as long as budget remains: revisit the database repeatedly with selection
criteria refined by what you have learned, and register new candidates in substantial
batches rather than a few at a time.

Running out of ready continuation work NEVER means the study is finished; it means it is
time to add more candidates. Do not wind the study down, and do not propose a final
answer, until the time budget is genuinely spent -- specifically, until too little time
remains for newly submitted DFT jobs to finish. Concluding early with cluster capacity
and budget unused is a failure of the study, not an efficient completion of it. Balance
this against quality: every candidate you register should be justifiable from your
findings and the literature, and breadth must never become padding.

Prioritize O adsorption calculations broadly across many candidates, focusing on
hypothesis-relevant unique sites instead of exhaustively evaluating all sites for each
candidate (it may be relevant to consider many adsorption sites for a few candidates).
Use the resulting G(O) values to identify the most promising candidates and sites before
proceeding with OH adsorption calculations (possibly delaying OH calculations to later
rounds). When evaluating overpotentials and ranking candidates, you will need to consider
both the overpotential calculated assuming an ideal OOH binding and the one calculated
via the scaling relation.

Supply risk and earth abundance -- RAISED PRIORITY:
A significant share of every remaining batch of candidate registrations must target LOW
SUPPLY-RISK materials, judged on BOTH of the available production-concentration metrics,
because they measure different things and a candidate is only genuinely low-risk when both
are low:

- average_HHI_P_excluding_O_H is ATOM-WEIGHTED across the material's substantive elements.
  A low value means any scarce metal present is DILUTE in the framework.
- max_HHI_P is the worst single element in the material, regardless of how little of it
  there is. A low value means no supply-constrained element is present at all.

Pursue two tracks in parallel, and say which track a candidate belongs to when you
register it:

TRACK A -- LEAN PRECIOUS-METAL CATALYSTS (the larger share). Target a low
average_HHI_P_excluding_O_H while allowing max_HHI_P to stay high: frameworks where Ir,
Rh, Ru or Pt is present but DILUTE, carried in an abundant host lattice. The aim is to
keep whatever active-site chemistry the study has shown to work while minimising how much
scarce metal each formula unit needs. When a precious-metal site performs well, the
immediate follow-up question is how far its concentration can be reduced before the
activity goes with it -- treat that as a hypothesis to test, not an afterthought.

TRACK B -- PRECIOUS-METAL-FREE LONG SHOTS (a smaller but real share). Target low values on
BOTH metrics, max_HHI_P included, so that no supply-constrained element appears anywhere in
the composition. Expect a lower hit rate than Track A. Pursue them anyway: a fully
earth-abundant catalyst that is active and stable under acidic OER conditions is a
transformative result rather than an incremental one, and that asymmetry justifies
spending real capacity on candidates that will often fail. Do NOT abandon a Track B line
merely because its first results are unremarkable, and do not let a run of poor Track B
results become a reason to stop registering them -- judge each candidate on the same
G(O)-deviation evidence you apply to everything else.

Neither track replaces the other, and neither replaces ordinary breadth: continue to
register candidates that are simply the most promising on activity grounds. Report both
HHI figures alongside both overpotentials whenever you rank or compare candidates, so the
cost dimension stays visible in the final analysis rather than being reconstructed at the
end.

The AQ-GNoME database is available for stability-based filtering. The Pourbaix stability
screening is fixed at pH = 0 and U = 1.2-2.0 V vs. SHE (acidic OER operating conditions).
Many additional filters are available (e.g. decomposition threshold, bandgap, HHI,
disorder probability) and literature should inform the selection of these criteria and
candidate choices. You should also consider catalytic activity, cost/availability, and
stability under operating conditions when selecting and evaluating candidates. Toxicity
of the constituent elements should also be considered where possible, though note that no
toxicity data is available in the dataset, hence this assessment will be limited to
qualitative reasoning based on literature. Revisit the AQ-GNoME database repeatedly
during the study using refined selection criteria based on emerging insights, to explore
new candidates.

HPC usage:
Keep the cluster genuinely fed. Have relevant DFT jobs pending/queued at all times, and
do not wait for the jobs of one round to finish before submitting the next. Submit in
LARGE BATCHES rather than a trickle: whenever the cluster has spare capacity, submissions
are effectively free, and bulk relaxations in particular are cheap and are the natural way
to buy breadth. How much spare capacity exists changes over the study, so judge it from
the live running/queued counts rather than from any standing assumption. If the queue is
running low, the first remedy is to register more candidates
and submit their bulk relaxations; the second is more terminations and adsorption sites
on the candidates that have earned them. Note that surface and adsorption jobs take much
longer than bulk jobs, so start them as soon as their prerequisites land. Only when the
remaining time is too short for a newly submitted job to finish should you stop
submitting and move to finalizing the in-flight results.

Report:
At the end of EACH ROUND as well as the end of the study, produce an extensive report
structured as a mini scientific paper. Every conclusion and claim must be directly
supported by concrete results from the study — cite specific candidates, sites,
terminations, G(O), G(OH), and overpotential values explicitly. Be critical of your
conclusions and assumptions: acknowledge limitations, uncertainties, and cases where the
data is inconclusive. Do not make claims that are not backed by data. The report should
include:

- A summary of the screening strategy and how it evolved.
- The best candidates identified, with their G(O), G(OH), ideal overpotential, and
scaling-relation overpotential.
- A comparison of the best candidates with available literature.
- What was learned, what worked, and what did not.
- Which hypotheses were confirmed or rejected, with explicit reference to the
supporting data.
- Any trends worth noting across the dataset, even if these trends do not lead to
competitive candidates.
- How many candidates were examined and at what depth, and what the coverage achieved
implies about the parts of the pool left unexplored.
- Any recommendations for future studies or next steps based on the findings and
limitations of the current study.

You have a budget of 200 days for this study. It is a continuing programme, not a task with
a deadline: keep screening, deepening and reporting for as long as the budget lasts. There
is no final report to work toward -- produce the round reports described above as you go,
and treat each one as the current state of the work rather than a conclusion.
