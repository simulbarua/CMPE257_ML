# Team task list

Each member owns one model. ADRec has multiple training stages, so data coordination and evaluation coordination are assigned to the baseline owners. Reassess workload weekly based on actual effort.

Model ownership: Simul — ADRec; Anusha — SASRec+; Mahalakshmi — GRU4Rec. Mahalakshmi leads the web interface, with model integration and testing shared by the team.

Completed experiments include five-seed Baby runs for all three models, five-seed ADRec runs on Beauty and MovieLens-100K, and the Baby diffusion-step comparison. See [results](../results/README.md). The tasks below define responsibilities across the project; they are not all pending work.

| ID | Owner | Task and completion evidence | Depends on |
|---|---|---|---|
| T01 | Simul | Verify upstream license, record ADRec commit, document setup and the source of each baseline; complete a small ADRec run | — |
| T02 | Anusha | Check the released datasets and loader; document counts, ID/padding rules, split semantics, and checksums | — |
| T03 | Mahalakshmi | Write shared evaluation protocol: targets, candidate filtering, metrics, seeds, and timing conditions; check agreement with paper/code and document discrepancies | T02 |
| T04 | Simul | Run ADRec stages; check loaded/frozen/trainable weights, save configuration and checkpoint, report reproducible results | T01–T03 |
| T05 | Anusha | Verify SASRec+ implementation source; train on agreed data/protocol; save configuration, checkpoint location and results | T01–T03 |
| T06 | Mahalakshmi | Run GRU4Rec with agreed data/protocol; save configuration, checkpoint location and results | T01–T03 |
| T07 | All; Mahalakshmi coordinates | Run shared evaluation, repeat agreed seeds, record variability and hardware, produce comparison table; each member validates their own model | T04–T06 |
| T08 | All | Agree prediction interface; Simul loads ADRec and ranks items, Anusha supplies valid example histories, Mahalakshmi builds UI; jointly test latency and invalid inputs | T04, T02 |
| T09 | All | Integrate and test demo; verify examples contain only history (no held-out target), label IDs clearly, document how to run app | T08 |
| T10 | All | Write model-specific report sections; jointly write analysis, limitations and slides; each member reviews another model's section | T07, T09 |
| T11 | Optional; all | Evaluate 3/5/10/full history on the same eligible users and targets; report this as reduced evaluation context, not retraining or cold-start performance | T07 |
| T12 | Optional; all | Add GRU4Rec and SASRec+ to demo using same input and candidate rules | T09, T05, T06 |

## Milestones

1. **Setup:** T01–T03 and a small run for each model. Freeze data/evaluation agreement.
2. **Experiments:** T04–T07. Use validation for tuning; reserve test results for final evaluation.
3. **Demo and submission:** T08–T10. Optional tasks only after core deliverables work.

## Review rotation

Anusha reviews Simul, Mahalakshmi reviews Anusha, and Simul reviews Mahalakshmi. Shared components require another member's review. Reassign work when needed rather than leaving a member blocked.

Track task progress in the [GitHub task tracker](GITHUB_TASKS.md).
