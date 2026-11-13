# Shared evaluation protocol

Implementation and usage: [shared-code guide](../docs/SHARED_CODE_GUIDE.md).

- Data: pinned Baby pickle checksum in `configs/shared.json`; retain aligned per-user holdouts.
- Validation history: train. Test history: train + val. Targets never enter the inference callback.
- Inputs: latest 50 items, left zero-padding; original positive product IDs.
- Outputs: score columns match item IDs, including column 0 for padding.
- Candidates: all real product IDs, excluding padding; no seen-item filtering.
- Metrics: HR and NDCG at 5, 10, 20, averaged per user, returned as fractions.
- Ties: ascending item ID. Reject invalid score shapes and nonfinite candidate scores.
- Selection: tune/select with validation only. Keep test results for final reporting.
- Record seed, software, hardware, checkpoint source, configuration and data checksum.

Known differences: inspected upstream trainer averages batch means. Shared evaluator weights users equally, explicitly excludes padding and resolves ties. Run and label untouched upstream evaluation separately when checking published numbers. The paper describes an 8:1:1 user split, but the released loader/file uses the same users with per-user item holdouts. Follow the released data here and disclose that discrepancy.

Training target strategy remains model-specific (`per_token` or `last`); use the same underlying histories. Optional shortened-history evaluation is not implemented yet. Timing benchmarks are not implemented yet and need a shared hardware/warm-up policy.
