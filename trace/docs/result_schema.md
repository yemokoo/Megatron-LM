# Result schema

Each result JSON must contain:

- `schema_version`
- a unique `run_id`
- `provenance` with classification, exact code SHA, data manifest, model
  identity/checksum, and configuration identity/checksum
- ordered `tasks`
- `metric`
- square `score_matrix`

Row `i`, column `j` is the score on task `j` after training task `i`. Entries
above the diagonal must be `null`. The last row must be complete.

`compare_results.py` reports the last-row mean and, for each task except the
last, the paper's early-task forgetting:

`AFR(T_k) = mean_{i=k+1..K}(R_k(k) - R_k(i))`.

It then averages these per-task values. Metrics have different scales and
semantics across TRACE, so the comparison is valid only when task-specific
metric implementations match.
