# Metric definitions and implementation qualifications

## What has been checked

The offline verifier recomputes the 110 cache-table entries, six DeepResearchBench makespans, and six RACE means at manuscript precision. This is an archived-result consistency check, not a new GPU run. Source syntax and command generation are checked separately. Runtime CUDA compatibility, live retrieval, judge calls and full end-to-end performance have not been revalidated during packaging.

## SCBench

The reported batch is the 60-session, seed-2026, 60-second arrival-window batch (`analysis/timeline/n60win60`). It is not the later 100-session sweep. Initial arrival times are sorted uniform samples conditioned on 60 arrivals in the window. Think times are normal with mean 5 seconds and standard deviation 1 second, lower-clipped at 0.5 seconds. Later requests arrive after the preceding response completes plus the fixed think time, so absolute later-arrival times depend on the policy.

TTFT is `max(decode_start, arrival) - arrival`; quantiles use sorted index `min(n-1, int(n*p))`. Completion is the largest recorded completion timestamp on the rebased replay clock. Reuse is cached tokens divided by prompt tokens on follow-up turns only.

The archived driver enables both waiting protection and queued-prefix L2 keepalive for Guard arms. The `kvaware` implementation also contains a cold-request admission hold (alpha 0.85, maximum 200 scheduling rounds). The Guard branch dispatches directly to the engine. These are not pure single-toggle scheduling ablations. The extra `fcfs-ka` record is an auxiliary keepalive control, not a main-table row.

Cache-project source snapshots were available without an immutable experiment-time repository revision. They are included with content hashes; an exact historical source/driver/environment reconstruction is not claimed.

## Mooncake

The trace window is [600,1500] seconds. Of 1954 candidate sessions, 107 exceeding the context limit are removed; the resulting 1847 sessions define 100% load. Sessions are truncated at eight turns. Materialized workloads preserve first-arrival times, request lengths and remapped block-sharing structure. Token content is synthesized per block hash.

Replay is closed-loop. Think time is `min(300, max(1, raw_gap - (2 + output_tokens/30)))`; the next request arrives after this workload's own previous response completes plus that think time. This is not an open-loop replay of every original request timestamp or every original inter-turn gap.

Table request latencies use arrivals in [180,900] seconds. Throughput counts completions in that interval and divides by 12 minutes. Session time is the mean sum of per-turn arrival-to-completion durations for sessions whose first arrivals fall inside that window; it excludes think gaps. Reuse uses follow-up prompt tokens in the measurement window. Runs drain after the window so eligible arrivals have complete latencies.

The five tags are `baseline`, `kvaware-nh`, `W40`, `retainU`, `W40-retain`. `kvaware-nh` disables harness holding. Session-aware runs use a fixed depth table `{1: .31, 2: .42, 3: .55, 4: .56, >=5: .81}`, calibrated from the earlier trace segment, and retention value `p * exp(-idle/400)`. No online updating of p from current-run completed conversations is implemented by this replay path. The paper's retained runs disable the optional queue-length retention gate. The engine also keeps queued/running prefixes recent when keepalive is enabled.

## DeepResearchBench

Six arms each attempt 100 instances. The archived first block comes from two earlier runs with the same intended configuration; subsequent instances come from the full batch experiment. The packaged portable runner executes fresh blocks and is not the historical retry/repair controller.

Makespan sums ten retained blocks' `max(finished_at)-min(started_at)` over COMPLETE instances. It includes successful retained instances' internal retries, but does not sum all historical failed-block attempts, engine-start time or inter-block idle time. One final dense-dense instance is failed. Historical whole-block retry cost is not recovered by this metric.

The Main default output cap is 4096; reviewer requests override it to 8192. Researcher cap is 512. H2O/SnapKV use chain-delta continuation; dense uses full transcripts. The SnapKV gate limits whole-prefill reservations, separately from its 4672-token retention budget. Original role latency/TTFT fields differ in their reasoning-stream visibility; they should not be interpreted as directly comparable prefill timings.

RACE means are computed from available judge outputs (97/98/96/99/99/99 by arm order in `per_arm.csv`). Completion, empty-report and judged categories are not mutually exclusive. In particular, the dense-dense failed instance also has a recorded judge score; the association may come from a different attempt and should be audited before treating coverage counts as disjoint. The common judged cohort is 89. The original paired bootstrap implementation, seed and exact resampling procedure are not included, so CI/p-value reproduction is not claimed.

The six explicit mode configurations are implemented. A separately preserved development-stage calibration/ranking program and pre-evaluation serialized selector freeze were not found in the packaged evidence. Selecting the minimum of the final six-arm outcomes is not evidence of pre-evaluation selection. No synthetic historical freeze artifact is supplied.

## Packaging changes

- Personal tokenizer/credential path defaults are replaced with generic local paths.
- Templates omit the original local proxy default and accept explicit deployment paths/endpoints through `prepare_run.py`.
- Algorithm code is retained; portability wrappers and offline verification are clearly separated from source snapshots.
- Raw web documents, complete DRB event traces, model weights, original machine logs, unrelated sweeps, shell history and author repository metadata are excluded.
- Existing third-party licenses and attribution remain in place.
- Runtime dependency metadata is supplied, but complete original Conda/CUDA/driver lockfiles are unavailable.

## BrowseComp-Plus integration

BrowseComp-Plus is packaged under `browsecomp/` with its own dependency lock,
retriever license, six formal configurations and a separate engine snapshot.
No original files in that component were overwritten or rewritten during the
merge. No GPU run or result aggregation for BrowseComp-Plus was performed:
the supplied component does not contain the formal results. Its selector
examples are explicitly synthetic, not historical development-set evidence.
Source syntax, file preservation, cohort size and example integrity were checked.
The BrowseComp pytest suite requires its declared environment; it is not covered
by the original three-benchmark offline verification.
