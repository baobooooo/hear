# HEAR: experiment code for anonymous review

This repository contains the **BrowseComp-Plus, DeepResearchBench, SCBench, and Mooncake** experiments associated with the submission.

Source snapshots are included for all four benchmarks. Recorded results and numerical verification are included for DeepResearchBench, SCBench, and Mooncake; BrowseComp-Plus results are not bundled in the supplied component. Portable command preparation and block execution utilities were added for artifact packaging; they are not represented as the original experiment-time drivers. No new GPU results were generated while preparing this artifact.

## Quick start: verify the reported numbers without GPUs

From the repository root, using Python 3.12:

```sh
python verify_paper_results.py
```

No network access, credentials, model downloads, or third-party Python packages are required. The command re-creates `verification.json` and checks:

- all 20 values in the four-row SCBench table;
- all 90 values in the fifteen-row Mooncake table;
- six DeepResearchBench makespans and six RACE means;
- 600 DeepResearchBench instance records, 588 judged outputs, and the common 89-instance quality cohort.

The archived values match at the precision reported in the paper. `mismatches` must be empty. Cache metrics are recomputed from per-request records. DeepResearchBench makespans are recomputed from per-instance CSV timestamps; RACE means are recomputed from the per-instance score CSV. Full DeepResearchBench event traces and retrieved web documents are not part of this repository.

Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for metric definitions and implementation details that qualify the manuscript descriptions.

## Repository map

| Location | Purpose |
|---|---|
| `browsecomp/` | BrowseComp-Plus harness, six formal configurations, 208-question IDs, retrieval service, separate SparseEngine source and tests |
| `deepresearchbench/harness/src/mtbench/` | LangGraph workflow, retrieval, role-specific requests, chain continuation, FIFO prefill gate |
| `deepresearchbench/sparseengine/src/` | Sparse inference engine used for OmniKV, H2O and SnapKV |
| `deepresearchbench/harness/vendor/DeepResearchBench/` | Upstream query set, criteria, prompts and RACE evaluator; upstream attribution retained |
| `configs/deepresearchbench/` | Full-transcript and chain-delta workflow templates |
| `configs/engines/` | Actual sparse engine parameters for each role and arm |
| `deepresearchbench/evidence/metrics/` | Recorded instance-level and arm-level metrics and quality scores |
| `cache_coordination/server_pro6000/` | SCBench harness and vLLM scheduler/offloading patches |
| `cache_coordination/server_h100/` | Mooncake harness, workload construction and patches |
| `cache_coordination/inputs_pro6000/` | SCBench instance pool and recorded generation lengths/tokens |
| `cache_coordination/inputs_h100/workloads/` | Materialized Mooncake workloads for the three loads |
| `cache_coordination/analysis/` | Recorded per-request results and the paper's plotting scripts |
| `scripts/prepare_run.py` | Generates explicit launch/replay commands and a new configuration directory; does not execute them |
| `scripts/run_drb_block.py` | Runs one ten-instance block against already-running isolated services |
| `CHECKSUMS.json` | SHA-256 inventory of the packaged files |

## GPU reproduction prerequisites

GPU execution targets Linux; the offline verifier also runs on Windows. Original experiments used Qwen3-8B on one RTX PRO 6000 for SCBench, Qwen3-8B on one H100 for Mooncake, and GLM-4.7-Flash FP8 on two H100s for DeepResearchBench.

Cache coordination uses vLLM 0.26.0 and a 96-GiB CPU KV tier. Allocate sufficient host RAM for the tier and runtime (the original Mooncake worker required at least 150 GiB available). Keep each arm's engine isolated and freshly initialized. Patches use vLLM internal interfaces and are not claimed compatible with other versions.

Use separate Conda environments for dense/cache serving, the client, and SparseEngine when dependencies conflict. The supplied package requirements describe dependencies, not a complete original CUDA/driver environment lockfile:

```sh
conda create -n hear-cache python=3.12 -y
conda run -n hear-cache python -m pip install -r requirements-cache.txt
conda create -n hear-client python=3.12 -y
conda run -n hear-client python -m pip install -e deepresearchbench/harness
# Install SparseEngine in a separate compatible GPU environment using its pyproject.toml.
# Its CUDA/Torch/kernel provider dependencies must match the local H100 runtime.
```

Activate the relevant environment before launching services so its binaries and libraries are visible. If `--python`, `--sparse-python`, or `--client-python` point to another environment, ensure its full runtime environment is activated for the corresponding generated command. A bare interpreter path is not a substitute for CUDA library setup.

Weights, search API keys, and paid judge credentials are not included. Supply a local model directory and a private Serper key file. Use the exact model tokenizer for DeepResearchBench; the packaged block launcher checks that it is readable before issuing requests. Live web retrieval and paid API judging can vary over time.

## Prepare a cache-coordination run

From the root of a checkout on the GPU machine:

```sh
python scripts/prepare_run.py scbench --policy guard40 \
  --model /path/to/Qwen3-8B --gpu 0 --output runs/scbench-guard40

python scripts/prepare_run.py mooncake --policy session --load 0.75 \
  --model /path/to/Qwen3-8B --gpu 3 --output runs/mooncake-075-session
```

The commands create `commands.json` and print two `nohup` commands. Inspect them, select an idle GPU and unused ports, start the engine, wait for its `/health` endpoint, then start the client. The generator never allocates a GPU, stops a process, or starts a server. The cache client performs a cache reset: **only point it at the dedicated fresh engine for this run**. Stop only services owned by your experiment before the next arm.

SCBench policies: `fcfs`, `cache`, `guard40`, `guard60`. Mooncake policies: `fcfs`, `cache`, `guard40`, `session`, `combined`; loads: `0.5`, `0.75`, `1.0`. SCBench uses 60 sessions, seed 2026 and a 60-second initial-arrival window. The Mooncake `cache` arm disables harness holding. Session-aware paper runs disable the optional queue-length retention gate.

## Prepare a DeepResearchBench block

```sh
python scripts/prepare_run.py deepresearchbench --arm omnikv-h2o \
  --model /path/to/GLM-4.7-Flash-FP8 --gpu 3 --researcher-gpu 4 \
  --port 23907 --researcher-port 23908 \
  --keys-file /path/to/private/serper.keys --output runs/drb-omnikv-h2o
```

The generated commands start two role services and run instances 1–10. Before launching the block, inspect the sparse service logs: H2O/SnapKV must select `sgl_fa3_sm90`, and OmniKV must select `tilelang_score`; `triton_mla` fallback is not the reported execution path. Kernel eligibility depends on the actual GPU environment and is not verified by the command generator.

Run each of the six arms (`dense-dense`, `omnikv-dense`, `dense-h2o`, `omnikv-h2o`, `dense-snapkv`, `omnikv-snapkv`) in blocks starting at 1, 11, ..., 91. Restart the owned role services between every arm/block pair so that KV and chain state are not shared. For later blocks, use `scripts/run_drb_block.py --start 11 ...` with a new output directory and freshly initialized services. Keep ten workflow instances concurrent, with four research agents per instance. Failed blocks must be accounted for separately rather than silently dropped.

The generator retains the role-specific parameter values. Main reviewer requests have an 8192-token cap; the writer uses 4096. Researcher requests use 512. SnapKV uses the shared file-lock prefill gate at 456837 tokens, released at the first streamed token. No automatic service teardown, historical repair loop, or GPU-holding helper is included.

## Figures

With matplotlib installed:

```sh
cd cache_coordination/analysis/timeline/h100mc
python ../../plot_cache_ref.py
python ../../plot_kvload_ref.py
cd ..
python ../plot_sessions_cmp.py scbench
```

Outputs are written under `cache_coordination/analysis/paper/figures/`. Historical output names `fig-kvload` and `fig-sessions-scbench` correspond to `mooncake_load` and `scbench_timeline` in the manuscript. Font availability can change rendering; byte-identical PDF output is not claimed.

## Quality evaluation and limits

The archived scores use the official reference-free RACE evaluator with `gpt-5.4-mini`; failed provider calls were retried at concurrency four. The upstream evaluator and criteria are included; its README describes its API configuration and input format. A new judge run requires externally supplied credentials and can incur cost. The historical provider-specific wrapper is not a portable evaluation entrypoint in this artifact.

For DeepResearchBench, the original bootstrap/t-test script and independent development-stage selector-freeze evidence have not been recovered in the material packaged here. This repository does not invent either procedure. The source does implement the six candidate execution configurations, their role assignment, and the reported cache policies. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact supported claims.

## Attribution and packaging

Third-party license notices and public upstream references are retained. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No author-identifying Git history, personal deployment paths, private hosts, conversation IDs, credentials, or original experiment-machine logs are included. Keep generated run directories and secret files outside the published snapshot. This artifact has not been uploaded by its preparation utilities.

## BrowseComp-Plus component

Start with [browsecomp/README.md](browsecomp/README.md). Keep its working directory and environment separate from DeepResearchBench:

```sh
cd browsecomp
# Activate a Python 3.11+ Conda environment for this component.
python -m pip install -r requirements.lock.txt
python -m pip install -e '.[tokenizer,test]'
python -m bcgraph.cli --help
python -m pytest -q
```

Configure running inference/retrieval services through `MAIN_URL`, `READER_URL`,
`MCP_URL`, `MODEL_PATH`, `MAIN_EPOCH`, and `READER_EPOCH`, then use:

```sh
python -m bcgraph.cli run --config configs/formal208/C.yaml \
  --queries /path/to/queries.tsv --ids-file configs/cohort-208.ids --output runs/C
```

A/B/C/D/E/F respectively map to vLLM-vLLM, OmniKV-vLLM, vLLM-H2O,
OmniKV-H2O, vLLM-SnapKV, and OmniKV-SnapKV. The supplied cohort contains
208 unique IDs. Corpus, queries, retrieval index, model weights and experiment
results must be supplied externally. Its `sparseengine-src/` is a distinct
benchmark-specific snapshot; do not replace it with the DeepResearchBench engine
or add both engine roots to one `PYTHONPATH`. There are 46 differing/new files
relative to the DeepResearchBench source among the incoming engine files.

The BrowseComp selector implementation and tests are included, but
`examples/selector/` contains **synthetic** profiles and verification examples.
Their presence does not establish that the formal paper configuration was frozen
before evaluation. Keep that distinction when citing the artifact. The root
`verify_paper_results.py` verifies only the other three benchmarks; it does not
claim numerical verification of BrowseComp-Plus results.

See [MERGE_NOTES.md](MERGE_NOTES.md) for integration and validation scope.
