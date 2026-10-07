# HEAR

A LangGraph research-agent harness with vLLM and SparseEngine backends, FAISS/MCP
retrieval, and an offline role-configuration selector.

## Configurations

| Arm | Main | Reader |
|---|---|---|
| A | vLLM | vLLM |
| B | OmniKV | vLLM |
| C | vLLM | H2O |
| D | OmniKV | H2O |
| E | vLLM | SnapKV |
| F | OmniKV | SnapKV |

`configs/formal208/` contains the six configurations; `configs/engines/` contains
engine settings. `configs/cohort-208.ids` identifies the shared 208-question subset.
Datasets, model weights, retrieval indexes and experiment results are not bundled.

## Usage

Use Python 3.11+ for the harness. Install GPU backends separately.

```bash
python -m pip install -e '.[tokenizer,test]'
python -m bcgraph.cli --help
python -m pytest -q
```

With services already running, set `MAIN_URL`, `READER_URL`, `MCP_URL`, `MODEL_PATH`,
`MAIN_EPOCH` and `READER_EPOCH`, then run:

```bash
python -m bcgraph.cli run --config configs/formal208/C.yaml \
  --queries /path/to/queries.tsv \
  --ids-file configs/cohort-208.ids --output runs/C
```

`select-config` filters development profiles, ranks feasible configurations by
makespan, and freezes a verified choice for `run --selection`. Measurements and
quality/stability checks are supplied externally; it does not switch running
services. Examples in `examples/selector/` use synthetic data.

Core code is under `src/bcgraph/`, serving code under `sparseengine-src/`, and
retrieval code under `retriever/`. Upstream licenses are retained alongside their
code. No additional project-wide license is granted by this package.
