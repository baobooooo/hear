# Four-benchmark artifact integration

The BrowseComp-Plus component is retained byte-for-byte under `browsecomp/`.
This namespacing preserves its relative configuration paths, selector-example
hashes, package layout and test fixtures. It also avoids mixing its SparseEngine
snapshot with the distinct DeepResearchBench snapshot.

The repository root documentation, validation summary and checksum inventory
are updated. Existing DeepResearchBench, SCBench and Mooncake algorithms,
recorded data and verification formulas are unchanged.

Validation performed during merge:
- Python AST syntax checks across the merged artifact.
- Byte-for-byte comparison of every incoming BrowseComp file against the supplied component.
- 208 unique cohort IDs, six arm configurations, and synthetic selector evidence/config hashes.
- Re-execution of the original offline verification for the other three benchmarks.
- Scan for known private deployment paths, hosts, conversation identifiers and credential patterns.
- ZIP CRC and delivered-file checksum checks.

The BrowseComp pytest suite was not executed: available local Conda environments
lack its required Pydantic v2/LangGraph/MCP/HTTPX test environment. Installation
and test commands are in its README. No dependency versions or tests were changed
to hide this limitation. No inference services, retrieval services or paid judge
calls were started. BrowseComp result reproduction is unverified because its
formal outputs are not in the input component.
