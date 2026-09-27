# ESCI Hybrid Product Search

A reproducible product-search system that combines **BM25 keyword retrieval**, **MiniLM dense retrieval**, **Reciprocal Rank Fusion (RRF)**, and **LightGBM LambdaRank**. It includes data contracts, dataset auditing, reproducible release manifests, offline evaluation, a FastAPI service, a browser demo, and a local learning workbench.

> This repository is an educational and portfolio implementation based on the public [Amazon Science ESCI dataset](https://github.com/amazon-science/esci-data). It does not use Amazon internal data, APIs, or online business metrics.

## What this project demonstrates

- Query/product/judgment data contracts with explicit locale and identity boundaries
- Streaming preparation and audit into SQLite-backed catalog and label stores
- BM25 inverted index persisted in SQLite
- Local `sentence-transformers/all-MiniLM-L6-v2` embeddings with exact blockwise vector search
- RRF fusion without incorrectly adding scores from different retrieval models
- Learning-to-rank with LightGBM LambdaRank and query-group validation
- NDCG, MRR, known-label recall, judged coverage, bootstrap comparisons, and error analysis
- SHA256-checked release packages and runtime signatures to reject incompatible artifacts
- FastAPI lifecycle management, bounded request admission, request IDs, and sanitized logs
- A double-click Tkinter learning workbench for search, reports, and step-by-step experiments

## Architecture

```text
ESCI Parquet
    │
    ▼
normalize → split by query → audit → catalog.sqlite + labels.sqlite
    │
    ├── BM25 SQLite inverted index
    └── MiniLM embeddings.npy + exact vector index
             │
query ───────┴──► BM25 + dense ─► RRF ─► LambdaRank ─► catalog lookup ─► API/page
                                      │
                                      └──────────────► offline evaluation reports
```

## Quick start on Windows CPU

The repository includes a small prepared `dev` dataset, two prebuilt search releases, and validation reports. The 1 GB raw Parquet files are intentionally excluded; their pinned source URLs and SHA256 manifests remain under `data/raw/`.

```powershell
py -3.12 -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements-lock.txt
& .venv/Scripts/python.exe -m pip install -e . --no-deps
```

The embedding weights are not committed. On the first machine, download/cache the pinned MiniLM revision once:

```powershell
& .venv/Scripts/python.exe -c "from product_search.settings import load_settings; from product_search.retrieval.embedder import Embedder; Embedder(load_settings('configs/dev.toml', project_root='.'))"
```

Then double-click `打开项目.cmd`. The local workbench has four pages:

1. **体验搜索** — launch the existing release and inspect BM25/dense sources;
2. **结果解读** — read the measured validation report;
3. **重做实验** — build indexes, train LambdaRank, and evaluate validation in order;
4. **学习路线** — follow the source-code reading order.

You can also run the test suite:

```powershell
& .venv/Scripts/python.exe -m pytest -q
```

For a terminal-first reproduction, HTTP examples, test-freeze rules, and troubleshooting, read [`docs/COMMANDS.md`](docs/COMMANDS.md). For the learning flow, read [`docs/USAGE.md`](docs/USAGE.md).

## Included reproducibility snapshot

| Item | Included content |
|---|---|
| Prepared data | `data/processed/dev-c882caf5632ad442` |
| BM25 + dense release | `artifacts/dev-indexes-20260926` |
| BM25 + dense + LambdaRank release | `artifacts/dev-ltr-20260926` |
| Validation evidence | `reports/validation-20260926` |
| Local benchmark | `reports/benchmark-20260926.json` |
| Data source manifests | `data/raw/*.manifest.json` |

The snapshot contains 11,934 products and 200/200/200 train/validation/test query groups in the prepared dev workflow. The existing validation report uses 200 queries. Test data is kept for the project’s explicit freeze-and-test workflow; the published report is validation, not an independent test claim.

## Measured result

On the included dev validation report:

| Method | NDCG@10 | MRR@10 |
|---|---:|---:|
| BM25 | 0.7682 | 0.7389 |
| Dense | 0.7829 | 0.7798 |
| RRF | 0.7777 | 0.7541 |
| LambdaRank | 0.7735 | 0.7406 |

Known E recall@100 is 0.8047 for BM25, 0.8636 for dense, and 0.8750 for RRF. The current LambdaRank model did not exceed dense on this development validation. These are offline public-dataset measurements, not online conversion or revenue metrics.

## Project map

- `src/product_search/schemas.py` — domain records and data boundaries
- `src/product_search/data/` — read, normalize, split, prepare, audit, and SQLite stores
- `src/product_search/retrieval/` — text processing, BM25, MiniLM, exact vectors, RRF, optional FAISS experiment
- `src/product_search/ranking/` — features, grouped training, model loading, and prediction
- `src/product_search/evaluation/` — metrics, comparisons, error analysis, freeze, and benchmark
- `src/product_search/service/` — SearchEngine, FastAPI, observability, and static page
- `src/product_search/workbench.py` / `desktop.py` — local learning workbench and process orchestration
- `tests/` — unit, integration, artifact, ranking, API, evaluation, and workbench tests
- `docs/FILE-GUIDE.md` — chronological file-building guide
- `docs/interview.md` — interview questions and project-specific answers

## Scope and limitations

This is a CPU-friendly, reproducible portfolio project. It is not a claim of production scale. The default vector search is exact and blockwise; FAISS HNSW is kept as an independent comparison experiment. The project does not include online click logs, distributed ingestion, Milvus, Kafka, Spark, Hive, or GPU-only training.

## Attribution

The data source is the public [amazon-science/esci-data](https://github.com/amazon-science/esci-data) repository. See [`NOTICE.md`](NOTICE.md) for the attribution and source revision used by this snapshot.
