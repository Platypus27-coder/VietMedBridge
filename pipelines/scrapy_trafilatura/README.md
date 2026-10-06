# ViBioMIR ingestion

Pipeline P0: source Parquet → inventory/mappings → Scrapy fixed-frontier download → raw bytes + event manifests → offline Trafilatura/PyMuPDF → conservative cleaning/LID/quality → exact dedup → documents + BTC ID mappings.

Python **3.11** is the development baseline requested in the implementation instructions. Scrapy 2.13.4 and Trafilatura 2.1.0 are pinned. Windows uses spawn workers and a separate process per crawl; Linux/WSL and Kaggle use the same modules. The only configuration source is `configs/config.yaml`, including deployment fields.

## Repository

```text
configs/config.yaml
src/
  inventory/       snapshot, normalization, disk-backed inventory, sample/shards
  crawler/         spider, middleware, scheduler, raw store, subprocess runner
  extraction/      HTML/PDF/text adapters, supervised spawn workers
  preprocessing/  cleaning, heuristic LID, quality, exact dedup
  storage/        atomic I/O, schemas, JSONL compaction, SQLite indexes
  reports/        gold coverage, domain estimates, manual review
  utils/          config, paths, environment detection
  cli.py
  validation.py
scripts/           setup, download, individual stages, Stage A, validation, packaging
notebooks/01_kaggle_stage_a.ipynb + .py
tests/             unit tests and explicit loopback integration harness
data/              source/snapshots, inventory, raw, manifests, processed, mappings
crawl_jobs/        isolated resumable jobs
checkpoints/       ordering, cooldowns, orchestration state
reports/           machine report, blank manual-review CSV and integrity report
requirements.txt / requirements-kaggle.txt / requirements-lock.txt
```

## Windows PowerShell

Select an installed Python 3.11. `setup_env.ps1` also detects the workspace-managed interpreter when present:

```powershell
.\scripts\setup_env.ps1 -Python 'path\to\python3.11.exe'
.\.venv\Scripts\Activate.ps1
python --version
python -m pip --version
python -m pytest
```

If ExecutionPolicy blocks a script, use only the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activation is optional: use `.\.venv\Scripts\python.exe` instead of `python`. Setup scripts check each external command's exit status. To reproduce the tested environment, add `-Locked`.

## Linux / WSL

```bash
cd /path/to/repo
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
source .venv/bin/activate
python -m pytest
```

Select another Python 3.11 executable with `PYTHON_EXECUTABLE=/path/to/python3.11`. The script uses `set -euo pipefail`. Use `./scripts/setup_env.sh requirements-lock.txt` to reproduce exact versions; the lock is a platform-specific tested snapshot, so resolve curated requirements first if a transitive dependency is Windows-only.

For many raw files under WSL, prefer the Linux filesystem (for example `~/projects/vibiomir`) over `/mnt/c/...`; source Parquet may still be read from a Windows mount.

## Dataset and Stage A

The supplied source is [AIGuruTinix/ViBioMIR](https://huggingface.co/datasets/AIGuruTinix/ViBioMIR), with `corpus` and `query` configurations. Its native Parquet files can be downloaded without the `datasets` package:

```bash
python scripts/download_dataset.py --revision 0148f6f80ffafed5c005af6d506ccfd9d3fb47a7
python scripts/01_build_inventory.py
python scripts/run_stage_a.py --input data/source/links_corpus.parquet --query data/source/query.parquet --sample-size 100 --seed 42 --config configs/config.yaml
```

The downloaded revision contains **4,394,718 corpus rows** with `id: int64, url: string`, and **1,200 query rows** with `id, query` only. It contains **no gold document IDs/qrels**. Leave `gold_doc_ids_column: null`; query IDs must never be treated as corpus IDs. The user confirmed that BTC does not provide qrels and accepted proceeding without gold coverage. Reports keep coverage unavailable and its rates null; LOW remains available in benchmark handoff.

The downloader resolves an immutable HF commit, verifies LFS SHA256/size, writes atomically and refuses to overwrite different existing files. Inventory creates immutable local snapshots and records file hashes and schemas in `data/source/dataset_manifest.json`. Configure the actual query/gold column names only when a labeled query file is available; the adapter rejects type mismatches instead of guessing or converting BTC IDs. Corpus `id` is renamed internally, never renumbered.

Stage A is capped at 1,000 URLs. Sampling uses deterministic domain and predicted HTML/PDF strata plus gold-associated URLs. A successful execution report does not mark the manual quality/coverage gate passed. Explicit user acceptance is stored separately in `review_decision.json`; it does not fill unprovided per-document CSV ratings. The separate Stage B runner requires this acceptance, a passing Stage A integrity report and pilot authorization. It is capped at 10,000 URLs and never advances to 100k/full.

All output paths follow `environment.output_dir` or `--output-dir`. Input paths may be absolute, but no personal machine paths are hard-coded in source. A different dataset requires a new output directory. Config and extractor identities accompany manifests.

## Individual stages and resume

```bash
python scripts/01_build_inventory.py --config configs/config.yaml
python scripts/02_build_crawl_shards.py --config configs/config.yaml
python scripts/03_run_crawl.py --shard 0 --config configs/config.yaml
python scripts/04_extract_raw.py --config configs/config.yaml
python scripts/05_build_documents.py --config configs/config.yaml
python scripts/06_report.py --config configs/config.yaml
python scripts/validate_ingestion.py --config configs/config.yaml
```

Use the same config/output root to rerun a stage. Completed raw downloads are verified and skipped. A stopped job reuses its pending frontier and JOBDIR; failed requests use a new batch/job identity so old dupefilter entries cannot suppress retries. If JOBDIR is missing, reconstruct pending work from the compact manifest. JSONL events are durable per record, ordered by attempt/sequence; malformed crash tails are preserved in a `.truncated` artifact, while corruption in sealed/middle records fails explicitly.

For a controlled pause during development:

```bash
python scripts/03_run_crawl.py --shard 0 --stop-after 35
python scripts/03_run_crawl.py --shard 0
```

Raw storage uses checksum-based hash directories. HTML/text is zstd; PDF is binary. Success events are written only after an atomic raw commit. Different frontier URLs can follow the same redirect target and each retain an outcome; raw content dedup avoids duplicate blobs. Offline extraction checks checksum/decompressed size before parsing and kills timed-out workers; title/Markdown/plain text are separate fields.

## Outputs

- `data/processed/documents/documents.parquet`: one canonical content per row; plain `text`, structural `text_markdown`, title, language, quality and representative provenance.
- `data/mappings/content_doc_map.parquet`: every extracted BTC ID → crawl URL → canonical content.
- `data/mappings/doc_outcomes.parquet`: every source row/ID, including invalid, failed, pending, LOW and QUARANTINE.
- `data/manifests/crawl_manifest.parquet`, `extraction_manifest.parquet`: compact latest states; event history remains in JSONL parts.
- `reports/stage_a/report.json`, `gold_coverage.parquet`, `domain_estimates.parquet`, `coverage_decisions.json`, `manual_review.csv`; assessment fields in the CSV stay blank for human review.
- `reports/integrity.json`: validation with exit 0 on pass and nonzero on failures.

Quality thresholds and duplicate suspicion thresholds are configurable. HIGH/MEDIUM and valid LOW can be benchmark-eligible; without qrels, LOW is retained and gold-impact audit is unavailable. Widespread duplicates are quarantined only with error evidence; legitimate mirrored documents retain mappings. Heuristic LID uses scripts/accents/stopwords and returns null confidence; fastText can be benchmarked before adding a model adapter.

`pack_raw.py` publishes tar containers of independent compressed members and a separate packed manifest without deleting recovery blobs. Benchmark copy/upload and indexing before changing full-scale layout. The normal extraction manifest continues to reference loose blobs unless a verified packed view is explicitly selected.

## Kaggle

Latest state: **shard 2 is RUNNING** on Kaggle CPU with a **100,000-URL frozen frontier** and the shared session deadline described below. [Shard 00002](https://www.kaggle.com/code/caoban203/vibiomir-crawl-shard-00002), version 1, uses the updated private runner `caoban203/vibiomir-runner-20261006-session-deadline`. Kaggle status and pulled remote source confirm the submission. Runtime snapshot at **07:31 on 2026-10-06 (Asia/Saigon)** confirms **6h42m of crawling**, beyond the former 6h cap: **5,646/94,355 network outcomes**, including **5,488 SUCCESS_RAW**, with health **PROGRESS** and increasing counters. **5,645 deferred-domain URLs** are retained separately; **88,709 URLs** remain at that sample. Extraction follows the crawl stage. Shards 0 and 1 are completed (**2/44**); shards 3–43 have not been launched. Latest audit: `outputs/kaggle_campaign/deployment.json` and `outputs/kaggle_campaign/shard_00002/submission.json`; live snapshot: `shard_00002/logs/live_snapshot.log`. Campaign shards run sequentially. URLs inside each shard are read from a fixed frontier ordered by URL hash, then scheduled with up to 16 simultaneous requests and 2 per domain; response/retry/cooldown timing determines outcome order. Outcomes and resume decisions use `crawl_url_id`.

Shard 1 is **COMPLETED**, with **100,000 URL outcomes**: **91,461 successful raw downloads**, 2,801 crawl failures and 5,738 deferred URLs. Extraction retained **90,554 successes** and 907 failures; crawl and extraction have **zero pending**. Its completed continuation is [shard 00001 resume 02](https://www.kaggle.com/code/caoban203/vibiomir-crawl-shard-00001-20261005-resume-02), version 1. It restored resume 01 version 1 checkpoint SHA256 **658e034b43836f11d59ef4f7638ec79f42de075774085d6afa0c54f0f4b85442**, preserving **96,366** outcomes and scheduling only **3,634** pending URLs. It reused the reviewed runner unchanged and retained the 2 GiB extraction-event cap. Its audit is `outputs/kaggle_campaign/shard_00001_resume_02/submission.json`; saved report/log/checksum: `shard_00001_resume_02/download/`.

Resume 02 finished at **00:10 on 2026-10-06 (Asia/Saigon)**, with close reason `finished`. It added **3,634 URL outcomes**, **3,567 successful raw downloads** and **3,547 successful extractions**, processing only new raw with the previous extraction identity. Processing/export took approximately **56m25s**. Raw integrity and the 90,634-file bundle verified on Kaggle. Downloaded report and checksum agree on SHA256 **e9283dff7cba2e827470db7e7e64c057a1b065b6ac311d824cb27d5412829875**. The full tar remains on Kaggle and has not been downloaded or fully verified locally.

The initial shard 1 session is [shard 00001](https://www.kaggle.com/code/caoban203/vibiomir-crawl-shard-00001), version 1, submitted on **2026-10-05 (Asia/Saigon)** and finished at **15:28**. It reused the immutable campaign Dataset and reviewed runner `caoban203/vibiomir-runner-20261005-resume-01`, with tqdm and 30-second progress updates. Its frozen frontier checksum matched; fresh startup confirmed `Restore: None`. The stopped session's audit is `outputs/kaggle_campaign/shard_00001/submission.json`; a copy is preserved in `shard_00001_resume_01/previous_deployment.json`.

Previous session completion checked on **2026-10-05 (Asia/Saigon)**: processing/export took approximately **6h46m**; raw integrity and the 56,036-file archive verified on Kaggle. Downloaded report and SHA256 sidecar agree; the full tar remains on Kaggle. Report: `outputs/kaggle_campaign/shard_00001/download/vibiomir_shard_00001_report.json`.

The first continuation is [shard 00001 resume 01](https://www.kaggle.com/code/caoban203/vibiomir-crawl-shard-00001-20261005-resume-01), version 1, which finished at **22:34 on 2026-10-05 (Asia/Saigon)**. It restored shard 1 checkpoint SHA256 **d763efc1e7ece0382fc60082845fa6cba6df7ed9d0ac009971bd0235b8ceb5b3** and reused the reviewed runner unchanged. Remote parameters confirmed `SHARD_ID=1`, `AUTO_RESTORE=True`, `REQUIRE_RESTORE=True`; mounted input and frontier checksum passed verification. The extraction-event cap remained 2 GiB. Its audit is `outputs/kaggle_campaign/shard_00001_resume_01/submission.json`; saved report: `shard_00001_resume_01/download/vibiomir_shard_00001_report.json`. That deployment is preserved in `shard_00001_resume_02/previous_deployment.json`.

The first continuation added **32,416 URL outcomes**, including **31,399 successful raw downloads** and **31,052 successful extractions**, while preserving the previous extraction identity and processing only new raw. Processing/export took **6h30m**. Raw integrity and the 87,081-file archive verified on Kaggle; downloaded report and SHA256 sidecar agree on **658e034b43836f11d59ef4f7638ec79f42de075774085d6afa0c54f0f4b85442**. The full tar remains on Kaggle and is the checkpoint source for resume 02.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/kaggle_remote.py kernels status caoban203/vibiomir-crawl-shard-00002
.\.venv\Scripts\python.exe -X utf8 scripts/kaggle_remote.py kernels logs caoban203/vibiomir-crawl-shard-00002 --follow
```

The runner displays **tqdm** bars alongside the 30-second JSON heartbeat. The crawl bar uses only this invocation's scheduled pending URLs as its denominator; failures also advance it because they are committed outcomes. Domains retained for later are excluded from that network denominator. Extraction displays a count/rate without a fabricated total. Frontier preparation and validation have their own bars. `scripts/prepare_kaggle_shard.py --destination outputs/kaggle_campaign/shard_00001 --shard-id 1` prepares a fresh shard using the existing runner Dataset, verifies its frontier checksum and rejects an already submitted/completed shard. For a partial-shard continuation, mount both the previous `.tar` and `.tar.sha256`, keep the same `SHARD_ID`, and set `REQUIRE_RESTORE=True` with `EXPECTED_RESTORE_SHA256` from the selected report. Missing or mismatched inputs fail before crawling. `scripts/prepare_kaggle_resume.py` prepares that continuation with previous notebook output in `kernel_sources`; it does not download the raw tar locally. The extraction implementation is unchanged, so previously processed raw retains its extraction identity.

The original [ViBioMIR crawl shard 00000](https://www.kaggle.com/code/caoban203/vibiomir-crawl-shard-00000) and its [continuation](https://www.kaggle.com/code/caoban203/vibiomir-crawl-shard-00000-20261005-resume-01) have finished. Original inputs were `caoban203/vibiomir-crawl-b915d5d5-20261004` and `caoban203/vibiomir-runner-20261004`. Kaggle expands the uploaded source ZIP; notebook 02 also detects expanded code and prioritizes the runner Dataset's `source_package_manifest.json`.

On this Windows workspace, the isolated Kaggle CLI can check the remote run without affecting pipeline dependencies:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_remote.py kernels status caoban203/vibiomir-crawl-shard-00000
.\.venv\Scripts\python.exe scripts/kaggle_remote.py kernels logs caoban203/vibiomir-crawl-shard-00000
.\.venv\Scripts\python.exe scripts/kaggle_remote.py kernels output caoban203/vibiomir-crawl-shard-00000 -p outputs\kaggle_campaign\download
```

`RUNNING` means the remote session is active; completion is established by its saved shard report and verified bundle. Check `crawl_remaining` and `extraction_remaining`: a resource/time stop can finish the notebook successfully while its shard report remains `PARTIAL`. A cumulative storage cap needs adjustment after checking output sizes; time-limited partial runs can resume with the same cap. Global 4.4M completion requires all 44 shard reports/outcomes, not only the first notebook's success status.

The runner emits a flushed `[ViBioMIR progress]` JSON line every **30 seconds**. Read live logs with `kernels logs caoban203/vibiomir-crawl-shard-00002 --follow`. Each line shows the stage, counters and `idle_seconds`. During `crawl_prepare`, `inspected_urls` counts frontier rows inspected, not downloaded URLs. During `crawl`, `completed_this_run` and `counts_this_run` count durable URL outcomes including failures; `requests`, `responses` and `retries` show network activity. During `extraction`, `committed_this_run` counts extraction outcomes in this invocation, not the number of successful documents. Validation and tar packing/verification also report counters. These are per-invocation counters; the final report establishes cumulative shard completion.

`PROGRESS` means an observed counter or stage changed. `ALIVE_NO_PROGRESS` means the telemetry thread is alive but no new work was observed during this interval. After **300 seconds** with no counter or stage change, `POSSIBLE_STALL` calls for checking the current stage and crawl log; it is not proof of a deadlock and does not kill/restart the job. Known queued `Retry-After` cooldowns instead show `WAITING_COOLDOWN`. Rewriting heartbeat timestamps does not reset the idle timer. The checkpoint files are under `/kaggle/working/vibiomir_batch/checkpoints/`: `kaggle_activity.json`, `run_state.json`, `crawl_heartbeat.json` and `extraction_state.json`.

**The original shard 0 version 1 predates this telemetry.** It stopped after the 6h crawl timeout with 14,179 PENDING URLs. Its reviewed continuation subsequently finished shard 0, verified the combined archive on Kaggle and preserved all previous outcomes. Historical reports/logs remain under `outputs/kaggle_campaign/logs/progress_check` and `resume_01/download`; the full raw archive remains on Kaggle. The completed shard 1 used the runner with telemetry.

Upload/unzip this repository under `/kaggle/working`, mount the two Parquet source files as a Dataset, then import `notebooks/01_kaggle_stage_a.ipynb`. The maintained cell script is `01_kaggle_stage_a.py`; regenerate the notebook with `python scripts/build_notebook.py` after editing it. It provides environment info, conditional installs, unambiguous dataset discovery, a 100-URL sample, subprocess crawling/extraction, reporting and packaging. No `.venv` is created; `/kaggle/input` is read-only.

Alternatively, upload `outputs/vibiomir_source.zip` and the two Parquet files as Kaggle inputs. The notebook's first cell detects exactly one source ZIP and extracts it to `/kaggle/working/vibiomir_repo`. Create the source ZIP with `python scripts/package_source.py`; it excludes datasets, environments, raw cache and outputs.

Equivalent commands after preparing a config with input paths under `/kaggle/input` and output root `/kaggle/working/vibiomir`:

```bash
python scripts/kaggle_bootstrap.py
python scripts/run_stage_a.py --config /kaggle/working/vibiomir/config.yaml --sample-size 100
python scripts/package_artifacts.py --config /kaggle/working/vibiomir/config.yaml --destination /kaggle/working/vibiomir_stage_a_outputs.zip
```

Internet OFF fails early with `Network unavailable. Use pre-downloaded raw dataset or enable allowed Internet access.` Offline extraction does not need Internet. The Stage A artifact ZIP excludes raw blobs and SQLite working indexes; it is a review/handoff package, not a complete raw-cache backup. Save Version or create an output Dataset to persist artifacts; save raw containers separately for future extraction/resume.

For the user's selected Kaggle deployment, use **`notebooks/02_kaggle_crawl_shard.ipynb`**. It runs one frozen frontier per saved CPU session. [Kaggle documents](https://www.kaggle.com/docs/notebooks) 12 hours for CPU/GPU sessions and 20 GB of saved working output. The notebook uses a shared **11-hour deadline from its first cell**, counting dependency setup, restore and frontier preparation. It reserves **60 minutes for extraction** and **30 minutes for validation/export**; crawl uses the time left before those reserves. `CRAWL_HOURS=None` and `EXTRACTION_HOURS=None` disable separate fixed stage caps. Explicit numeric values remain optional additional caps. Extraction can use unused crawl time and stops before the export reserve. Expired budgets pause without starting new work; a zero remaining budget never enables an unlimited Scrapy timeout. Saved runs request the platform maximum timeout of **43,200 seconds**. The remaining hour is a margin for shutdown/export overhead, not extra crawl time. A 4 GiB compressed-raw budget and free-disk guard also pause the crawler before packaging. A shard may need more than one session.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_kaggle_campaign.py
.\.venv\Scripts\python.exe scripts/build_notebook.py --source notebooks/02_kaggle_crawl_shard.py
.\.venv\Scripts\python.exe scripts/package_source.py --destination outputs/kaggle_campaign/input/vibiomir_source.zip
```

The prepared private Dataset contains 44 fixed frontiers (43 × 100,000 + 92,746 URLs), the immutable source snapshot and native `url_doc_map.parquet`. Partitioning uses contiguous ranges of the frozen SHA256-ordered inventory; these boundaries must not be regenerated from a changed inventory during a campaign. Each session reads one frontier without copying/rebuilding the global SQLite inventory. Keep runs sequential so large domains do not receive requests from multiple notebook sessions at once.

Attach that Dataset, import notebook 02, select CPU with Internet ON, set `SHARD_ID=0`, and use **Save Version → Save & Run All** with outputs enabled. Saved outputs are `vibiomir_shard_00000.tar`, its `.tar.sha256`, and `_report.json`. The tar includes raw bytes, event history, extraction output, frontier and durable checkpoints; every member and the complete archive are checksummed and verified before its temporary loose cache is removed. Packing streams files and releases metadata after each member. Outputs stay within `/kaggle/working`; source inputs remain read-only.

Check the report's `status`, `crawl_remaining`, `extraction_remaining`, `integrity_passed` and `bundle.verified`. `PARTIAL` is an exported checkpoint, not a complete shard. Mount the previous tar **and its SHA256 file** as inputs in another run, keep the same `SHARD_ID`, and the notebook auto-restores exactly one matching bundle. Set `RESTORE_BUNDLE` explicitly if several versions are mounted. Restored queues/indexes are reconstructed from committed events; successful URLs and completed failures are skipped. After `COMPLETED`, preserve that output and increment `SHARD_ID`. A forced session termination before export still requires previously saved outputs; local checkpoint writes alone do not guarantee cross-session persistence.

Per-shard reports cover only that frontier. The original 4,394,718 source rows/IDs and invalid rows remain in the campaign inputs; global outcomes, canonical dedup and complete ID mappings are finalized after collecting all shards. Extraction quality cannot be inferred from HTTP success counts. The user-approved unavailable-domain and no-qrels policies also apply on Kaggle. The existing Windows 10k artifacts and their authorization record remain unchanged.

## Troubleshooting and operating gates

- **403/robots denied:** explicit terminal outcome. The user chose `defer_for_later`: keep failed URLs/outcomes and handle them later. Known deferred domains get `DEFERRED_DOMAIN` without a network request. `reports/stage_b/deferred_urls.parquet` retains all unsuccessful URLs for later review; no archive fetch is automatic.
- **302/meta-refresh cycles:** `REDIRECT_LOOP` is retained for later. The original redirect middleware allowed duplicate filtering to discard 124 `www.qdnd.vn` requests without an errback in the 10k pilot. The corrected middleware records cycles before scheduling another request; regression tests cover self, two-URL and meta-refresh cycles plus ordinary redirects.
- **429/503 Retry-After:** cooldown is persisted per hostname; scheduler parks unsent requests outside downloader slots. Rerun retry batches after the deadline; AutoThrottle alone does not parse Retry-After.
- **Empty extraction:** inspect raw checksum, encoding, MIME and manual preview; fallback is offline. Short valid text is retained when the extraction config allows it. Scanned PDF is flagged; no OCR in P0.
- **Disk full/raw corruption:** explicit failure status or validation error. Restore raw from a verified backup; do not erase successful history or IDs. Select a new output root for another snapshot.
- **JOBDIR:** preserve Scrapy version/config while paused; serialized requests are untrusted executable state, so restore only your own job artifacts. Event manifests remain the recovery source of truth.
- **Local/private addresses:** production rejects them, including DNS-resolved destinations and redirects. Only the automated test harness accepts its exact loopback fixture origin.
- **Language:** heuristic labels are metadata, not filtering decisions; ambiguous/short text remains unknown.

## Stage B on the current Windows machine

The user accepted Stage A content, confirmed qrels are unavailable, deferred inaccessible domains and selected the current Windows machine. `configs/config.yaml` therefore uses 8 concurrent requests, 2/domain, one extraction worker and a 20 MiB response cap. Disk/network planning budgets are 10 GiB / 5 GiB; these are planning values, not hard traffic limits. Initial free RAM/disk and sampled process-tree RSS are recorded. The independent inventory copy avoids changing Stage A or sharing a writable SQLite database.

```powershell
.\.venv\Scripts\python.exe scripts/run_stage_b.py --config configs/config.yaml --output-dir outputs/stage_b_windows_10k --stage-a-root outputs/stage_a_native --sample-size 10000
```

After the runner has stopped, rerun the same command to resume from durable manifests/JOBDIR and skip verified successful raw downloads. Keep one writer and the same config, sample and output root. Reports are under `outputs/stage_b_windows_10k/reports/stage_b/`: `report.json`, `benchmark.json`, `deferred_urls.parquet`, manual-review CSV and coverage decisions. Live progress/resources are under `checkpoints/`. RSS is a sampled sum of process working sets, not an exact allocation peak; shared pages may be counted twice.

Stage B resumes URLs without an outcome and retains completed failures for later, as requested by the user. The individual crawl CLI still supports explicit retry batches. HTTP/meta-refresh cycles produce `REDIRECT_LOOP` through the errback before duplicate filtering, so every frontier URL retains a terminal outcome. A normally finished crawler uses a new job for remaining work instead of replaying exhausted fingerprints.

The current pilot also has a detached Windows supervisor. It waits for the existing runner, attempts at most two recoveries if the runner fails, then verifies every packed raw checksum, measures packing/local copying and creates `outputs/vibiomir_stage_b_10k_outputs.zip`. It holds process handles to avoid confusing reused PIDs and never creates a second writer while the existing runner is alive. Status/logs: `checkpoints/supervisor_state.json`, `logs/supervisor.log`. Keep the machine on with Internet available while the crawl runs.

```powershell
.\.venv\Scripts\python.exe scripts/pilot_status.py
```

The status reader takes a snapshot and closes the file handle before parsing; it ignores an unfinished JSONL tail without changing it. Do not compact/repair an active writer's event parts. A supervisor can be attached explicitly with `run_stage_b.py --background-supervisor` and the same config/output/Stage A/sample arguments; do not launch another while one is recorded as running. After a manual run without supervision, use `python scripts/finalize_stage_b.py` for packing/local-copy measurements.

Review the 10k report before authorizing 100k. Full-scale storage, transfer, per-domain throughput and RAM estimates require measurements on the target environment.

The baseline 10k assistant quality assessment is at `outputs/stage_b_windows_10k/reports/stage_b/quality_audit/quality_review.md`, with a separate `agent_review.csv`, flagged URLs and raw comparison evidence. It identified charset corruption, navigation/homepage/template outputs and semantic structure/content loss. Original manifests/documents and the human review CSV are retained for comparison.

Reproduce the automated offline evidence with `python scripts/audit_quality.py --root outputs/stage_b_windows_10k`. This refreshes `audit.json` with a pending qualitative-review state; it does not approve quality or authorize another crawl. `scripts/audit_extraction_variants.py` runs supervised diagnostic variants on selected raw 10k cases without replacing pipeline outputs. The separate `outputs/vibiomir_stage_b_10k_quality_audit.zip` contains the completed assessment and evidence.

See `IMPLEMENTATION_STATUS.md` for commands actually run, measured results and outstanding validation. The Windows runner remains capped at its accepted 10k pilot; the separate Kaggle runner handles the newly requested deployment by bounded shard. Embedding, chunking production, OCR, browser rendering and automatic archive fallback remain separate work.

Extraction v2 uses immutable raw bytes, strict UTF-8/BOM/meta/header charset checks, and GB18030 for GB2312/GBK. The HTML pipeline combines normal Trafilatura extraction with source DOM containers verified in the native 10k audit. `structure_source` records the actual selection path; `implementation_version` and the code SHA256 included in extraction identity distinguish the repaired implementation. Nested H2/H3, list text, link text, table group labels and spans are preserved. Short wholly bold section titles are inferred as H3 only for the two checked publishers; `inferred_heading_count` distinguishes them from native headings.

`SOURCE_REDIRECT_HOME`, `UNRESOLVED_TEMPLATE`, `JAVASCRIPT_CHALLENGE` and `NAVIGATION_ONLY` retain their URL/raw records with QUARANTINE quality. Use `doc_outcomes.handoff_eligible` when preparing chunks; the stored documents include diagnostic quarantine content. Legitimate short LOW documents remain eligible in benchmark mode.

Re-extract the completed cached pilot into a separate directory without network access:

```powershell
.\.venv\Scripts\python.exe scripts/reextract_offline.py --source-root outputs/stage_b_windows_10k --output-root outputs/stage_b_windows_10k_reextract_v2
.\.venv\Scripts\python.exe scripts/finalize_reextract_quality.py --before outputs/stage_b_windows_10k --after outputs/stage_b_windows_10k_reextract_v2
```

Run one writer per destination. The offline runner copies verified packed raw containers and an independent inventory, retains packed offsets, rebuilds all native ID mappings and validates integrity. `checkpoints/offline_reextract_state.json` reports `status=completed` only after extraction, documents, integrity and report stages finish and the baseline SHA256 checks pass. During extraction, `checkpoints/extraction_state.json` records committed progress. The quality finalizer waits for completion, compares identical raw SHA256/URL IDs, includes the original 223 raw review cases and publishes `reports/stage_b/quality_comparison/quality_review.md`, `comparison.json`, all URL deltas and fixed 40 evidence. Its `completion.json` separately records the native regression result and review ZIP CRC.
