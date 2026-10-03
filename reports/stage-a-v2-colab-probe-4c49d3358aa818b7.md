# Colab diagnostic review: 4c49d3358aa818b7

The updated full ZIP confirms that **all 11 official Long Châu IDs are
accessible using the guarded Scrapling browser from Colab**. The next step
is offline extraction review with notebook 01f; no further fetch is required
for these captures. Fetch success has not yet changed the Stage A corpus.

## First upload: one-ID diagnostic

The uploaded `browser-probe-4c49d3358aa818b7.zip` confirms that official ID
206172 is accessible from the tested Colab runtime. Both `chrome_http` and
`dynamic_browser` received HTTP 200 for the original Long Châu URL. Its
canonical URL, H1 and six article H2 sections match the expected article.
The browser screenshot shows the actual article and opening paragraph; it
is not an error/challenge page. The browser's rendered document status is
also 200 and its guard reported no errors.

The run completed at **2026-10-03 00:33:39 Asia/Saigon**. HTTP took 3.897
seconds and extracted 6,565 characters. Browser took 19.757 seconds and
extracted 6,794 characters. These are two observations of **one official ID**,
not two recovered documents. The other ten 01d URLs have not been tested in
this Colab archive.

Integrity checks passed:

- ZIP SHA-256: `f148a3835cbfbe69d074417aabf385bcdf12577f1939e640d608c1aa48f9cf9c`.
- Experiment manifest digest matches `4c49d3358aa818b7`.
- `attempts.csv` SHA-256 matches its summary:
  `fd183117515a6fc695b152d8d256631624328665a1a92597f9571bd67f9cc39e`.
- All six asset hashes match their marker records. Both extracted-text hashes
  and character counts match the actual UTF-8 text files.
- The notebook SHA and package fingerprint match Git commit `2f9d010`.
- Robots HTTP 200 permits the article; the HTTP robots hash is
  `447fb8ea609815abb58295e8af2c8370f85a765d7f73786a4c276f17dbad131b`.

The browser recorded 100 allowed requests, 149 requests blocked outside the
host scope, and 8 blocked by the request budget. Its main document still
loaded successfully. Resource blocks are deliberate probe decisions, not
additional missing official article IDs.

Extraction review found genuine article content and its conclusion, but
generic extraction also retained author biography, display controls and
related-article titles. The browser variant additionally retained cookie
consent text. All six article H2 labels are present in raw HTML but absent
from both extracted text variants, even after whitespace normalization;
article structure therefore needs preservation in the extraction stage.
The recorded `quality_tier=HIGH` does not establish clean
article boundaries. `NO_SECTIONS` comes from the probe passing the default
section count into its quality helper; the HTML actually contains the six
article headings. `ONE_GIANT_PARAGRAPH` also needs to be interpreted with
the extractor's single-newline formatting. This sample passes the fetch
review for expanding the diagnostic; it is not ready for canonical ingestion
without extraction cleanup and source/span validation.

The requested follow-up was `N_URLS=11` in the same notebook 01e, keeping
`PROBE_LABEL` and `DATA_ROOT`. The full upload below completes that diagnostic
and reuses both existing sample checkpoints.

The exact cause of the earlier Colab 403 responses remains unisolated because
their bodies were not archived and the runtime/configuration changed.
This result demonstrates successful current access for the sample, not
guaranteed access for the remaining URLs or the entire benchmark corpus.

## Full upload: 11-ID diagnostic

The updated `browser-probe-4c49d3358aa818b7 (1).zip` completed at
**2026-10-03 09:20:43 Asia/Saigon**. It contains 11 distinct IDs, 22 method
checkpoints and 56 captured assets. ID/URL pairs match the official mapping
in `stage-a-v2-longchau-access-gap.csv`. Both ID 206172 markers are unchanged
from the first upload, demonstrating resume without repeating the sample.

- ZIP SHA-256: `d3d70ce7122ff2e57a0c0c32f5fcb98541e07a5951273758dc3c1b6f485d1a99`.
- Manifest digest still matches `4c49d3358aa818b7`.
- Full `attempts.csv` SHA-256:
  `c4c716d838f4c676e62cc7e9e8e9b8ba366ee04dd5f3e5d2abdc3560b3fdb807`.
- All 56 asset hashes match. Marker counts, unique ID/method pairs and
  outcomes match the pinned CSV and summary.
- All 11 browser responses/rendered documents are HTTP 200, contain the
  actual article and matching canonical URL, and report no guard errors.
- The ten new `chrome_http` attempts are HTTP 403. Every error body has
  title `Just a moment...`, a JavaScript/cookie prompt and response header
  `cf-mitigated: challenge`. Cloudflare documents this header as a
  [Challenge Page signal](https://developers.cloudflare.com/cloudflare-challenges/challenge-types/challenge-pages/detect-response/).

There are **12 candidate method results but only 11 candidate IDs**:
one successful HTTP sample plus eleven browser observations. The browser
took approximately 20–25 seconds per ID. This supports browser fallback
for these failures while keeping fast HTTP as the first tier; it does not
justify sending the entire 4.39-million-URL corpus through a browser.

## Extraction repair and cached replay

Generic probe text is retained as historical evidence. The new
`longchau-article-dom-v1` adapter selects the audited article container,
requires matching canonical identity, and keeps the title, standfirst and
source body in DOM order. Inline text is joined without injecting spaces
inside decimal values or combining Unicode. Headings, lists, table rows
and source captions stay within the article; UI outside that container is
excluded. Changed layout fails explicitly instead of importing menus.

Replay of the full archive produced **11/11 READY_FOR_EXTRACTION_REVIEW**,
all using original browser **response bytes**, rather than rendered DOM.
Each document has 4–13 known headings, 22–93 paragraph blocks and 4,102–9,341
source characters. Structure validation and hashes pass; rerunning reuses
all eleven output checkpoints. Every nonempty source paragraph and heading
inside the raw article body is retained. No tables occur in these eleven
bodies; table/caption preservation is covered by synthetic fixtures.
Zero heuristic quality flags does not
replace human review of article boundaries or establish retrieval relevance.

The local clean review ZIP is stored at
`data/recovery-extraction-4c49d3358aa818b7-local.zip` (ignored by Git), SHA-256
`1a168f3f44d5291a6ec1601a89a6c791828a1c7e9419f68e8f5b40efd6a91ebc`.
Its source-text/structure markers are under extraction signature
`6bb03593135e4c85b804f8fb803b39319ea68fce5ad1eac93d2836a8fecc806f`.
Colab may have a different code fingerprint/runtime identity and therefore
a different review signature, while retaining the same pinned source inputs.

Run **01f on a fresh Colab CPU runtime**, keeping the same Drive `DATA_ROOT`.
Its separate code lock fetches the new extractor without changing the
Stage A code lock. Input CSV/marker/asset hashes are verified before replay.
It exports per-ID text, structure, raw provenance, hashes and static review
HTML in a new versioned ZIP. It performs no additional article requests
and does not alter the Stage A raw or canonical corpus.

These eleven captures address the 01d Long Châu group only. The other
72 IDs among the original 83 unresolved entries still require their own
access/review work, including the four earlier 01c candidates. Do not
report 928 canonical successes before reviewed ingestion has occurred.
