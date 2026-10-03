# Stage A: Long Châu HTTP/browser diagnostic

The earlier Colab results do **not** prove these URLs cannot be crawled.
On 2026-10-03 (Asia/Saigon), a controlled local follow-up fetched all 11
official URLs from the 01d failure ledger with HTTP 200. Generic extraction
produced 4,636–9,811 characters per URL and matching article titles. The
[per-ID result ledger](stage-a-v2-local-http-preview.csv) records original URLs,
response hashes and extracted-text file hashes. These are article candidates
pending content/extraction review, not accepted canonical corpus documents.

The local HTTP test used Scrapling 0.4.15, curl_cffi 0.16.3,
`impersonate="chrome"`, explicit VietMedBridge User-Agent, no synthetic Google
referer, concurrency 1, a 3-second host gap, one attempt per URL, fresh robots
checks and manually guarded redirects. Each URL had an allowed robots decision.

An ordinary Scrapling `AsyncDynamicSession` also fetched and rendered official
ID 206172 with status 200. This local session used the installed **Edge
(Chromium)** executable with a temporary profile, Playwright 1.63.0 and no
login/profile reuse. The screenshot shows the correct article heading and
opening paragraph. The rendered page extracted 6,794 characters; the HTTP
response extracted 6,565. There were no guard errors. The broader extraction
includes some author, related-link and cookie UI text, so an extraction review
is still required before canonical ingestion.

Local evidence stays under the ignored data directory:

- `data/local_http_preview_20261003/`: all 11 response bodies, extracted text,
  per-ID JSON and summary.
- `data/local_browser_probe_20261003/`: the first URL's HTTP response, browser
  response, rendered DOM, extracted text, screenshot and request logs.

For ID 206172, the first local HTTP response SHA-256 is
`ec0068382ea192f66227208d31354ecab414fca9477b10b248fdc2bd1e5cd91a`.
Its browser rendered-DOM SHA-256 is
`bb269fe4ec06835b03c6b7b363119be83fa62795b2ecea37a8f363c3b568991c`.
Raw article text and screenshots are not committed to GitHub.

This follow-up changes both transport configuration and network environment
relative to 01d. 01c already used Chrome HTTP impersonation and still saw 403
from Colab. Therefore local success cannot be attributed solely to Chrome TLS,
and it does not prove access from a Colab IP. The successful responses identify
Cloudflare in their headers, but there is no captured 403 body from the earlier
runs to establish why those requests were denied.

## Colab follow-up

The uploaded Colab archive `4c49d3358aa818b7` has now passed integrity and
fetch review for ID 206172: both HTTP and browser returned the correct article
with status 200. See [the Colab review](stage-a-v2-colab-probe-4c49d3358aa818b7.md).
The one-URL diagnostic can expand to 11 while preserving its checkpoints;
generic extraction still needs article-boundary cleanup before ingestion.

Run `notebooks/experiments/stage-a-1000/01e_colab_browser_diagnosis.ipynb` in a fresh CPU runtime, using the same
Drive data root and the existing 01d outputs. It has a separate
`browser_probe_code_lock.json` and does not change Stage A's code lock.

Start with `N_URLS=1`. The notebook compares Chrome HTTP transport with a fresh
Chromium JavaScript session and archives responses including errors, filtered
headers, rendered DOM, screenshots, hashes and request/robots logs. Guard
setup completes before navigation; CDP Request-stage interception checks
redirect destinations. Service workers, worker constructors, WebSockets and
popups are disabled for this small diagnostic. Only GET requests to the
explicit host scope are permitted, with per-host pacing, robots checks for
both the crawler and browser product, request limits and a 429 hold. Third-party
and out-of-scope resource blocks appear in the request log; a failed render
under this restricted configuration is not proof that a full browser cannot
render the page. There is no automatic challenge solver or proxy rotation.

After checking the sample's article, set `N_URLS=11` and resume the same
experiment. Each ID/method has an atomic Drive checkpoint with hashed assets.
To retry a failed experiment, change `PROBE_LABEL` so its evidence is retained.
The final cell exports a ZIP containing the manifest, summary, CSV, markers and
assets for review. Do not count the captures as Stage A recovery until content
review and a separate ingestion step succeed.

## Validation

The complete regression suite, including the two live protocol checks, passed:
**42 tests**. All eight notebook files passed notebook schema and cell syntax
validation.

Unit checks cover scope limits, robots denials/errors, browser product rules,
request budgets, failed setup preventing navigation, and diagnostic header
filtering. Optional integration checks use real Chromium with synthetic CDP
responses and no website traffic: an allowed first URL redirects to a
Disallow URL and is blocked; an allowed redirect returning 403 has its actual
error HTML, DOM and screenshot archived. The local live sample above verifies
the adapter also works with an actual article. Notebook cells are validated
for notebook schema and Python syntax; the uploaded one-URL Colab execution
has also been reviewed as described above.
