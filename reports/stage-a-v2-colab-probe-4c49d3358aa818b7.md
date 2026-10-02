# Colab diagnostic review: 4c49d3358aa818b7

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
consent text. The recorded `quality_tier=HIGH` does not establish clean
article boundaries. `NO_SECTIONS` comes from the probe passing the default
section count into its quality helper; the HTML actually contains the six
article headings. `ONE_GIANT_PARAGRAPH` also needs to be interpreted with
the extractor's single-newline formatting. This sample passes the fetch
review for expanding the diagnostic; it is not ready for canonical ingestion
without extraction cleanup and source/span validation.

Next: in the **same notebook 01e**, set `N_URLS=11` in part 1 and rerun parts
1–4, keeping `PROBE_LABEL` and `DATA_ROOT`. Existing ID/method checkpoints
are verified and reused; only the remaining ten IDs are fetched. Export the
updated ZIP for review. No new notebook or GPU is needed for this step. Keep
the original Stage A corpus unchanged until a separate reviewed ingestion.

The exact cause of the earlier Colab 403 responses remains unisolated because
their bodies were not archived and the runtime/configuration changed.
This result demonstrates successful current access for the sample, not
guaranteed access for the remaining URLs or the entire benchmark corpus.
