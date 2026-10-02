# Stage A Long Châu access gap

The 01d Colab follow-up ran on 2026-10-02 against the 11 non-candidates from
01c experiment `6a5793cdc03be0cb`. All 11 official IDs are unique and belong
to `nhathuoclongchau.com.vn`. Each original 01c attempt and follow-up 01d
attempt received target HTTP 403. The 01d robots probe returned HTTP 200 and
`ROBOTS_OK_ALLOWED` for every URL. Each follow-up made one page request, saw no
redirect, and used a 3-second per-host delay. No new article candidate resulted.

The source files supplied by the Colab operator were `attempts.csv` (SHA-256
`def3ac4cf8beb4be7d950282951db76e18804bc705c3b5a63c83eb801623ea59`)
and `summary.json` (SHA-256
`4c589678978803434b92b6968ac64a6cf0407190077b146f7705ba7db0213cf4`).
The attempts checksum matches the summary. The experiment key is
`3e6c01b92a7e3194`. The compact [ID/URL ledger](stage-a-v2-longchau-access-gap.csv)
is derived from these files; it does not contain article text.

This is a target-site access failure, not an extraction failure. A single
official [sample article](https://nhathuoclongchau.com.vn/bai-viet/6-cach-can-bang-noi-tiet-to-o-do-tuoi-trung-nien.html)
is publicly indexed, but that does not establish access from Colab for the
other ten URLs or permission for bulk reuse. Long Châu's
[terms](https://nhathuoclongchau.com.vn/chinh-sach/tos) reserve rights in site
content. Do not count these 11 IDs as recovered or replace their text with
search snippets, unrelated articles, or generated content.

Next source-access action: ask the benchmark organizer or Long Châu for an
authorized article snapshot, WARC, feed or API for the **same official URLs**,
with terms of use. Preserve each `doc_id` and record source method, capture
date, original/final URL, raw hash and text hash before any later ingestion.
Until such a source exists, retain these IDs as unresolved in the coverage
ledger. The four 01c article candidates remain pending human review and are
separate from this 11-ID gap.
