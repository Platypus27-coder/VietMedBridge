"""Deterministic, domain/format-stratified pilot sampling with bounded SQL memory."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .artifacts import atomic_json, publish_file, utc_now
from .dataset import _connection, load_snapshot, require_schema, validate_link_subset


def copy_parquet(connection, query: str, destination: Path) -> None:
    # SQL string quoting only; this never constructs a shell command.
    literal = str(destination).replace("'", "''")
    connection.execute(f"COPY ({query}) TO '{literal}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def create_stage_a_sample(data_root: str | Path, *, size: int = 1000, seed: int = 20261002,
                          work_dir: str | Path | None = None) -> dict:
    if size < 1:
        raise ValueError("Sample size must be positive.")
    root = Path(data_root)
    snapshot = load_snapshot(root)
    links_entry = snapshot["files"]["links_corpus.parquet"]
    links = root / links_entry["path"]
    require_schema(links, "url")
    target = root / "reports" / "inventory"
    target.mkdir(parents=True, exist_ok=True)
    if work_dir:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vmb-inventory-", dir=work_dir) as temporary:
        with _connection(temporary) as con:
            con.read_parquet(str(links)).create_view("links")
            con.execute("""CREATE VIEW inventory_base AS SELECT id, url,
                regexp_replace(url, '#.*$', '') AS canonical_url,
                lower(regexp_extract(url, '^(https?)://', 1)) AS scheme,
                lower(regexp_extract(url, '(?i)^https?://([^/:?#]+)', 1)) AS domain,
                regexp_extract(url, '(?i)^https?://[^/?#]+([^?#]*)', 1) AS path,
                lower(regexp_extract(regexp_replace(url, '[?#].*$', ''), '\\.([a-zA-Z0-9]+)$', 1)) AS extension,
                regexp_matches(url, '(?i)^https?://[^/?#\\s]+') AS url_valid
                FROM links""")
            con.execute("""CREATE VIEW domain_rank AS SELECT domain, count(*) AS population,
                row_number() OVER (ORDER BY count(*) DESC, domain) AS domain_rank
                FROM inventory_base GROUP BY domain""")
            con.execute("""CREATE VIEW inventory AS SELECT b.*,
                CASE WHEN r.domain_rank <= 30 THEN 'top'
                     WHEN r.domain_rank <= 300 THEN 'medium' ELSE 'long_tail' END AS domain_tier,
                CASE WHEN extension = 'pdf' THEN 'pdf' WHEN extension = 'xml' THEN 'xml'
                     WHEN extension IN ('', 'html', 'htm', 'php', 'aspx') THEN 'html_like' ELSE 'other' END AS format_hint,
                CASE WHEN ends_with(b.domain, '.vn') THEN 'vi_like'
                     WHEN ends_with(b.domain, '.cn') THEN 'zh_like' ELSE 'unknown' END AS language_hint
                FROM inventory_base b JOIN domain_rank r USING (domain)""")
            con.execute("""CREATE VIEW strata AS SELECT *,
                domain_tier || ':' || format_hint || ':' || language_hint || ':' ||
                CASE WHEN url_valid THEN 'valid' ELSE 'invalid' END AS stratum FROM inventory""")
            con.execute(f"""CREATE TEMP TABLE pilot AS WITH within_domain AS (
                SELECT *, row_number() OVER (PARTITION BY stratum, domain ORDER BY hash(id, {int(seed)}), id) AS domain_position
                FROM strata
            ), within_stratum AS (
                SELECT *, row_number() OVER (PARTITION BY stratum ORDER BY domain_position, hash(domain, {int(seed)}), id) AS stratum_position
                FROM within_domain
            ) SELECT id, url, domain, domain_tier, format_hint, language_hint, stratum
              FROM within_stratum ORDER BY stratum_position, hash(stratum, {int(seed)}), id LIMIT {int(size)}""")
            con.execute("""CREATE VIEW stratum_counts AS SELECT s.stratum, s.population, coalesce(p.sample_rows, 0) AS sample_rows
                FROM (SELECT stratum, count(*) AS population FROM strata GROUP BY stratum) s
                LEFT JOIN (SELECT stratum, count(*) AS sample_rows FROM pilot GROUP BY stratum) p USING (stratum)""")
            names = {kind: Path(temporary) / f"{kind}.parquet" for kind in ("url_inventory", "stage_a_links", "stratum_counts")}
            copy_parquet(con, "SELECT * FROM strata", names["url_inventory"])
            copy_parquet(con, "SELECT * FROM pilot ORDER BY stratum, domain, id", names["stage_a_links"])
            copy_parquet(con, "SELECT * FROM stratum_counts ORDER BY stratum", names["stratum_counts"])
            counts = con.execute("SELECT count(*) FROM pilot").fetchone()[0]
            strata = con.execute("SELECT * FROM stratum_counts ORDER BY stratum").fetchall()
        files = {kind: {"path": (Path("reports/inventory") / path.name).as_posix(),
                        "sha256": publish_file(path, target / path.name)} for kind, path in names.items()}
    validate_link_subset(root / files["stage_a_links"]["path"], links, work_dir=work_dir)
    result = {
        "dataset_revision": snapshot["revision"], "corpus_sha256": links_entry["sha256"],
        "rows": counts, "seed": seed, "files": files,
        "strata": [dict(zip(("stratum", "population", "sample_rows"), row)) for row in strata],
        "method": "balanced strata and domains; deterministic SQL hash ordering, DuckDB pinned",
        "limitations": ["URL/domain language hints are not detected content languages",
                         "pilot is for feasibility; unweighted rates are not full-corpus forecasts",
                         "canonical_url is inventory-only; requests keep the official URL"],
        "created_at": utc_now(),
    }
    atomic_json(target / "stage_a.json", result)
    return result
