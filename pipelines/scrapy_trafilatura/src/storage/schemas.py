import pyarrow as pa


def schema(strings=(), integers=(), floats=(), booleans=()):
    return pa.schema([(name, pa.string()) for name in strings] + [(name, pa.int64()) for name in integers] +
                     [(name, pa.float64()) for name in floats] + [(name, pa.bool_()) for name in booleans])


CRAWL_SCHEMA = schema(
    strings='crawl_url_id event_id snapshot_id config_sha256 run_id shard_id fetch_url final_url domain status content_type declared_content_type effective_content_type declared_charset detected_charset charset_detection_method sniff_reason raw_path raw_member raw_sha256 error_type error_message crawl_timestamp retry_after source_provenance'.split(),
    integers='attempt_no event_seq http_status raw_locator_version raw_size_bytes compressed_size_bytes attempt_count redirect_count'.split(),
    floats='elapsed_ms retry_not_before'.split(), booleans='retry_exhausted content_type_mismatch'.split())
EXTRACT_SCHEMA = schema(
    strings='crawl_url_id event_id raw_sha256 snapshot_id extraction_run_id extractor_config_sha256 extract_status extractor extractor_version title text text_markdown text_format structure_source language language_method quality_tier quality_reason content_hash hash_normalization_version error_type error_message extraction_timestamp processed_path extraction_charset encoding_method source_issue implementation_version'.split(),
    integers='extraction_attempt_no event_seq char_count paragraph_count heading_count list_item_count table_count page_count inferred_heading_count'.split(),
    floats='language_confidence replacement_char_ratio extraction_elapsed_ms'.split(), booleans='fast_pass_success fallback_used encoding_conflict'.split())
DOCUMENT_SCHEMA = schema(
    strings='canonical_content_id representative_crawl_url_id url final_url domain title text text_markdown text_format structure_source language language_method content_type quality_tier quality_reason content_hash hash_normalization_version extractor extractor_version extractor_config_sha256 extraction_charset encoding_method source_issue implementation_version'.split(),
    integers='char_count paragraph_count'.split(), floats='language_confidence'.split())
