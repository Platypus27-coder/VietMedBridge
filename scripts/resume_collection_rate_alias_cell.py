# Run below the failed collection cell in the same Colab runtime.
import json

def _collection_reason_aliases(response, original=collector.drive_reasons):
    aliases = {
        'RATE_LIMIT_EXCEEDED': 'rateLimitExceeded',
        'USER_RATE_LIMIT_EXCEEDED': 'userRateLimitExceeded',
    }
    return tuple(sorted({aliases.get(reason, reason) for reason in original(response)}))

# Patch the already loaded reader, including notebooks pinned to the prior commit.
# The default argument captures the old function so re-running cannot recurse.
collector.drive_reasons = _collection_reason_aliases
previous = globals().get('flow')
if previous is not None:
    previous.close()
reader = collector.CollectionDriveSource(session)
flow = collector.CollectAll(PROJECT, WORK)
try:
    if not flow.saved('official'):
        raise RuntimeError('Thiếu checkpoint danh sách BTC; chưa bắt đầu thu gom.')
    print('Đã sửa nhận dạng rate limit; dùng lại archive và metadata đã checkpoint.', flush=True)
    REPORT = flow.run(reader, collector.inventory(session), RETAINED)
    print(json.dumps({key: REPORT[key] for key in (
        'state', 'counts_by_official_id', 'archive_errors', 'all_official_content_available'
    )}, ensure_ascii=False, indent=2), flush=True)
    print('Kết quả:', flow.root, flush=True)
    if REPORT['archive_errors']:
        raise RuntimeError('Chưa đọc hết nguồn; xem reason/cooldown trong archive_errors. Checkpoint được giữ lại.')
finally:
    flow.close()
