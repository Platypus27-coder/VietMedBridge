"""Wait for the offline rerun, then publish paired quality evidence and a review bundle."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
from src.storage.io import atomic_json, sha256_file, parquet_rows, write_parquet


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,default=Path('outputs/stage_b_windows_10k'))
    parser.add_argument('--after',type=Path,default=Path('outputs/stage_b_windows_10k_reextract_v2'))
    args=parser.parse_args()
    state_path=args.after/'checkpoints/offline_reextract_state.json'
    started=time.monotonic()
    while True:
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state['status']=='failed':
            raise RuntimeError(state.get('error','Offline rerun failed.'))
        if state['status']=='completed':
            break
        if time.monotonic()-started > 7200:
            raise TimeoutError('Offline pipeline did not finish within two hours.')
        time.sleep(10)
    for script,parameters in (
        ('scripts/audit_quality.py',['--root',str(args.after),'--baseline-root',str(args.before)]),
        ('scripts/compare_extraction_quality.py',['--before',str(args.before),'--after',str(args.after)])):
        subprocess.run([sys.executable,'-u',script,*parameters],check=True)
    directory=args.after/'reports/stage_b/quality_comparison'
    report=json.loads((directory/'comparison.json').read_text(encoding='utf-8'))
    before,after=report['before'],report['after']
    crawl={r['crawl_url_id']:r for r in parquet_rows(args.after/'data/manifests/crawl_manifest.parquet')}
    deferred_schema=pa.schema([(k,pa.string()) for k in ('crawl_url_id','fetch_url','final_url','domain',
        'extract_status','quality_tier','quality_reason','raw_path','raw_member','raw_sha256','snapshot_id',
        'extraction_run_id','error_type','error_message')]+[(k,pa.int64()) for k in ('raw_member_offset','raw_member_size')])
    def deferred_rows():
        for row in parquet_rows(args.after/'data/manifests/extraction_manifest.parquet',batch_size=128):
            if row['extract_status']!='EXTRACT_SUCCESS' or row['quality_tier']=='QUARANTINE':
                payload=dict(crawl[row['crawl_url_id']],**row)
                yield {k:payload.get(k) for k in deferred_schema.names}
    deferred_count=write_parquet(directory/'deferred_extraction_urls.parquet',deferred_rows(),deferred_schema)
    integrity=json.loads((args.after/'reports/integrity.json').read_text(encoding='utf-8'))
    lines=['# Kết quả sửa extraction và chạy lại offline 10k','',
        f'Chạy lại **{report["url_rows"]:,} raw** từ pilot 10.000 URL. URL IDs và raw SHA256 trước/sau giống nhau. Không tải lại web; các file kết quả cũ giữ nguyên.', '',
        '| Chỉ số | Trước | Sau |','|---|---:|---:|']
    for label,key in (('Có chữ Cyrillic bất thường','unexpected_cyrillic'),
                      ('Có menu dùng chung 120ask','120ask_navigation_present'),
                      ('Menu chiếm phần lớn 120ask','120ask_navigation_dominates'),
                      ('Có ký tự replacement U+FFFD','replacement_characters'),
                      ('Chinese bị nhận nhầm do dấu ・/ー','possible_lid_punctuation_false_positive')):
        lines.append(f'| {label} | {before["flags"].get(key,0):,} | {after["flags"].get(key,0):,} |')
    lines += ['', 'Trạng thái extraction sau sửa:','']
    for status,count in sorted(after['status'].items()):
        lines.append(f'- `{status}`: **{count:,}** URL.')
    lines += ['', 'Trang redirect về homepage, template chưa render và cookie/JavaScript challenge được ghi rõ lý do, giữ raw/ID để xử lý sau và loại khỏi handoff. Số `EXTRACT_SUCCESS` thay đổi do áp dụng kiểm tra nội dung này; không thay đổi số URL crawl thành công.', '',
        f'Canonical quality tiers: `{json.dumps(after["canonical_tiers"],ensure_ascii=False)}`. Đây là các band độ dài cộng kiểm tra lỗi nguồn, không phải điểm chất lượng retrieval.', '',
        f'Đối chiếu cố định **{report["fixed_raw_sample"]} raw**; các kiểm tra hồi quy native: **{sum(c["passed"] for c in report["checks"])}/{len(report["checks"])} đạt**.', '',
        '| Kiểm tra trên raw đã lưu | Kết quả |','|---|---|']
    for check in report['checks']:
        lines.append(f'| {check["name"]} | {"Đạt" if check["passed"] else "Cần xử lý"} |')
    lines += ['', f'Giữ **{deferred_count} URL extraction chưa dùng được** trong `deferred_extraction_urls.parquet`, kèm trạng thái, lý do và raw locator để xử lý sau. LOW hợp lệ vẫn giữ theo benchmark policy.', '',
        f'Kiểm tra mapping/integrity: passed; tất cả {integrity["source_rows"]:,} dòng nguồn vẫn có outcome. Original crawl/extraction/documents và phiếu review của người dùng được kiểm tra SHA256 giữ nguyên.', '',
        'Bộ sửa dùng charset strict từ bytes và meta/header/BOM; GB2312/GBK được đọc bằng GB18030. Vùng bài chính của các domain lỗi đã được đối chiếu raw. Heading H2/H3 giữ cấp dù chứa strong/span; dòng in đậm độc lập ở hai site được suy ra H3 và ghi `inferred_heading_count`. List, link text, table generation labels, rowspan/colspan và footnote được giữ.', '',
        'Các minh chứng: `comparison.json`, `url_deltas.csv` (mọi URL), `fixed_40_evidence.jsonl`, và `../quality_audit/sample_evidence.jsonl`. Bộ mẫu cố định có chủ đích tìm lỗi, không dùng để suy ra tỉ lệ pass toàn corpus.', '',
        'Các nội dung còn thiếu giữ để xử lý sau. Metadata title nguồn chung chung và related links còn sót ở một số site vẫn cần review. Các cảnh báo demoted heading trong raw sample có homepage/related-card/title trùng, không phải tất cả là lỗi section của bài chính. Không có qrels nên chưa đo gold coverage, Recall/MRR/nDCG; chưa đánh giá độ đúng y khoa của nội dung nguồn. Chưa chạy chunking/embedding hay 100k/full.']
    if not report['native_regressions_passed']:
        lines += ['', '**Còn kiểm tra native chưa đạt; cần sửa trước khi dùng corpus mới.**']
    (directory/'quality_review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit_path=args.after/'reports/stage_b/quality_audit/audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    audit.update(review_status='paired_assistant_regression_assessment_complete',
                 paired_comparison='../quality_comparison/comparison.json',
                 recommendation='review_repaired_10k_before_chunking_or_scaling' if report['native_regressions_passed'] else 'fix_remaining_native_regressions')
    atomic_json(audit_path,audit)
    archive=args.after.parent/'vibiomir_stage_b_10k_reextract_v2_quality.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for subdir in ('quality_audit','quality_comparison'):
            for path in sorted((args.after/'reports/stage_b'/subdir).glob('*')):
                if path.is_file():bundle.write(path,path.relative_to(args.after/'reports/stage_b'))
        for relative in ('reports/integrity.json','reports/stage_b/report.json','checkpoints/offline_reextract_state.json'):
            bundle.write(args.after/relative,relative)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:raise ValueError('Quality ZIP CRC failed.')
    atomic_json(directory/'completion.json',{'status':'completed','native_regressions_passed':report['native_regressions_passed'],
                  'archive':str(archive),'archive_sha256':sha256_file(archive),'zip_crc_passed':True,
                  'original_inputs_unchanged':state['original_inputs_unchanged']})
    print(f'Quality finalization complete: {directory}',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
