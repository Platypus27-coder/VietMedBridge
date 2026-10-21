"""One CPU notebook: all archives, current BTC membership, durable recovery."""
from pathlib import Path
from textwrap import dedent
import json
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def notebook():
    source = (ROOT/'scripts/collect_all_btc_sources.py').read_text(encoding='utf-8')
    retained = json.loads((ROOT/'configs/intake_ai_guru_full_20261019.json').read_text())['retained']
    setup = dedent('''
        #@title 1. Colab CPU — toàn bộ AI Guru, không chỉ batch 10 nguồn
        import importlib
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        print('ALL BTC CONTENT: read metadata across all archive versions, retain existing checkpoints.',flush=True)
        if not Path('/var/colab/hostname').is_file():
            raise RuntimeError('Chạy notebook này trên Colab CPU; không dùng GPU/Kaggle cho bước thu gom Drive.')
        from google.colab import drive,auth
        drive.mount('/content/drive')
        PROJECT=Path('/content/drive/MyDrive/VietMedBridge')
        parent='data_staging/btc-new-71be0e12-20261019-v1'
        if not (PROJECT/parent/'full_intake_scope.json').is_file():
            PROJECT=Path('/content/drive/.shortcut-targets-by-id/1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK/VietMedBridge')
        PROJECT=PROJECT.resolve()
        WORK=Path('/content/vmb_collect_all_work')
        WORK.mkdir(parents=True,exist_ok=True)
        CHECKOUT=Path('/content/VietMedBridge-collect-all')
        REPO='https://github.com/Platypus27-coder/VietMedBridge.git'
        COMMIT='bbf09ec01d542a5b06a4c2beec245571abeb3515'
        if not (CHECKOUT/'.git').is_dir():
            subprocess.run(['git','clone','--no-checkout',REPO,str(CHECKOUT)],check=True)
        if subprocess.check_output(['git','-C',str(CHECKOUT),'remote','get-url','origin'],text=True).strip()!=REPO:
            raise ValueError('Unexpected checkout repository.')
        subprocess.run(['git','-C',str(CHECKOUT),'fetch','--depth','1','origin',COMMIT],check=True)
        subprocess.run(['git','-C',str(CHECKOUT),'checkout','--detach','FETCH_HEAD'],check=True)
        subprocess.run([sys.executable,'-m','pip','install','-e','.', 'numpy'],cwd=CHECKOUT,check=True)
        sys.path[:0]=[str(WORK),str(CHECKOUT/'src')]
        importlib.invalidate_caches()
        auth.authenticate_user()
    ''').strip()
    prepare = '#@title 2. Khôi phục SQL checkpoint và mapping đã xử lý — không đọc lại corpus\n'
    prepare += 'SOURCE = '+repr(source)+'\nRETAINED = json.loads('+repr(json.dumps(retained))+')\n'
    prepare += dedent('''
        (WORK/'collect_all_btc_sources.py').write_text(SOURCE,encoding='utf-8')
        import collect_all_btc_sources as collector
        # Re-uploading a repaired notebook must replace the old imported reader
        # even when the user retains the same Colab runtime and Drive checkpoint.
        collector=importlib.reload(collector)
        from vietmedbridge.isolated_intake import DriveSource,fetch_reference
        from vietmedbridge.artifacts import read_json,verify_file,digest_json
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        credentials,_=google.auth.default()
        session=AuthorizedSession(credentials)
        reader=collector.CollectionDriveSource(session)
        flow=collector.CollectAll(PROJECT,WORK)
        scope=read_json(PROJECT/parent/'full_intake_scope.json')
        # Only membership/mapping files are read; no old document text, vectors, or index.
        if not flow.saved('official'):
            links=PROJECT/parent/'references/5586f423916005c1-links.parquet'
            flow.official(links)
        mappings=[(PROJECT/parent/'references/baseline_aliases.parquet',scope['base_plan']['baseline']['sha256'])]
        batch_root=PROJECT/parent/'freeze_batch_v1'
        batch=read_json(batch_root/'freeze_batch.json')
        collector.checked(batch)
        for entry in batch['builds']:
            build=(PROJECT/entry['build_dir']).resolve()
            if not build.is_relative_to(PROJECT):
                raise ValueError('Build mapping escapes project.')
            candidate=read_json(build/entry['candidate_name'])
            sha=candidate['candidate_manifest_sha256']
            if sha!=entry['candidate_manifest_sha256'] or digest_json({k:v for k,v in candidate.items() if k!='candidate_manifest_sha256'})!=sha:
                raise ValueError('Frozen mapping producer changed.')
            item=candidate['health']['files']['canonical_aliases']
            path=(build/item['path']).resolve()
            if not path.is_relative_to(build):
                raise ValueError('Alias mapping escapes build.')
            mappings.append((path,item['sha256']))
        for path,checksum in mappings:
            # A committed mapping is not opened/rehashed again in another runtime.
            if not flow.saved('existing:'+checksum):
                flow.existing(path,checksum,path.relative_to(PROJECT).as_posix())
        print('EXISTING_MAPPING_READY: baseline + all 10 frozen sources retained.',flush=True)
    ''')
    run = dedent('''
        #@title 3. Thu gom tất cả archive và xuất danh sách URL BTC còn thiếu/thất bại
        files=collector.inventory(session)
        print('AI_GURU_PHYSICAL_ARCHIVES:',len(files),flush=True)
        print('Đọc mọi bản resume; không bỏ archive theo số shard/ngày hoặc snapshot cũ.',flush=True)
        try:
            REPORT=flow.run(reader,files,RETAINED)
            if not REPORT['archive_errors'] and REPORT['raw_path_classification']['pending_leaves']:
                print('Cập nhật bộ lọc raw cũ từ metadata đã lưu; không mở lại TAR.',flush=True)
                REPORT=flow.repair_raw_paths()
            print(json.dumps(REPORT,ensure_ascii=False,indent=2),flush=True)
            print('Kết quả:',flow.root,flush=True)
            print('btc_membership.csv: mỗi ID BTC có trạng thái và provenance.',flush=True)
            print('missing_or_unusable.csv: chỉ các URL chưa có nội dung dùng được.',flush=True)
            print('parts/: nội dung khôi phục từ archive; cần đi qua chunk/freeze chuẩn trước embedding.',flush=True)
            if REPORT['archive_errors']:
                raise RuntimeError('Có archive chưa đọc được; không kết luận đủ BTC. Các source đã hoàn tất vẫn có checkpoint.')
            if REPORT['counts_provisional']:
                details=REPORT.get('raw_path_repair') or {}
                raise RuntimeError('Phân loại raw chưa cập nhật xong: '+json.dumps(details.get('errors',[]),ensure_ascii=False))
        finally:
            flow.close()
    ''').strip()
    book=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(dedent('''
        # 02 — Thu gom toàn bộ nội dung BTC từ AI Guru (Colab CPU)

        **Chọn CPU → Run all.** Không chạy lại 02/03 cũ, không cần GPU hay Kaggle token.
        Quét toàn bộ folder AI Guru qua Drive API, gồm shard cũ, mới, mọi bản resume,
        các TAR lồng nhau và retained. Mỗi URL được đối chiếu với snapshot BTC
        `71be0e12`; ID và URL lấy nguyên từ file BTC. Không đếm file/thành công HTTP
        như tài liệu hợp lệ. Giữ riêng snapshot gốc của crawl/extraction.

        Dùng lại mapping baseline và 10 nguồn đã freeze để tránh xử lý lại tài liệu.
        Chỉ lưu thêm nội dung vượt kiểm tra metadata/extraction vào kho recovery trên Drive.
        Không copy nguyên TAR/raw HTML, không encode model, không rebuild index hay đổi
        candidate của luồng submit. Dữ liệu tạm chỉ ở runtime Colab; không tải về PC.

        Checkpoint sau mỗi leaf/archive gồm một SQL snapshot nén trên Drive. Đổi VM
        phục hồi một snapshot; archive/mapping đã commit không mở lại. Một leaf ngắt
        giữa chừng có thể đọc lại leaf đó. Không tự xóa checkpoint hoặc archive gốc.

        **Đã sửa lỗi giới hạn 128 header:** native leaf đọc đủ bốn manifest thì
        dừng liệt kê raw. Container vẫn duyệt hết mọi TAR con, gồm resume ở cuối.
        Nếu bản trước báo 26 archive lỗi, mở bản này và Run all trên CPU; mapping
        và archive đã commit được bỏ qua, chỉ các archive lỗi được thử tiếp.
        Không xóa folder checkpoint và không chạy đồng thời hai phiên.

        Bộ đọc in `reason` và byte range khi Drive trả 403. Chỉ retry lỗi tốc độ
        tạm thời; quyền truy cập/quota tải/403 không rõ nguyên nhân được báo riêng.
        Metadata đã tải xong được checkpoint trên Drive dù file tiếp theo lỗi.

        Native metadata chấp nhận `data/raw/...` và `raw/...`; retained portable
        giữ contract `raw/...`. Checkpoint của bộ lọc cũ được sửa một lần từ
        metadata đã lưu (gồm metadata của các lượt import trước); không mở TAR,
        không encode lại. File metadata thiếu được báo riêng, không tự tải lại.
        Bản ghi SUCCESS_RAW có hash/path/snapshot sai được xếp RAW_METADATA_REVIEW,
        không gọi nhầm là lỗi crawl. Xem `counts_provisional` trước khi chốt số liệu.

        Đầu ra `data_staging/btc-all-ai-guru-71be0e12-v1`:
        `collection_report.json`, `btc_membership.csv`, `missing_or_unusable.csv`,
        các Parquet chứa text recovery cùng nguồn gốc raw/hash/snapshot.
        `ALL_FOLDER_SOURCES_COLLECTED` chỉ xác nhận đọc xong folder; xem riêng
        `all_official_content_available` và các nhóm trạng thái để biết có đủ nội dung.
        Nội dung recovery vẫn cần chunk/freeze chuẩn; đây chưa phải pipeline inference
        hay xác nhận mọi URL trên Internet đều truy cập được.
        Chỉ chạy một phiên CPU notebook này trên cùng output folder.
    ''').strip()),nb.v4.new_code_cell(setup),nb.v4.new_code_cell(prepare),nb.v4.new_code_cell(run)])
    book.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
    return book


if __name__=='__main__':
    nb.write(notebook(),ROOT/'notebooks/02_colab_collect_all_btc_ai_guru.ipynb')
