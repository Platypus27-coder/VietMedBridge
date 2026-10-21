"""One CPU notebook: checked text first, then saved raw, durable delta outputs."""
import json
from pathlib import Path
from textwrap import dedent
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def notebook():
    collect=(ROOT/'scripts/collect_all_btc_sources.py').read_text(encoding='utf-8')
    maximize=(ROOT/'scripts/maximize_collection.py').read_text(encoding='utf-8')
    plan=json.loads((ROOT/'configs/intake_ai_guru_full_20261019.json').read_text(encoding='utf-8'))['base_plan']
    setup=dedent('''
        #@title 1. Colab CPU — kết nối Drive, code/parser/chunk policy cố định
        import importlib
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        print('MAXIMIZE: CPU only; preserve paid corpus and active submission.',flush=True)
        if not Path('/var/colab/hostname').is_file():
            raise RuntimeError('Chạy notebook này trên Colab CPU.')
        from google.colab import drive,auth
        drive.mount('/content/drive')
        PROJECT=Path('/content/drive/MyDrive/VietMedBridge')
        relative='data_staging/btc-all-ai-guru-71be0e12-v1/collection_report.json'
        if not (PROJECT/relative).is_file():
            PROJECT=Path('/content/drive/.shortcut-targets-by-id/1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK/VietMedBridge')
        if not (PROJECT/relative).is_file():
            raise FileNotFoundError('Chưa thấy báo cáo collection đã hoàn tất trong VietMedBridge.')
        PROJECT=PROJECT.resolve()
        WORK=Path('/content/vmb_maximize_work')
        WORK.mkdir(parents=True,exist_ok=True)
        CHECKOUT=Path('/content/VietMedBridge-maximize')
        REPO='https://github.com/Platypus27-coder/VietMedBridge.git'
        COMMIT='bbf09ec01d542a5b06a4c2beec245571abeb3515'
        if not (CHECKOUT/'.git').is_dir():
            subprocess.run(['git','clone','--no-checkout',REPO,str(CHECKOUT)],check=True)
        if subprocess.check_output(['git','-C',str(CHECKOUT),'remote','get-url','origin'],text=True).strip()!=REPO:
            raise ValueError('Unexpected checkout repository.')
        subprocess.run(['git','-C',str(CHECKOUT),'fetch','--depth','1','origin',COMMIT],check=True)
        subprocess.run(['git','-C',str(CHECKOUT),'checkout','--detach','FETCH_HEAD'],check=True)
        subprocess.run([sys.executable,'-m','pip','install','-e',str(CHECKOUT),'numpy'],check=True)
        runtime_lock=PROJECT/'data_staging/btc-maximize-71be0e12-v1/runtime_lock.json'
        if runtime_lock.is_file():
            pinned=json.loads(runtime_lock.read_text())
            subprocess.run([sys.executable,'-m','pip','install',*[k+'=='+v for k,v in pinned.items()
                if k!='vietmedbridge']],check=True)
        sys.path[:0]=[str(WORK),str(CHECKOUT/'src')]
        importlib.invalidate_caches()
        from vietmedbridge.artifacts import atomic_json,runtime_versions
        if not runtime_lock.is_file(): atomic_json(runtime_lock,runtime_versions())
        os.environ['HF_HOME']='/content/hf_cache_maximize'
        os.environ['TOKENIZERS_PARALLELISM']='false'
        auth.authenticate_user()
    ''').strip()
    prepare='#@title 2. Chỉ nhận checkpoint đã chốt; không đọc corpus/vector cũ\n'
    prepare+='COLLECT_SOURCE='+repr(collect)+'\nMAXIMIZE_SOURCE='+repr(maximize)+'\n'
    prepare+='TOKENIZER_REVISION='+repr(plan['tokenizer_revision'])+'\nCHUNK_POLICY='+repr(plan['chunking'])+'\n'
    prepare+=dedent('''
        (WORK/'collect_all_btc_sources.py').write_text(COLLECT_SOURCE,encoding='utf-8')
        (WORK/'maximize_collection.py').write_text(MAXIMIZE_SOURCE,encoding='utf-8')
        import collect_all_btc_sources as collector
        import maximize_collection as maximizer
        collector=importlib.reload(collector)
        maximizer=importlib.reload(maximizer)
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        from vietmedbridge.chunks import ChunkConfig,load_bge_tokenizer
        from vietmedbridge.golden import run_golden_suite
        from vietmedbridge.artifacts import verify_file
        credentials,_=google.auth.default()
        reader=collector.CollectionDriveSource(AuthorizedSession(credentials))
        COLLECTION=collector.CollectAll(PROJECT,WORK/'collection')
        SPEC={'model_id':'BAAI/bge-m3','revision':TOKENIZER_REVISION,'special_tokens':False}
        CONFIG=ChunkConfig(**CHUNK_POLICY)
        FLOW=maximizer.MaximizeCollection(COLLECTION,WORK/'delta',tokenizer_spec=SPEC,chunk_config=CONFIG)
        LAZY={}
        def resources():
            if not LAZY:
                print('Load BGE tokenizer only; no GPU/model weights.',flush=True)
                tokenizer,spec=load_bge_tokenizer(TOKENIZER_REVISION)
                if spec!=SPEC: raise ValueError('Tokenizer policy differs from the paid corpus.')
                links=PROJECT/'data_staging/btc-new-71be0e12-20261019-v1/references/5586f423916005c1-links.parquet'
                verify_file(links,COLLECTION.binding['official_sha256'])
                golden=run_golden_suite(CHECKOUT/'tests/golden/cases.json',tokenizer,SPEC,CONFIG)
                if not golden['passed']: raise ValueError('Golden source-span regression failed.')
                LAZY.update(tokenizer=tokenizer,links=links,golden=golden)
            return LAZY
        print(json.dumps(FLOW.status(),ensure_ascii=False,indent=2),flush=True)
    ''')
    ready=dedent('''
        #@title 3. Ưu tiên text đã cứu: chunk/freeze/index-input chỉ phần mới
        needs=FLOW.con.execute("SELECT count(*) FROM todo WHERE state='TEXT_READY' AND coalesce(json_extract(candidate,'$.recovery_root'),'')!='maximize'").fetchone()[0]
        if needs or not FLOW.saved('freeze:metadata-text-v1'):
            r=resources()
            FLOW.chunk_ready(r['tokenizer'],phase='metadata-text-v1')
            RESULT=FLOW.freeze_delta('metadata-text-v1',r['links'],r['tokenizer'],r['golden'])
        else:
            RESULT=FLOW.saved('freeze:metadata-text-v1')
        print(json.dumps(RESULT,ensure_ascii=False,indent=2),flush=True)
        FLOW.export_remaining()
    ''').strip()
    raw=dedent('''
        #@title 4. Tận dụng raw đã crawl: tiếp tục đúng cursor, giữ gate hash/challenge
        try:
            FLOW.recover_raw(reader)
            RESULT=FLOW.saved('freeze:raw-text-v1')
            pending=FLOW.con.execute("SELECT count(*) FROM todo WHERE state='TEXT_READY' AND json_extract(candidate,'$.recovery_root')='maximize'").fetchone()[0]
            if not RESULT and (pending or (FLOW.root/'processed/raw-text-v1/build.json').is_file()):
                r=resources()
                FLOW.chunk_ready(r['tokenizer'],phase='raw-text-v1')
                RESULT=FLOW.freeze_delta('raw-text-v1',r['links'],r['tokenizer'],r['golden'])
            elif RESULT:
                FLOW._publish_delta_manifest()
            print(json.dumps(RESULT,ensure_ascii=False,indent=2),flush=True)
            print(json.dumps(FLOW.export_remaining(),ensure_ascii=False,indent=2),flush=True)
            print('Outputs:',FLOW.root,flush=True)
            print('delta_candidates.json: chỉ dữ liệu mới, cần GPU embedding rồi ghép vào retrieval.',flush=True)
            print('remaining_for_crawl.csv: URL chưa có nội dung đạt gate; gửi nhóm crawl theo nguyên nhân.',flush=True)
        finally:
            FLOW.close()
            COLLECTION.close()
    ''').strip()
    intro=dedent('''
        # 02 — Tận dụng phần còn thiếu từ AI Guru (Colab CPU)

        **CPU → Run all**, một phiên trên cùng output folder. Không đổi cấu hình.
        Cell 3 đưa text đã khôi phục vào dữ liệu chuẩn trước. Cell 4 tiếp tục cứu
        raw đã crawl; có thể ngắt sau checkpoint và Run all tiếp trong phiên mới.

        Không mở lại corpus/vector/BM25 đã xử lý, không đổi candidate luồng submit.
        Dùng cùng tokenizer/chunk policy và producer code đã khóa. Không gọi model
        embedding, không gọi lại website. BGE ở đây chỉ là tokenizer CPU.

        Raw SUCCESS không chứng minh có bài viết: hash/length phải khớp, parser và
        gate nội dung phải qua. Trang challenge, lỗi, retained LOW/QUARANTINE được
        ghi riêng để review/crawl tiếp. Không hứa phục hồi được toàn bộ raw.

        Mỗi source metadata được lập hàng đợi một lần. TAR raw cần một lượt đọc
        header để tìm đúng member; cursor được lưu trên Drive mỗi 1.024 header hoặc
        32 text mới. Bản raw cùng hash + URL + trạng thái review không parse lại.
        Source/cursor đã commit được dùng lại ở runtime khác. Đoạn đang ghi chưa
        commit có thể làm lại khi bị ngắt. Không quét lại cả TAR chỉ để xác minh.

        Kết quả riêng: `VietMedBridge/data_staging/btc-maximize-71be0e12-v1/`.
        `delta_candidates.json`: candidate + model inputs của dữ liệu mới.
        `remaining_for_crawl.csv`: mọi ID còn thiếu, kèm nguyên nhân.
        `raw_recovery_outcomes.csv`: hash/URL đã thử và kết quả thật.
        `checkpoint.json` + SQL snapshot: tiếp tục qua phiên/tài khoản khác.

        Dữ liệu tạm chỉ ở máy Colab, không tải về PC. Không copy nguyên TAR/raw ra
        ổ Colab hoặc Drive; chỉ đọc raw cần thử trong bộ nhớ, lưu text/metadata và
        chunks đã xử lý. Dữ liệu cũ và luồng submit đang chạy được giữ nguyên.
    ''').strip()
    book=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(intro),nb.v4.new_code_cell(setup),
        nb.v4.new_code_cell(prepare),nb.v4.new_code_cell(ready),nb.v4.new_code_cell(raw)])
    book.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}}
    return book


if __name__=='__main__':
    nb.write(notebook(),ROOT/'notebooks/02_colab_maximize_collected_data.ipynb')
