"""Generate separate CPU-transfer and Kaggle GPU notebooks for new frozen inputs."""
import ast
import json
from pathlib import Path
import subprocess
from textwrap import dedent
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
RELEASE = '73b41596103334e341be449cbff57879b90f4d9f'
NAMES = ('new_data_kaggle.py', 'new_data_embeddings.py', 'new_data_cache_lookup.py', 'new_data_drive_api.py')


def vendors():
    def released(path):
        return subprocess.check_output(['git', 'show', RELEASE+':'+path], cwd=ROOT).decode('utf-8')
    api = released('scripts/drive_receipts_api.py')
    old = released('scripts/submission_gpu_progress.py')
    node = next(n for n in ast.parse(old).body if isinstance(n, ast.ClassDef) and n.name == 'ResumableContentBlocks')
    prefix = '''"""Released resumable receipt lookup; model producers are unchanged."""
from concurrent.futures import ThreadPoolExecutor
import gzip
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import time
from vietmedbridge.artifacts import (atomic_json, digest_json, local_workspace, publish_file,
    read_json, sha256_file, utc_now, verify_file)
from vietmedbridge.content_embeddings import ContentBlocks as NativeContentBlocks, checked, seal
from vietmedbridge.scale_vectors import bound
try:
    from .new_data_drive_api import LookupPublishUnavailable
except ImportError:
    from new_data_drive_api import LookupPublishUnavailable
'''
    (ROOT/'scripts/new_data_drive_api.py').write_text(api, encoding='utf-8')
    (ROOT/'scripts/new_data_cache_lookup.py').write_text(prefix+'\n\n'+ast.get_source_segment(old, node)+'\n', encoding='utf-8')


def cell(source):
    return nb.v4.new_code_cell(dedent(source).strip())


def book(title, cells, gpu=False):
    value = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(dedent(title).strip()), *cells])
    value.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                      'language_info': {'name': 'python'}}
    if gpu:
        value.metadata['accelerator'] = 'GPU'
        value.metadata['kaggle'] = {'isInternetEnabled': True, 'enableGpu': True, 'enableTpu': False,
                                    'language': 'python', 'sourceType': 'notebook'}
    return value


def transfer():
    sources = {n: (ROOT/'scripts'/n).read_text(encoding='utf-8') for n in NAMES}
    setup = cell('''
        #@title 1. Colab CPU — chuẩn bị gói Kaggle cho batch 10 nguồn mới
        import copy
        import importlib
        import json
        import os
        from pathlib import Path
        import re
        import subprocess
        import sys
        if not Path('/var/colab/hostname').is_file():
            raise RuntimeError('Notebook chuyển dữ liệu này chạy trên Colab CPU. Kaggle dùng 04_new_data_account_0 hoặc 1.')
        print('NEW DATA: CPU export/import; không chạy lại 02/03, không nạp model weights.', flush=True)
        KAGGLE_ACCOUNTS = 2 #@param {type:"integer"}
        KAGGLE_NOTEBOOK_OUTPUT = "" #@param {type:"string"}
        # Để trống để xuất gói. Sau GPU: owner/notebook[,owner2/notebook2] để nhập Outputs.
        from google.colab import drive, auth, userdata
        drive.mount('/content/drive')
        PROJECT = Path('/content/drive/MyDrive/VietMedBridge')
        relative = 'data_staging/btc-new-71be0e12-20261019-v1'
        if not (PROJECT/relative/'freeze_batch_v1/freeze_batch.json').is_file():
            PROJECT = Path('/content/drive/.shortcut-targets-by-id/1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK/VietMedBridge')
        PROJECT = PROJECT.resolve()
        lock = json.loads((PROJECT/relative/'intake_code_lock.json').read_text())
        REPO_URL = 'https://github.com/Platypus27-coder/VietMedBridge.git'
        if lock.get('repo_url') != REPO_URL or not re.fullmatch('[0-9a-f]{40}', lock.get('commit', '')):
            raise ValueError('Intake producer code lock không hợp lệ.')
        try:
            credentials = json.loads(userdata.get('KAGGLE_JSON'))
            owner, api_key = credentials['username'], credentials['key']
            if not re.fullmatch('[A-Za-z0-9_-]+', owner) or not api_key:
                raise ValueError()
        except Exception:
            raise RuntimeError('Thêm Colab Secret KAGGLE_JSON chứa nội dung kaggle.json và bật Notebook access.') from None
        os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'] = owner, api_key
        del credentials, api_key
        CHECKOUT = Path('/content/VietMedBridge-new-kaggle')
        WORK_DIR = Path('/content/vmb_new_kaggle_cpu')
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        if not (CHECKOUT/'.git').is_dir():
            subprocess.run(['git', 'clone', '--no-checkout', REPO_URL, str(CHECKOUT)], check=True)
        if subprocess.check_output(['git', '-C', str(CHECKOUT), 'remote', 'get-url', 'origin'], text=True).strip() != REPO_URL:
            raise ValueError('Checkout thuộc repo khác.')
        subprocess.run(['git', '-C', str(CHECKOUT), 'fetch', '--depth', '1', 'origin', lock['commit']], check=True)
        subprocess.run(['git', '-C', str(CHECKOUT), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.', 'numpy', 'kaggle>=2.2.2', 'kagglehub'], cwd=CHECKOUT, check=True)
        sys.path[:0] = [str(WORK_DIR), str(CHECKOUT/'src')]
        importlib.invalidate_caches()
        auth.authenticate_user()
    ''')
    prepare = cell('''
        #@title 2. Kết nối manifest frozen và cache đã có — CPU, không mở vector payload
    ''')
    prepare.source += '\nSOURCES = json.loads('+repr(json.dumps(sources, ensure_ascii=False))+')\n'
    prepare.source += dedent('''
        for name, source in SOURCES.items():
            (WORK_DIR/name).write_text(source, encoding='utf-8')
        import new_data_kaggle as pipeline
        from new_data_embeddings import embedding_contract
        from new_data_drive_api import DriveReceiptSource
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        credentials, _ = google.auth.default()
        config, runtime, identities, anchor = embedding_contract(PROJECT/'data', CHECKOUT)
        data = (PROJECT/'data').resolve()
        api_source = DriveReceiptSource(lambda: AuthorizedSession(copy.copy(credentials)), data, data)
        api_source.verify_candidate(anchor)
        print('Runtime GPU khóa từ cache cũ:', runtime, flush=True)
        print('Batch: 10 frozen builds; task còn thiếu được chọn riêng cho từng model.', flush=True)
    ''')
    run = cell('''
        #@title 3. Xuất gói mới / nhập Outputs rồi xuất lượt còn thiếu
        if KAGGLE_NOTEBOOK_OUTPUT.strip():
            from kaggle.api.kaggle_api_extended import KaggleApi
            from vietmedbridge.kaggle_outputs import output_handles
            api = KaggleApi()
            api.authenticate()
            for handle in output_handles(KAGGLE_NOTEBOOK_OUTPUT):
                output = pipeline.download_outputs(api, handle, WORK_DIR, project=PROJECT)
                saved = pipeline.checked(pipeline.read_json(output/'job.json'))
                flow = pipeline.NewDataKaggle(PROJECT, WORK_DIR/'cpu', identities,
                    accounts=saved['team']['accounts'], gpu_count=saved['team']['gpu_count'])
                report = flow.install(output, receipt_api=api_source)
                print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
                KAGGLE_ACCOUNTS = saved['team']['accounts']
        flow = pipeline.NewDataKaggle(PROJECT, WORK_DIR/'cpu', identities, accounts=KAGGLE_ACCOUNTS, gpu_count=2)
        target, job = flow.export(CHECKOUT, config=config, runtime=runtime, code_commit=lock['commit'], receipt_api=api_source)
        if target is None:
            print(json.dumps(job, ensure_ascii=False, indent=2), flush=True)
            if job['state'] == 'WAITING_FOR_KAGGLE_OUTPUTS':
                print('Giữ lượt hiện tại:', 'https://www.kaggle.com/datasets/'+job['dataset_handle'], flush=True)
                print('Nhập Outputs của các acc còn thiếu trước; không xuất lượt chồng lên phiên cũ.', flush=True)
            elif all(v['completed_parts'] == v['total_parts'] for v in job['coverage'].values()):
                print('NEW_DATA_EMBEDDINGS_COMPLETE — BGE và Qwen đủ batch mới. Tiếp theo mới ghép/index.', flush=True)
            else:
                raise RuntimeError('Không còn task nhưng mapping chưa đủ. Giữ checkpoint; chưa xác nhận hoàn tất.')
        else:
            pipeline.upload_job(flow, target, job, owner)
            print('Hai acc Add Input CÙNG dataset trên; chạy 04_new_data_account_0 và 04_new_data_account_1.', flush=True)
            print('Nếu KAGGLE_ACCOUNTS=1: chỉ chạy acc 0. Không Add Input gói 700k cũ.', flush=True)
            print('Giới hạn kế hoạch: 16 GB vector mới/acc/lượt; ROUND_COMPLETE chưa có nghĩa cả batch xong.', flush=True)
    ''')
    return book('''
        # 04 — Xuất data mới sang Kaggle / nhập checkpoint về Drive (Colab CPU)

        **Chạy notebook này trước trên Colab CPU → Run all.** Giữ KAGGLE_ACCOUNTS=2
        nếu dùng hai tài khoản T4 x2; một tài khoản thì chọn 1 ngay trước lần xuất đầu.
        Secret `KAGGLE_JSON` là của người xuất, bật Notebook access. Không gửi key vào chat.

        Nhận batch **10/10 nguồn đã hoàn tất 03**. Giữ model/runtime và text hash cũ.
        Chọn text thiếu bằng lookup SQL; không nạp weights, không đọc vector để kiểm kê.
        Mỗi input part chọn xong lưu draft map + missing-text task + journal trên Drive.
        Khi đổi runtime, phần đã có journal được bỏ qua; chỉ input part chưa commit mới đọc lại.

        Gói Kaggle chứa text thiếu, không copy vector cũ. Hai GPU được phân công theo
        hash để text trùng luôn vào cùng GPU và dùng lại vector vừa tạo. Mỗi lượt
        đặt trần 16 GB kết quả/acc; GPU có deadline để kịp lưu Outputs trước khi runtime ngắt.
        Có thể cần nhiều lượt. Đây là embedding child data mới, chưa ghép catalog/index,
        chưa xử lý phần Qwen cũ còn thiếu và chưa xuất submission.

        Sau GPU: điền `KAGGLE_NOTEBOOK_OUTPUT="owner/notebook,owner2/notebook2"` → Run all.
        Nhập payload đã lưu và kiểm SHA/model/shape trước khi ghi coverage. Tự xuất lượt
        còn thiếu; không đọc lại corpus đã chọn. Vector vào `data/model_cache`, progress
        ở `data_staging/btc-new-71be0e12-20261019-v1/freeze_batch_v1/embeddings_v1`.
        Các file tạm nằm trong runtime Colab, không tải corpus về máy cá nhân.
        Luồng submit và active candidate được giữ nguyên. Chỉ một phiên CPU xuất/nhập cùng team.
    ''', [setup, prepare, run])


def recovery_transfer():
    value = transfer()
    value.cells[0].source = dedent('''
        # 04 — Xuất phần text vừa cứu sang Kaggle (Colab CPU)

        **Colab CPU → Run all.** Secret `KAGGLE_JSON` của người xuất, bật Notebook access.
        Nhận tất cả candidate đã freeze trong maximize `delta_candidates.json`.
        Không đọc TAR, không chạy lại 02/03, không nạp model weights.

        Xuất chỉ text còn thiếu vector của BGE-M3 và Qwen3-Embedding-8B.
        Hai tài khoản dùng CÙNG dataset được in ở `NEW_DATASET_READY` và hai bản
        `04_new_data_account_0.ipynb`, `04_new_data_account_1.ipynb` trên Kaggle.
        Import notebook → Add Input → Internet ON → T4 x2 → Save & Run All.

        Sau GPU: điền `KAGGLE_NOTEBOOK_OUTPUT="owner/notebook,owner2/notebook2"`
        rồi Run all để nhập checkpoint. Gói này chỉ chứa các batch text đã cứu;
        không khẳng định coverage embedding toàn bộ kho đã đủ. Raw chưa đọc và
        active submission không bị thay đổi. Scratch chỉ nằm trong máy Colab.
    ''').strip()
    setup = value.cells[1].source
    setup = setup.replace('chuẩn bị gói Kaggle cho batch 10 nguồn mới', 'xuất text đã cứu; không đọc raw TAR')
    setup = setup.replace("relative = 'data_staging/btc-new-71be0e12-20261019-v1'",
                          "relative = 'data_staging/btc-maximize-71be0e12-v1'")
    setup = setup.replace("PROJECT/relative/'freeze_batch_v1/freeze_batch.json'", "PROJECT/relative/'delta_candidates.json'")
    setup = setup.replace("lock = json.loads((PROJECT/relative/'intake_code_lock.json').read_text())",
                          "lock = {'repo_url':'https://github.com/Platypus27-coder/VietMedBridge.git','commit':'bbf09ec01d542a5b06a4c2beec245571abeb3515'}")
    setup = setup.replace('/content/VietMedBridge-new-kaggle', '/content/VietMedBridge-recovery-kaggle')
    setup = setup.replace('/content/vmb_new_kaggle_cpu', '/content/vmb_recovery_kaggle_cpu')
    value.cells[1].source = setup
    prepare = value.cells[2].source
    prepare = prepare.replace('import new_data_kaggle as pipeline',
        "for name in SOURCES:\n    sys.modules.pop(Path(name).stem, None)\nimport new_data_kaggle as pipeline")
    prepare = prepare.replace("print('Batch: 10 frozen builds; task còn thiếu được chọn riêng cho từng model.', flush=True)", '')
    prepare += '\nRECOVERY_SOURCE=' + repr((ROOT/'scripts/recovery_kaggle_batch.py').read_text(encoding='utf-8')) + '\n'
    prepare += dedent('''
        (WORK_DIR/'recovery_kaggle_batch.py').write_text(RECOVERY_SOURCE,encoding='utf-8')
        sys.modules.pop('recovery_kaggle_batch',None)
        from recovery_kaggle_batch import load_recovery_batch
        BATCH_ROOT,BATCH,_=load_recovery_batch(PROJECT)
        BATCH_DIR=BATCH_ROOT.relative_to(PROJECT).as_posix()
        def recovery_flow(work,*,accounts=2,gpu_count=2,batch_dir=None):
            return pipeline.NewDataKaggle(PROJECT,work,identities,accounts=accounts,gpu_count=gpu_count,
                batch_loader=lambda project:load_recovery_batch(project,batch_dir or BATCH_DIR))
        print('RECOVERY_BATCH:',BATCH['documents'],'documents |',BATCH['input_rows_before_cross_batch_dedup'],
            'input rows before cross-batch text dedup |',len(BATCH['builds']),'frozen builds',flush=True)
    ''')
    value.cells[2].source = prepare
    value.cells[3].source = cell('''
        #@title 3. Xuất recovery dataset / nhập Outputs về Drive
        if KAGGLE_NOTEBOOK_OUTPUT.strip():
            from kaggle.api.kaggle_api_extended import KaggleApi
            from vietmedbridge.kaggle_outputs import output_handles
            api=KaggleApi()
            api.authenticate()
            roots=[p.relative_to(PROJECT).as_posix() for p in (PROJECT/relative/'embedding_batches').glob('*/embeddings_v1')]
            for handle in output_handles(KAGGLE_NOTEBOOK_OUTPUT):
                output=pipeline.download_outputs(api,handle,WORK_DIR,project=PROJECT,import_roots=roots)
                saved=pipeline.checked(pipeline.read_json(output/'job.json'))
                if 'source_batch_dir' not in saved:
                    raise ValueError('Outputs này chưa khóa recovery batch; dùng đúng notebook/gói vừa xuất.')
                flow=recovery_flow(WORK_DIR/'cpu',accounts=saved['team']['accounts'],
                    gpu_count=saved['team']['gpu_count'],batch_dir=saved['source_batch_dir'])
                print(json.dumps(flow.install(output,receipt_api=api_source),ensure_ascii=False,indent=2),flush=True)
                KAGGLE_ACCOUNTS=saved['team']['accounts']
        flow=recovery_flow(WORK_DIR/'cpu',accounts=KAGGLE_ACCOUNTS)
        target,job=flow.export(CHECKOUT,config=config,runtime=runtime,code_commit=lock['commit'],receipt_api=api_source)
        if target is None:
            print(json.dumps(job,ensure_ascii=False,indent=2),flush=True)
        else:
            pipeline.upload_job(flow,target,job,owner)
            print('Hai acc dùng CÙNG dataset này: 04_new_data_account_0 và 04_new_data_account_1.',flush=True)
        print('Recovery vectors only. Phần raw chưa đọc và luồng submit giữ nguyên.',flush=True)
    ''').source
    return value


def full_transfer():
    value = recovery_transfer()
    value.cells[0].source = dedent('''
        # 04 — Toàn bộ dữ liệu đã xử lý: baseline + 10 nguồn + mọi recovery batch

        **Colab CPU → Run all**, `KAGGLE_ACCOUNTS=2`, Outputs để trống khi xuất.
        Bật Notebook access cho Secret `KAGGLE_JSON`. Không chạy lại 02/03.

        Trước GPU phải có `FULL_SCOPE_READY` với **ít nhất 802.028 ID BTC**.
        Notebook nhận mọi model input đã freeze, giữ cả nguồn legacy đã được
        yêu cầu tận dụng. Số document rows hoặc text không phải số ID BTC khác nhau.
        Tạo manifest membership từ hai SQL metadata checkpoint; không đọc TAR,
        không chunk lại và không mở vector payload để kiểm kê.

        BGE/Qwen đã có manifest COMPLETE được kế thừa bằng metadata. Phần còn
        thiếu dùng lookup cache theo text hash; input part đã chọn được bỏ qua
        khi chạy tiếp. Gói mỗi lượt có trần 16 GB vector mới/acc: xem rõ scheduled
        và unscheduled tasks. Toàn bộ hàng đợi được giữ trên Drive, không bỏ task.

        Khi có `NEW_DATASET_READY`: hai acc Add Input CÙNG dataset, chạy hai bản
        `04_new_data_account_0.ipynb` / `04_new_data_account_1.ipynb`, Internet ON,
        T4 x2, Save & Run All. Sau GPU điền hai handle Outputs rồi Run all notebook
        CPU này để nhập và xuất lượt còn thiếu. FULL_SCOPE_READY là đủ đầu vào;
        chỉ FULL_DATA_EMBEDDINGS_COMPLETE mới xác nhận đủ vector cho mọi nguồn.
    ''').strip()
    value.cells[1].source = value.cells[1].source.replace('xuất text đã cứu; không đọc raw TAR', 'xuất TOÀN BỘ dữ liệu đã freeze')
    value.cells[1].source = value.cells[1].source.replace('/content/VietMedBridge-recovery-kaggle', '/content/VietMedBridge-full-kaggle')
    value.cells[1].source = value.cells[1].source.replace('/content/vmb_recovery_kaggle_cpu', '/content/vmb_full_kaggle_cpu')
    prepare = value.cells[2].source.split('\nRECOVERY_SOURCE=')[0]
    for name in ('recovery_kaggle_batch.py', 'full_btc_embeddings.py'):
        prepare += '\n(WORK_DIR/'+repr(name)+').write_text('+repr((ROOT/'scripts'/name).read_text(encoding='utf-8'))+",encoding='utf-8')\n"
        prepare += 'sys.modules.pop('+repr(Path(name).stem)+',None)\n'
    prepare += dedent('''
        from full_btc_embeddings import prepare_full_scope,load_full_batch,PARENT
        SCOPE_ROOT,SCOPE=prepare_full_scope(PROJECT,CHECKOUT,WORK_DIR/'metadata',minimum_ids=802028)
        BATCH_DIR=SCOPE_ROOT.relative_to(PROJECT).as_posix()
        def full_flow(work,*,accounts=2,gpu_count=2,batch_dir=None):
            return pipeline.NewDataKaggle(PROJECT,work,identities,accounts=accounts,gpu_count=gpu_count,
                batch_loader=lambda project:load_full_batch(project,batch_dir or BATCH_DIR,minimum_ids=802028))
        print(json.dumps({k:SCOPE[k] for k in ('state','official_ids','official_total','reused_ids','new_chunked_ids',
            'source_builds','source_documents_sum','input_rows_before_cross_source_dedup')},ensure_ascii=False,indent=2),flush=True)
        print('Membership report:',SCOPE_ROOT/'scope.json',flush=True)
    ''')
    value.cells[2].source = prepare
    value.cells[3].source = cell('''
        #@title 3. TOÀN BỘ scope: xuất text thiếu / nhập Outputs rồi tiếp tục hàng đợi
        if KAGGLE_NOTEBOOK_OUTPUT.strip():
            from kaggle.api.kaggle_api_extended import KaggleApi
            from vietmedbridge.kaggle_outputs import output_handles
            api=KaggleApi()
            api.authenticate()
            roots=[p.relative_to(PROJECT).as_posix() for p in (PROJECT/PARENT).glob('*/embeddings_v1')]
            for handle in output_handles(KAGGLE_NOTEBOOK_OUTPUT):
                output=pipeline.download_outputs(api,handle,WORK_DIR,project=PROJECT,import_roots=roots)
                saved=pipeline.checked(pipeline.read_json(output/'job.json'))
                if saved.get('coverage_scope',{}).get('official_ids',0)<802028:
                    raise ValueError('Outputs này không thuộc full-data scope; nhập bằng notebook xuất tương ứng.')
                flow=full_flow(WORK_DIR/'cpu',accounts=saved['team']['accounts'],
                    gpu_count=saved['team']['gpu_count'],batch_dir=saved['source_batch_dir'])
                print(json.dumps(flow.install(output,receipt_api=api_source),ensure_ascii=False,indent=2),flush=True)
                KAGGLE_ACCOUNTS=saved['team']['accounts']
        flow=full_flow(WORK_DIR/'cpu',accounts=KAGGLE_ACCOUNTS)
        target,job=flow.export(CHECKOUT,config=config,runtime=runtime,code_commit=lock['commit'],receipt_api=api_source)
        if target is None:
            print(json.dumps(job,ensure_ascii=False,indent=2),flush=True)
            if job['state']=='NO_PENDING_KAGGLE_TASKS' and all(v['completed_parts']==v['total_parts'] for v in job['coverage'].values()):
                print('FULL_DATA_EMBEDDINGS_COMPLETE — mọi nguồn trong full scope đủ BGE và Qwen.',flush=True)
        else:
            print('FULL_SCOPE_ROUND:',job['coverage_scope'],'| scheduled',job['scheduled_tasks'],
                '| still queued',job['unscheduled_tasks'],'| all scheduled',job['all_remaining_tasks_scheduled'],flush=True)
            pipeline.upload_job(flow,target,job,owner)
            print('Hai acc dùng CÙNG dataset; account_0 và account_1. Hàng đợi còn thiếu lưu trên Drive.',flush=True)
    ''').source
    return value


def gpu(account_id=0, *, expected_scope=None):
    setup = cell('''
        # Kaggle only. This cell prints before any install or download.
        import hashlib
        import importlib
        import json
        import os
        from pathlib import Path
        import re
        import subprocess
        import sys
        import time
        SESSION_STARTED = time.monotonic()
        KAGGLE_ACCOUNT_ID = __ACCOUNT__
        SESSION_HOURS = 10.0
        print('Kaggle input preflight | account', KAGGLE_ACCOUNT_ID, '| metadata only; no model weights', flush=True)
        if Path('/var/colab/hostname').is_file() or not Path('/kaggle/input').is_dir():
            raise RuntimeError('Notebook GPU này chạy trên Kaggle. Colab CPU dùng 04_colab_export_new_data_kaggle.')
        jobs = {}
        for path in Path('/kaggle/input').rglob('job.json'):
            value = json.loads(path.read_text())
            if value.get('schema') == 'new-data-kaggle-v1':
                jobs[value.get('manifest_sha256')] = path.parent
        if len(jobs) != 1:
            raise RuntimeError('Add Input đúng MỘT dataset NEW_DATASET_READY từ notebook xuất data mới. Gỡ gói 700k/continuation cũ.')
        JOB_DIR = next(iter(jobs.values()))
        JOB = json.loads((JOB_DIR/'job.json').read_text())
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        if JOB.get('manifest_sha256') != digest({k:v for k,v in JOB.items() if k != 'manifest_sha256'}):
            raise ValueError('Job header changed.')
        if JOB.get('repo_url') != 'https://github.com/Platypus27-coder/VietMedBridge.git' or not re.fullmatch('[0-9a-f]{40}', JOB.get('code_commit', '')):
            raise ValueError('Job producer repo/commit invalid.')
        if not 0 <= KAGGLE_ACCOUNT_ID < JOB['team']['accounts'] or JOB['team']['gpu_count'] != 2:
            raise ValueError('Account ID không thuộc team đã xuất; không tự đổi team trong notebook GPU.')
        expected_sources = {'new_data_kaggle.py','new_data_embeddings.py','new_data_cache_lookup.py','new_data_drive_api.py'}
        if set(JOB['worker_sources']) != expected_sources:
            raise ValueError('Missing worker sources.')
        for name, checksum in JOB['worker_sources'].items():
            if hashlib.sha256((JOB_DIR/name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Worker source checksum changed.')
        CHECKOUT = Path('/kaggle/temp/VietMedBridge-new-data')
        WORK_DIR = Path('/kaggle/temp/vmb_new_data')
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(WORK_DIR/'hf_cache')
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        if not (CHECKOUT/'.git').is_dir():
            subprocess.run(['git','clone','--no-checkout',JOB['repo_url'],str(CHECKOUT)], check=True)
        if subprocess.check_output(['git','-C',str(CHECKOUT),'remote','get-url','origin'], text=True).strip() != JOB['repo_url']:
            raise ValueError('Checkout belongs to another repository.')
        subprocess.run(['git','-C',str(CHECKOUT),'fetch','--depth','1','origin',JOB['code_commit']], check=True)
        subprocess.run(['git','-C',str(CHECKOUT),'checkout','--detach','FETCH_HEAD'], check=True)
        print('Installing pinned encoder dependencies; no Google Drive mount on Kaggle.', flush=True)
        subprocess.run([sys.executable,'-m','pip','install','-e','.[strong]'], cwd=CHECKOUT, check=True)
        sys.path[:0] = [str(JOB_DIR),str(CHECKOUT/'src')]
        importlib.invalidate_caches()
        from vietmedbridge.portable_embeddings import runtime_install_commands
        for command in runtime_install_commands(JOB['runtime']):
            subprocess.run(command, check=True)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(CHECKOUT/'src')+os.pathsep+str(JOB_DIR)
        probe = "import torch; print('FRESH CUDA:',torch.__version__,torch.cuda.is_available(),torch.cuda.device_count(),flush=True); assert torch.cuda.is_available() and torch.cuda.device_count()==2"
        subprocess.run([sys.executable,'-u','-c',probe], env=env, check=True)
        print('READY | batch',JOB['batch'],'| team',JOB['team'],'| account',KAGGLE_ACCOUNT_ID,flush=True)
    '''.replace('__ACCOUNT__', str(account_id)))
    # A CPU-only first cell works against existing immutable uploaded datasets.
    # Native worker/model files are still loaded from the sealed job itself.
    marker = "CHECKOUT = Path('/kaggle/temp/VietMedBridge-new-data')"
    preflight, bootstrap = setup.source.split(marker, 1)
    preflight += '\n'+(ROOT/'scripts/kaggle_input_preflight.py').read_text(encoding='utf-8')
    preflight += '\nPREFLIGHT = inspect_job_inputs(JOB_DIR, JOB, expected_scope='+repr(expected_scope)+')\n'
    setup = cell(preflight)
    bootstrap = cell("# GPU setup: run only after reviewing the input inventory.\n"
        +"subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], check=True)\n"
        +marker+bootstrap)
    run = cell('''
        # Each T4 runs an independent fresh Python worker; output stays in /kaggle/working.
        elapsed = (time.monotonic()-SESSION_STARTED)/3600
        if elapsed >= SESSION_HOURS-0.1:
            raise RuntimeError('Bootstrap used the save deadline; restart a fresh saved version before encoding.')
        command = [sys.executable,'-u',str(JOB_DIR/'new_data_kaggle.py'),'launch',
            '--job',str(JOB_DIR),'--checkout',str(CHECKOUT),
            '--output','/kaggle/working/vmb_new_data','--work',str(WORK_DIR/'workers'),
            '--account-id',str(KAGGLE_ACCOUNT_ID),'--hours',str(SESSION_HOURS-elapsed)]
        subprocess.run(command, env=env, check=True)
        print('Save & Run All: chờ version hoàn tất rồi nhập Outputs về Drive bằng notebook Colab CPU.',flush=True)
    ''')
    return book(f'''
        # 04 — Embedding data mới: Kaggle acc {account_id}

        **Add Input dataset NEW_DATASET_READY → Internet ON → GPU T4 x2 → Save Version / Save & Run All.**
        Acc 0 và acc 1 dùng CÙNG dataset; chủ dataset cấp quyền đọc cho acc còn lại.
        Đã đặt KAGGLE_ACCOUNT_ID={account_id}. Nếu gói xuất chỉ một acc, chỉ dùng bản acc 0.
        Model BGE-M3 và Qwen3-Embedding-8B, runtime/producer từ job; không mount Drive.

        Chỉ chạy text thiếu trong lượt đã giao. Mỗi task tối đa 512 text có checkpoint
        trong /kaggle/working. Deadline 10 giờ từ cell đầu để kịp lưu Outputs; quota
        tài khoản và ngắt runtime có thể khiến lượt chưa hoàn tất. ROUND_COMPLETE chỉ
        là hết việc của lượt này. Nhập Outputs về Drive để xuất lượt còn thiếu.
        Chỉ checkpoint của version đã lưu thành công mới chuyển qua runtime khác được.
        Không dùng dataset hay Outputs của luồng Qwen 700k cũ ở notebook này.
    ''', [setup, bootstrap, run], gpu=True)


def full_gpu(account_id=0):
    # Observed FULL_SCOPE_READY manifest. All continuation rounds of this scope
    # share it; a 700k/3,080-only/ten-source dataset is rejected before weights.
    value = gpu(account_id, expected_scope='0310f6d51a13bdff63cb9d9dfe337d9aee8898df9f34857c84a2708d9f941721')
    value.cells[0].source = dedent(f'''
        # 04 — Full-data embedding: Kaggle account {account_id}

        Add Input cùng dataset `vmb-full-…` đã được Colab xác minh.
        Account ID đã đặt sẵn: {account_id}. Hai tài khoản dùng cùng một dataset.

        **Kiểm tra trước:** có thể dùng Kaggle CPU, chỉ chạy cell 1.
        Cell này đọc job metadata và kiểm tên file task trên Kaggle; không mở Drive,
        không quét lại corpus/vector cũ, không cài hay nạp model weights.
        Cần thấy INPUT_INVENTORY_COMPLETE với 802.028 ID và 13 nguồn.
        Báo cáo có số task/text rows theo model và tài khoản; text rows có thể trùng
        giữa nguồn/model, không phải số URL hoặc số text khác nhau.

        Khi quyết định chạy nguyên text đã freeze: chọn T4 x2, Internet ON,
        Save Version / Save & Run All. Kiểm kê file không sửa hoặc chứng nhận
        chất lượng ngữ nghĩa của text; các lỗi nội dung đã biết vẫn giữ nguyên.

        Dataset là một lượt giới hạn dung lượng output, chỉ chứa text thiếu.
        ROUND_COMPLETE không có nghĩa toàn scope hoàn tất. Chờ version lưu Outputs
        thành công rồi nhập về Drive bằng 04_colab_export_full_data_kaggle.ipynb.
    ''').strip()
    return value


def main():
    vendors()
    nb.write(transfer(), ROOT/'notebooks/04_colab_export_new_data_kaggle.ipynb')
    nb.write(recovery_transfer(), ROOT/'notebooks/04_colab_export_recovered_data_kaggle.ipynb')
    try:
        from .write_full_queue_notebooks import full_queue_transfer, full_queue_gpu
    except ImportError:
        from write_full_queue_notebooks import full_queue_transfer, full_queue_gpu
    nb.write(full_queue_transfer(), ROOT/'notebooks/04_colab_export_full_data_kaggle.ipynb')
    for account in range(2):
        nb.write(gpu(account), ROOT/f'notebooks/kaggle/04_new_data_account_{account}.ipynb')
        nb.write(full_queue_gpu(account), ROOT/f'notebooks/kaggle/04_full_data_account_{account}.ipynb')


if __name__ == '__main__':
    main()
