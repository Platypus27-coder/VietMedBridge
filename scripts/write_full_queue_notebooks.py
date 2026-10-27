"""Keep all text inputs on Kaggle; subsequent rounds only upload a small plan."""
from pathlib import Path
from textwrap import dedent, indent
import nbformat as nb

try:
    from .write_new_data_kaggle_notebooks import ROOT, book, cell, full_transfer, gpu
except ImportError:
    from write_new_data_kaggle_notebooks import ROOT, book, cell, full_transfer, gpu

SCOPE_SHA = '0310f6d51a13bdff63cb9d9dfe337d9aee8898df9f34857c84a2708d9f941721'
FULL_LIBRARY_HANDLE = 'kaisercoding/vmb-full-inputs-d6ab9a432ac6-1792733353195313801'
SOURCE_EXECUTION_PLAN = '63b6fe57cbf42fbf355f4d1c476a1fa68801c4b91d03c65855274f35861dec15'
FULL_LIBRARY_JOB = 'd6ab9a432ac64e5aadb39a826b7dc2176094cc66d4f4e3d9e17c121e071c0711'


def full_queue_transfer():
    value = full_transfer()
    value.cells[0].source = dedent('''
        # 04 — Toàn bộ text thiếu lên Kaggle một lần; kế hoạch GPU riêng

        **Colab CPU → Run all**, KAGGLE_ACCOUNTS=2, Outputs trống khi xuất lần đầu.
        Giữ Secret KAGGLE_JSON đã dùng. Không chạy lại 02/03, không chunk/index lại.

        Notebook dùng cùng 802.028 ID BTC / 13 nguồn đã chốt và cùng journal chọn
        text còn thiếu. Gói all-text chứa toàn bộ hàng đợi text đã chọn, không bị
        cắt theo dung lượng vector. BGE/Qwen/cache/model code giữ nguyên.
        Các file missing Parquet có sẵn được đóng gói một lần theo nguồn/model;
        task dùng offset trong cùng file, tránh tạo lại hàng chục nghìn file nhỏ.

        Colab in hai link: **ALL_TEXT_DATASET_READY** và **GPU_PLAN_DATASET_READY**.
        Hai tài khoản Kaggle Add Input CÙNG hai dataset đó, dùng
        04_full_data_account_0.ipynb / 04_full_data_account_1.ipynb.
        Dataset text giữ nguyên qua các lượt. Kế hoạch GPU có giới hạn dự kiến
        16 GB vector mới/acc/lượt và deadline lưu Outputs; đây là cấu hình của
        pipeline. Nhiều lượt GPU vẫn có thể cần thiết.

        Sau GPU, điền hai handle vào KAGGLE_NOTEBOOK_OUTPUT rồi Run all notebook
        CPU này. Nhập Outputs hợp lệ trước, tạo kế hoạch nhỏ mới loại phần đã xong,
        dùng lại dataset text. Chờ cả hai acc có Outputs nhập về để đổi kế hoạch.
        ROUND_COMPLETE chỉ xác nhận kế hoạch của acc, chưa phải toàn bộ dữ liệu.

        MIGRATE_UNSTARTED_LEGACY=True chỉ dùng khi chưa chạy GPU gói bounded cũ.
        Lần chuyển này đã được chủ dự án xác nhận. Job, checkpoint và dataset cũ
        vẫn được giữ để đối chiếu/nhập Outputs, không xóa active candidate.

        Đóng gói không sửa nội dung trích xuất. Các cờ LOW và lỗi nội dung đã biết
        vẫn giữ nguyên; số 802.028 là phạm vi đã có text, chưa phải toàn bộ BTC.
    ''').strip()
    value.cells[1].source = value.cells[1].source.replace(
        'KAGGLE_ACCOUNTS = 2 #@param {type:"integer"}',
        'KAGGLE_ACCOUNTS = 2 #@param {type:"integer"}\n'
        'MIGRATE_UNSTARTED_LEGACY = True #@param {type:"boolean"}')
    value.cells[1].source = value.cells[1].source.replace("'kagglehub'", "'kagglehub==1.0.3'")
    prepare = value.cells[2].source.replace(
        'import new_data_kaggle as pipeline',
        "for module_name in ('new_data_kaggle','new_data_embeddings','new_data_cache_lookup','new_data_drive_api'):\n"
        '    sys.modules.pop(module_name,None)\n'
        'importlib.invalidate_caches()\nimport new_data_kaggle as pipeline')
    source = (ROOT/'scripts/full_queue_embeddings.py').read_text(encoding='utf-8')
    prepare += '\n(WORK_DIR/\'full_queue_embeddings.py\').write_text('+repr(source)+",encoding='utf-8')\n"
    source = (ROOT/'scripts/qwen_token_batching.py').read_text(encoding='utf-8')
    prepare += '\n(WORK_DIR/\'qwen_token_batching.py\').write_text('+repr(source)+",encoding='utf-8')\n"
    source = (ROOT/'scripts/kaggle_output_transfer.py').read_text(encoding='utf-8')
    prepare += '\n(WORK_DIR/\'kaggle_output_transfer.py\').write_text('+repr(source)+",encoding='utf-8')\n"
    source = (ROOT/'scripts/kaggle_output_paged.py').read_text(encoding='utf-8')
    prepare += '\n(WORK_DIR/\'kaggle_output_paged.py\').write_text('+repr(source)+",encoding='utf-8')\n"
    prepare += dedent('''
        sys.modules.pop('full_queue_embeddings',None)
        sys.modules.pop('kaggle_output_transfer',None)
        sys.modules.pop('kaggle_output_paged',None)
        for module_name in tuple(sys.modules):
            if module_name == 'kagglehub' or module_name.startswith('kagglehub.'):
                sys.modules.pop(module_name,None)
        importlib.invalidate_caches()
        from kaggle_output_paged import install_paged_output_transfer
        install_paged_output_transfer()
        import full_queue_embeddings as queue
        def queue_flow(work,*,accounts=2,gpu_count=2,batch_dir=None):
            return queue.FullQueueKaggle(PROJECT,work,identities,accounts=accounts,gpu_count=gpu_count,
                batch_loader=lambda project:load_full_batch(project,batch_dir or BATCH_DIR,minimum_ids=802028))
    ''')
    value.cells[2].source = prepare
    value.cells[3].source = cell('''
        #@title 3. Xuất TOÀN BỘ text một lần / nhập kết quả rồi xuất kế hoạch nhỏ
        if KAGGLE_NOTEBOOK_OUTPUT.strip():
            from kaggle.api.kaggle_api_extended import KaggleApi
            from vietmedbridge.kaggle_outputs import output_handles
            api=KaggleApi()
            api.authenticate()
            roots=[p.relative_to(PROJECT).as_posix() for p in (PROJECT/PARENT).glob('*/embeddings_v1')]
            for handle in output_handles(KAGGLE_NOTEBOOK_OUTPUT):
                output=pipeline.download_outputs(api,handle,WORK_DIR,project=PROJECT,import_roots=roots,allow_failed=True,
                    manifest_path='vmb_full_queue/result/new-data-result.json')
                saved=pipeline.checked(pipeline.read_json(output/'job.json'))
                if saved.get('coverage_scope',{}).get('official_ids',0)<802028:
                    raise ValueError('Outputs không thuộc scope full-data; dùng notebook xuất tương ứng.')
                factory=queue_flow if saved.get('schema')=='full-data-text-library-v1' else full_flow
                flow=factory(WORK_DIR/'cpu',accounts=saved['team']['accounts'],
                    gpu_count=saved['team']['gpu_count'],batch_dir=saved['source_batch_dir'])
                print(json.dumps(flow.install(output,receipt_api=api_source),ensure_ascii=False,indent=2),flush=True)
                KAGGLE_ACCOUNTS=saved['team']['accounts']
        flow=queue_flow(WORK_DIR/'cpu',accounts=KAGGLE_ACCOUNTS)
        target,library=flow.export_inputs(CHECKOUT,config=config,runtime=runtime,
            code_commit=lock['commit'],receipt_api=api_source)
        if library.get('state')=='NO_PENDING_KAGGLE_TASKS':
            print(json.dumps(library,ensure_ascii=False,indent=2),flush=True)
            if all(v['completed_parts']==v['total_parts'] for v in library['coverage'].values()):
                print('FULL_DATA_EMBEDDINGS_COMPLETE',flush=True)
        else:
            library_handle=queue.upload_library(flow,target,library,owner)
            plan_target,plan=flow.create_plan(library,receipt_api=api_source,
                supersede_unstarted_legacy=MIGRATE_UNSTARTED_LEGACY)
            print('ALL_TEXT_DATASET_READY:','https://www.kaggle.com/datasets/'+library_handle,flush=True)
            print('ALL_INPUTS:',library['scheduled_tasks'],'tasks | not exported:',library['unscheduled_tasks'],flush=True)
            if plan_target is None:
                print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
                if plan.get('dataset_handle'):
                    label='CURRENT_PLAN_WAITING_FOR_OUTPUTS:' if plan.get('state')=='WAITING_FOR_KAGGLE_OUTPUTS' else 'GPU_PLAN_DATASET_READY:'
                    print(label,'https://www.kaggle.com/datasets/'+plan['dataset_handle'],flush=True)
                if plan.get('state')=='WAITING_FOR_KAGGLE_OUTPUTS':
                    print('Chưa tạo lượt mới. Cần nhập Outputs acc:',
                        sorted(set(plan['required_accounts'])-set(plan['imported_accounts'])),flush=True)
                if plan.get('state')=='NO_PENDING_KAGGLE_TASKS':
                    if not all(v['completed_parts']==v['total_parts'] for v in plan['coverage'].values()):
                        raise RuntimeError('Task hết nhưng mapping chưa đủ; giữ checkpoint, chưa xác nhận hoàn tất.')
                    print('FULL_DATA_EMBEDDINGS_COMPLETE',flush=True)
            else:
                plan_handle=queue.upload_plan(flow,plan_target,plan,owner)
                print('GPU_PLAN_DATASET_READY:','https://www.kaggle.com/datasets/'+plan_handle,flush=True)
                print('Hai acc Add Input CÙNG thư viện text và CÙNG kế hoạch GPU. Lượt sau giữ thư viện text, thay kế hoạch.',flush=True)
    ''').source
    return value


def full_queue_gpu(account_id=0, expected_accounts=2):
    preflight = cell('''
        #@title 1. Kiểm tra toàn bộ thư viện text + kế hoạch (CPU được, chưa nạp model)
        import hashlib
        import importlib
        import json
        import os
        from pathlib import Path
        import re
        import subprocess
        import sys
        import time
        SESSION_STARTED=time.monotonic()
        KAGGLE_ACCOUNT_ID=__ACCOUNT__
        EXPECTED_EXECUTION_ACCOUNTS=__EXPECTED_ACCOUNTS__
        SESSION_HOURS=11.5
        print('ALL-TEXT Kaggle preflight | account',KAGGLE_ACCOUNT_ID,flush=True)
        if not Path('/kaggle/input').is_dir() or Path('/var/colab/hostname').is_file():
            raise RuntimeError('GPU notebook chạy trên Kaggle; Colab dùng notebook xuất full-data.')
        libraries,plans={},{}
        for path in Path('/kaggle/input').rglob('job.json'):
            item=json.loads(path.read_text())
            if item.get('schema')=='full-data-text-library-v1':
                libraries[item.get('manifest_sha256')]=(path.parent,item)
            elif item.get('schema') in ('full-data-execution-plan-v1','full-data-execution-plan-v2'):
                plans[item.get('manifest_sha256')]=(path,item)
        if len(libraries)!=1:
            raise RuntimeError('Add Input đúng MỘT dataset ALL_TEXT_DATASET_READY; gỡ gói 1.84 GB/700k cũ khỏi notebook này.')
        JOB_DIR,JOB=next(iter(libraries.values()))
        compatible=[(path,plan) for path,plan in plans.values() if plan.get('library_job')==JOB['manifest_sha256']]
        if len(compatible)!=1:
            raise RuntimeError('Add Input đúng MỘT GPU_PLAN_DATASET_READY khớp thư viện text; gỡ kế hoạch cũ khi đổi lượt.')
        PLAN_PATH,PLAN=compatible[0]
        if PLAN.get('team')!={'accounts':EXPECTED_EXECUTION_ACCOUNTS,'gpu_count':2}:
            raise ValueError('Dùng kế hoạch GPU mới đúng số acc đã cấu hình; gỡ kế hoạch 2 acc cũ trước khi chạy GPU.')
        if JOB.get('repo_url')!='https://github.com/Platypus27-coder/VietMedBridge.git' or not re.fullmatch('[0-9a-f]{40}',JOB.get('code_commit','')):
            raise ValueError('Library producer repo/commit invalid.')
        if not 0<=KAGGLE_ACCOUNT_ID<PLAN['team']['accounts'] or PLAN['team']['gpu_count']!=2:
            raise ValueError('Account ID/GPU team differs from exported library.')
        expected_sources={'new_data_kaggle.py','new_data_embeddings.py','new_data_cache_lookup.py',
            'new_data_drive_api.py','full_queue_embeddings.py'}
        if set(JOB['worker_sources'])!=expected_sources:
            raise ValueError('Missing locked library/worker source.')
        EXECUTOR_DIR=PLAN_PATH.parent if PLAN['schema']=='full-data-execution-plan-v2' else JOB_DIR
        if PLAN['schema']=='full-data-execution-plan-v2':
            expected_executors={'full_queue_embeddings.py','new_data_kaggle.py'}
            if PLAN.get('qwen_execution') in ('qwen-token-batches-v1','qwen-token-batches-v2'):
                expected_executors.add('qwen_token_batching.py')
            if set(PLAN.get('executor_sources',{}))!=expected_executors:
                raise ValueError('Missing locked execution scheduler sources.')
            for name,checksum in PLAN['executor_sources'].items():
                if hashlib.sha256((EXECUTOR_DIR/name).read_bytes()).hexdigest()!=checksum:
                    raise ValueError('Execution scheduler checksum changed.')
        if PLAN.get('qwen_execution') not in ('qwen-token-batches-v1','qwen-token-batches-v2'):
            SESSION_HOURS=min(SESSION_HOURS,10.0)
            print('LEGACY_EXECUTION_PLAN: giữ deadline cũ; bản tối ưu cần kế hoạch mới sau khi nhập đủ Outputs.',flush=True)
        else:
            print('QWEN_TOKEN_BATCHING_READY: benchmark trên GPU; deadline 11.5 giờ tính từ cell đầu.',flush=True)
    '''.replace('__ACCOUNT__', str(account_id)).replace('__EXPECTED_ACCOUNTS__', str(expected_accounts)))
    preflight.source += '\n'+(ROOT/'scripts/kaggle_input_preflight.py').read_text(encoding='utf-8')
    preflight.source += '\nPREFLIGHT=inspect_full_queue_inputs(JOB_DIR,JOB,PLAN,expected_scope='+repr(SCOPE_SHA)+')\n'
    bootstrap = gpu(account_id).cells[2].source
    bootstrap += "\nenv['PYTHONPATH']=os.pathsep.join((str(CHECKOUT/'src'),str(EXECUTOR_DIR),str(JOB_DIR)))\n"
    bootstrap += "print('EXECUTION_TEAM:',PLAN['team'],'| account',KAGGLE_ACCOUNT_ID,flush=True)\n"
    run = cell('''
        #@title 3. Chạy phần được giao; log gọn và lưu checkpoint vào Outputs
    ''')
    run.source += '\n' + (ROOT/'scripts/kaggle_output_guard.py').read_text(encoding='utf-8')
    run.source += '\n' + cell('''
        elapsed=(time.monotonic()-SESSION_STARTED)/3600
        if elapsed>=SESSION_HOURS-0.1:
            raise RuntimeError('Start a fresh saved version; bootstrap used the save deadline.')
        command=[sys.executable,'-u',str(EXECUTOR_DIR/'full_queue_embeddings.py'),'launch',
            '--job',str(JOB_DIR),'--plan',str(PLAN_PATH),'--checkout',str(CHECKOUT),
            '--output','/kaggle/working/vmb_full_queue','--work',str(WORK_DIR/'workers'),
            '--account-id',str(KAGGLE_ACCOUNT_ID),'--hours',str(SESSION_HOURS-elapsed)]
        GPU_RESULT=run_with_bounded_output(command,env=env,
            log_path=Path('/kaggle/working/vmb_full_queue/gpu-worker.log'),
            result_path=Path('/kaggle/working/vmb_full_queue/result/new-data-result.json'),
            job=JOB['manifest_sha256'],plan=PLAN['manifest_sha256'],account=KAGGLE_ACCOUNT_ID,
            report=lambda message:print(message,flush=True))
        print('Chờ Save & Run All lưu Outputs hoàn tất, rồi nhập về Drive bằng notebook full-data CPU.',flush=True)
    ''').source
    return book(f'''
        # 04 — Toàn bộ text trên Kaggle, kế hoạch GPU riêng: account {account_id}

        Add Input **ALL_TEXT_DATASET_READY** và **GPU_PLAN_DATASET_READY**.
        Các tài khoản dùng cùng hai dataset; account ID đã đặt sẵn {account_id}.
        Bản này yêu cầu kế hoạch {expected_accounts} acc, mỗi acc T4 x2; gỡ kế hoạch cũ trước khi chạy.
        Chủ dataset cấp quyền đọc cả hai gói cho các tài khoản còn lại.

        Có thể chọn CPU và chỉ chạy cell 1 để kiểm đủ file thư viện và kế hoạch.
        Cell này không nạp weights hoặc mở Drive. ALL_TEXT_INPUTS_PRESENT xác nhận
        toàn bộ text task đã đóng gói hiện diện; số task của kế hoạch có thể ít hơn.

        Chạy GPU: T4 x2, Internet ON, Save Version / Save & Run All. Mỗi GPU có
        journal theo đúng library+plan; task xong được giữ, text hash trùng được dùng
        lại. Runtime/quota hoặc deadline có thể khiến kế hoạch còn dở. Outputs phải
        lưu thành công trước khi nhập về Drive. Sau khi nhập cả hai tài khoản,
        Colab tạo kế hoạch nhỏ mới; giữ dataset text và thay dataset kế hoạch.

        Log đầy đủ nằm ở vmb_full_queue/gpu-worker.log trong Outputs. Màn hình
        chỉ hiện tiến độ và GPU_RUN_SUMMARY để tránh IOPub timeout vì JSON quá lớn.
        Không sửa worker khóa trong dataset, không xuất lại thư viện text.
        Nếu process_state=FAILED nhưng có partial manifest hợp lệ, notebook vẫn
        kết thúc để giữ Outputs; đó chưa phải hoàn tất embedding.

        Giữ nguyên text đã freeze, gồm cả cờ LOW. Kiểm kê file không sửa chất lượng
        ngữ nghĩa. Đây là corpus child embedding, chưa phải query/index/submission.
    ''', [preflight, cell(bootstrap), run], gpu=True)


def three_account_transfer():
    value = full_queue_transfer()
    value.cells[0].source = dedent('''
        # 04 — Chuyển phần còn thiếu sang 3 acc Kaggle / 6 GPU

        **Chạy notebook này trên Colab CPU → Run all.** Số acc chạy đã đặt sẵn 3.
        Thư viện text 5,5 GB vẫn giữ nguyên, kể cả khóa producer/team gốc.
        Chỉ xuất kế hoạch nhỏ mới và hai file điều phối đã khóa SHA; không chọn
        lại text, không chunk/crawl/index lại và không tải vector cũ để kiểm kê.

        Chủ dự án đã xác nhận kế hoạch ab7954 chưa chạy GPU. Notebook chỉ được
        thay đúng kế hoạch đó khi chưa có biên nhận nhập kết quả của nó. Hai
        biên nhận cũ và toàn bộ task đã nhập được giữ trong cùng journal.

        Ngân sách dự kiến 18 GB/acc, chừa dung lượng dưới mức 20 GB Outputs theo
        [tài liệu Kaggle](https://www.kaggle.com/docs/notebooks). Đây là giới hạn
        lưu Outputs; không cam kết một phiên sẽ xử lý hết hàng đợi còn lại.

        Colab in **ALL_TEXT_DATASET_READY** (thư viện cũ) và
        **GPU_PLAN_DATASET_READY** (kế hoạch mới). Cả ba acc Add Input cùng hai
        gói; dùng 04_full_data_3acc_account_0 / 1 / 2, account ID đã đặt sẵn.
        Chọn T4 x2, Internet ON, Save Version → Save & Run All.

        Kế hoạch mới khóa bộ tối ưu qwen-token-batches-v2. Trên GPU, chỉ chọn
        batch token nhanh hơn ít nhất 5% và vượt canary so với native batch 16;
        nếu không thì giữ native. Deadline 11,5 giờ từ cell đầu, dành khoảng
        30 phút trước giới hạn 12 giờ. Không cam kết số task hay mức tăng tốc.

        Sau khi ba phiên lưu Outputs xong, điền ba handle notebook vào
        KAGGLE_NOTEBOOK_OUTPUT rồi chạy lại notebook CPU này để nhập và tạo
        lượt tiếp theo. Không tự chuyển team thư viện gốc sang 3.
    ''').strip()
    value.cells[1].source = value.cells[1].source.replace(
        'KAGGLE_ACCOUNTS = 2 #@param {type:"integer"}',
        'KAGGLE_ACCOUNTS = 2  # Thư viện gốc; giữ để tiếp tục đúng checkpoint.\n'
        'EXECUTION_ACCOUNTS = 3  # Ba acc, mỗi acc T4 x2.\n'
        'CONFIRMED_UNSTARTED_PLAN = None  # Chỉ tạo lượt mới sau khi nhập đủ Outputs.')
    value.cells[3].source = value.cells[3].source.replace(
        'print(json.dumps(flow.install(output,receipt_api=api_source),ensure_ascii=False,indent=2),flush=True)',
        'report=flow.install(output,receipt_api=api_source)\n'
        "        print(json.dumps({k:v for k,v in report.items() if k!='selection_ids'},ensure_ascii=False,indent=2),flush=True)")
    value.cells[3].source = value.cells[3].source.replace(
        'supersede_unstarted_legacy=MIGRATE_UNSTARTED_LEGACY)',
        'supersede_unstarted_legacy=MIGRATE_UNSTARTED_LEGACY,\n'
        '        execution_accounts=EXECUTION_ACCOUNTS,output_bytes=queue.ELASTIC_OUTPUT_BYTES,\n'
        '        replace_unstarted_plan=CONFIRMED_UNSTARTED_PLAN)')
    value.cells[3].source = value.cells[3].source.replace(
        "print('Hai acc Add Input CÙNG thư viện text và CÙNG kế hoạch GPU. Lượt sau giữ thư viện text, thay kế hoạch.',flush=True)",
        "print('Ba acc dùng cùng hai dataset; account 0/1/2 đã đặt sẵn trong notebook GPU.',flush=True)")
    value.cells[3].source = value.cells[3].source.replace(
        "plan_handle=queue.upload_plan(flow,plan_target,plan,owner)\n        print('GPU_PLAN_DATASET_READY:',",
        "plan_handle=queue.upload_plan(flow,plan_target,plan,owner)\n"
        "        print('GPU_PLAN_ASSIGNMENTS:',{a:sum(plan['task_workers'][t]//2==a for t in plan['tasks']) for a in range(EXECUTION_ACCOUNTS)},flush=True)\n"
        "        print('GPU_PLAN_DATASET_READY:',")
    return value


def four_account_transfer():
    value = three_account_transfer()
    value.cells[0].source = dedent('''
        # 04 — Tiếp tục phần còn thiếu trên 4 acc Kaggle / 8 GPU

        Chạy **Colab CPU → Run all**. Đã đặt sẵn 4 acc, mỗi acc T4 x2.
        Các Outputs trước đã nhập Drive thì để KAGGLE_NOTEBOOK_OUTPUT trống.
        KAGGLE_ACCOUNTS=2 là khóa thư viện gốc; EXECUTION_ACCOUNTS=4 là
        đội chạy lượt mới. Không sửa khóa thư viện để đổi số acc.

        Notebook đọc biên nhận đã lưu để loại task đã nhập của mọi acc,
        dùng key index để chọn phần text còn thiếu và xuất kế hoạch nhỏ.
        Dataset text 5,5 GB dùng lại; không crawl/chunk/index lại hoặc
        tải vector cũ để kiểm kê. Mỗi task trong kế hoạch chỉ có một GPU
        phụ trách; bốn acc dùng các worker 0–7 và cùng một kế hoạch.

        Nếu lượt cũ chưa nhập đủ Outputs, notebook giữ lượt đó và báo
        WAITING_FOR_KAGGLE_OUTPUTS. Không tự thay một kế hoạch có thể
        đang chạy. Retry sau lỗi upload dùng lại đúng kế hoạch đã chốt.

        Khi có GPU_PLAN_DATASET_READY, cả bốn acc Add Input thư viện text
        cũ và kế hoạch mới, cấp quyền đọc hai dataset cho các thành viên.
        Dùng 04_full_data_4acc_account_0 / 1 / 2 / 3, ID đã đặt sẵn.
        T4 x2, Internet ON → Save Version → Save & Run All.

        Kế hoạch có bộ tối ưu qwen-token-batches-v2 và deadline 11,5 giờ.
        Chỉ chọn nhánh đo được nhanh hơn và vượt kiểm tra sai số; không
        cam kết tốc độ tăng bốn lần hoặc hoàn thành toàn bộ trong một lượt.
        Ngân sách Outputs dự kiến 18 GB/acc vẫn áp dụng cho từng acc.

        GPU_PLAN_ASSIGNMENTS báo số task của từng acc 0–3. Sau lượt chạy,
        nhập đủ bốn Outputs trước khi tạo lượt kế tiếp. Task và URL là
        hai đơn vị khác nhau; đây là embedding corpus, chưa phải submission.
    ''').strip()
    value.cells[1].source = value.cells[1].source.replace(
        'EXECUTION_ACCOUNTS = 3  # Ba acc, mỗi acc T4 x2.',
        'EXECUTION_ACCOUNTS = 4  # Bốn acc 0/1/2/3, mỗi acc T4 x2.')
    value.cells[3].source = value.cells[3].source.replace(
        'Ba acc dùng cùng hai dataset; account 0/1/2 đã đặt sẵn trong notebook GPU.',
        'Bốn acc dùng cùng hai dataset; account 0/1/2/3 đã đặt sẵn trong notebook GPU.')
    return value


def completed_output_import():
    value = four_account_transfer()
    value.cells[0].source = dedent('''
        # 04 — Nhập kết quả Kaggle đã hoàn thành về Drive (Colab CPU)

        Acc 0 đã đặt sẵn: **kaisercoding/impact-3**, đúng manifest 1.198 task
        của plan 63b6fe57cbf4. Link người chạy cung cấp có scriptVersionId
        362515679; đây là ID phiên Kaggle, không phải số version 1/2/3.
        Chạy trên Colab CPU → Run all, dùng Secret KAGGLE_JSON đã có.
        Notebook chỉ nhập Outputs; không xuất thư viện hoặc tạo kế hoạch GPU.

        Trước khi tải vector, kiểm tra đúng mã kết quả 00541f1dea44 và lấy
        hai báo cáo qwen-tuning nhỏ từ cùng version đã khóa. In
        QWEN_TUNING_REPORT để biết mode, batch, kernel, tốc độ đo và lý do
        các lựa chọn bị loại; lưu báo cáo trên Drive để đọc sau. Báo cáo
        thiếu không làm mất kết quả vector hợp lệ. Không chạy model/GPU.

        File đang tải và checkpoint tải được lưu trong
        VietMedBridge/data_staging/kaggle_output_transfers trên Drive để runtime
        mới dùng lại. Không tải dữ liệu về máy tính cá nhân. Chỉ file còn thiếu
        được tải; SHA được kiểm trước khi nhập cache native. Chỉ khi có
        NEW_DATA_KAGGLE_IMPORTED mới tính task là đã nhập Drive hoàn tất.

        Thư viện, candidate và kế hoạch đang chạy của các acc khác giữ nguyên.
    ''').strip()
    value.cells[1].source = value.cells[1].source.replace(
        'KAGGLE_NOTEBOOK_OUTPUT = ""', 'KAGGLE_NOTEBOOK_OUTPUT = "kaisercoding/impact-3"')
    source = (ROOT/'scripts/kaggle_qwen_diagnostics.py').read_text(encoding='utf-8')
    value.cells[2].source += '\n(WORK_DIR/\'kaggle_qwen_diagnostics.py\').write_text('+repr(source)+",encoding='utf-8')\n"
    value.cells[2].source += "sys.modules.pop('kaggle_qwen_diagnostics',None)\nimportlib.invalidate_caches()\n"
    value.cells[3].source = cell('''
        #@title 3. Nhập acc 0 về Drive; không tạo hay thay kế hoạch GPU
        from kaggle.api.kaggle_api_extended import KaggleApi
        from vietmedbridge.kaggle_outputs import output_handles
        from kaggle_qwen_diagnostics import read_qwen_tuning
        if not KAGGLE_NOTEBOOK_OUTPUT.strip():
            raise ValueError('Điền handle notebook có Outputs đã lưu.')
        api=KaggleApi()
        api.authenticate()
        download_work=PROJECT/'data_staging/kaggle_output_transfers'
        download_work.mkdir(parents=True,exist_ok=True)
        roots=[p.relative_to(PROJECT).as_posix() for p in (PROJECT/PARENT).glob('*/embeddings_v1')]
        for handle in output_handles(KAGGLE_NOTEBOOK_OUTPUT):
            expected=None
            if handle.split('/')[:2]==['kaisercoding','impact-2']:
                expected={
                    'manifest_sha256':'11e88194e4e65edeabc66608f1c8cd30c35d1552089d60bb42110ad246653e79',
                    'job':'d6ab9a432ac64e5aadb39a826b7dc2176094cc66d4f4e3d9e17c121e071c0711',
                    'execution_plan':'c8d94f5f75f79ac89bd110b74b33befc3cbcf519be4fed2cf67be4df58fe7f56',
                    'account_id':0,'completed_tasks':918}
            if handle.split('/')[:2]==['kaisercoding','impact-3']:
                expected={
                    'manifest_sha256':'00541f1dea440b984f647aff3161b076ea24e3ef6501d534cfcd50c6e73316a4',
                    'job':'d6ab9a432ac64e5aadb39a826b7dc2176094cc66d4f4e3d9e17c121e071c0711',
                    'execution_plan':'63b6fe57cbf42fbf355f4d1c476a1fa68801c4b91d03c65855274f35861dec15',
                    'account_id':0,'completed_tasks':1198}
            def observer(transport,result):
                if result.get('plan',{}).get('qwen_execution') in ('qwen-token-batches-v1','qwen-token-batches-v2'):
                    read_qwen_tuning(transport,result,PROJECT,identities['qwen']['encoder'])
                else:
                    print('QWEN_TUNING_REPORTS_NOT_PRODUCED: legacy execution plan; native import continues.',flush=True)
            output=pipeline.download_outputs(api,handle,download_work,project=PROJECT,
                import_roots=roots,allow_failed=True,
                manifest_path='vmb_full_queue/result/new-data-result.json',expected_result=expected,
                on_manifest=observer)
            saved=pipeline.checked(pipeline.read_json(output/'job.json'))
            if (saved.get('schema')!='full-data-text-library-v1'
                    or saved.get('coverage_scope',{}).get('official_ids')!=802028):
                raise ValueError('Outputs không thuộc thư viện full-data 802.028 ID đã chốt.')
            flow=queue_flow(WORK_DIR/'cpu',accounts=saved['team']['accounts'],
                gpu_count=saved['team']['gpu_count'],batch_dir=saved['source_batch_dir'])
            report=flow.install(output,receipt_api=api_source)
            print(json.dumps({k:v for k,v in report.items() if k!='selection_ids'},
                ensure_ascii=False,indent=2),flush=True)
            print('DRIVE_IMPORT_COMPLETE:',handle,'| tasks',len(report['selection_ids']),flush=True)
        print('Đã nhập kết quả. Chưa tạo kế hoạch mới; giữ phiên các acc còn lại.',flush=True)
    ''').source
    return value


def two_account_continuation():
    value=three_account_transfer()
    value.cells[0].source=dedent('''
        # 04 — Chạy tiếp acc 0/1 ngay; giữ riêng phần acc 2 chưa có thông tin

        Chạy trên **Colab CPU → Run all**. Đã đặt sẵn hai acc, mỗi acc T4 x2,
        và source plan c8d94f. Không cần chờ acc 2. Chỉ một phiên CPU tạo kế hoạch.

        Acc 0/1 phải có biên nhận nhập Outputs trên Drive. Phần đã nhập được
        loại trước khi chia việc. Toàn bộ task source plan giao acc 2 được ghi
        reservation riêng, không giao lại cho acc 0/1. Kế hoạch/Outputs cũ giữ
        nguyên để khi acc 2 về có thể nhập sau. Những task acc 0/1 chưa xong vẫn
        được đưa vào lượt tiếp theo, cùng phần thư viện chưa từng được giao.

        Chỉ đọc manifest và key index đã lưu; không đọc corpus, chunk lại hay
        kiểm kê vector payload. Thư viện text 5,5 GB giữ nguyên. Colab chỉ xuất
        kế hoạch nhỏ mới và in GPU_PLAN_DATASET_READY, dùng hai notebook
        04_continue_2acc_account_0 / 1. Cả hai Add Input cùng thư viện cũ và
        kế hoạch mới, T4 x2, Internet ON → Save & Run All. Cấp quyền đọc kế
        hoạch mới cho acc 1. Bộ tối ưu Qwen v2 đã khóa trong kế hoạch mới.

        Chạy lại cùng notebook dùng kế hoạch đã chốt; không tự tạo lượt trùng.
        Chờ nhập đủ Outputs acc 0/1 của lượt mới để cấp lượt kế tiếp; không
        chờ acc 2. Khi acc 2 đã nhập, phần nó chưa hoàn thành trở lại hàng đợi.
    ''').strip()
    value.cells[1].source=value.cells[1].source.replace(
        'EXECUTION_ACCOUNTS = 3  # Ba acc, mỗi acc T4 x2.',
        'EXECUTION_ACCOUNTS = 2  # Hai acc 0/1, mỗi acc T4 x2; acc 2 giữ riêng.')
    value.cells[1].source+='\nCONTINUE_FROM_PLAN="c8d94f5f75f79ac89bd110b74b33befc3cbcf519be4fed2cf67be4df58fe7f56"\n'
    value.cells[3].source=value.cells[3].source.replace(
        'replace_unstarted_plan=CONFIRMED_UNSTARTED_PLAN)',
        'replace_unstarted_plan=CONFIRMED_UNSTARTED_PLAN,continue_completed_from=CONTINUE_FROM_PLAN)')
    value.cells[3].source=value.cells[3].source.replace(
        'Ba acc dùng cùng hai dataset; account 0/1/2 đã đặt sẵn trong notebook GPU.',
        'Hai acc 0/1 dùng cùng thư viện cũ và kế hoạch mới; phần acc 2 giữ riêng.')
    value.cells[3].source=value.cells[3].source.replace(
        "print('GPU_PLAN_ASSIGNMENTS:',",
        "print('RESERVED_PENDING_ACCOUNTS:',plan.get('reservations',[]),'| tasks',len(plan.get('reserved_task_ids',[])),flush=True)\n"
        "        print('GPU_PLAN_ASSIGNMENTS:',")
    return value


def qwen_benchmark_notebook():
    preflight=cell('''
        #@title 1. Kaggle T4 x2 — dùng thư viện text cũ, không cần kế hoạch mới
        import hashlib,importlib,json,os,re,subprocess,sys,time
        from pathlib import Path
        if not Path('/kaggle/input').is_dir() or Path('/var/colab/hostname').is_file():
            raise RuntimeError('Notebook đo tốc độ này chạy trên Kaggle T4 x2.')
        KAGGLE_ACCOUNT_ID=0
        libraries={}
        for path in Path('/kaggle/input').rglob('job.json'):
            item=json.loads(path.read_text())
            if item.get('schema')=='full-data-text-library-v1':
                libraries[item.get('manifest_sha256')]=(path.parent,item)
        if len(libraries)!=1:
            raise RuntimeError('Add Input đúng một thư viện text vmb-full-inputs-d6ab9a432ac6; không cần xuất dataset mới.')
        JOB_DIR,JOB=next(iter(libraries.values()))
        body={k:v for k,v in JOB.items() if k!='manifest_sha256'}
        sha=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        if (sha!=JOB['manifest_sha256'] or sha!='d6ab9a432ac64e5aadb39a826b7dc2176094cc66d4f4e3d9e17c121e071c0711'
                or JOB.get('repo_url')!='https://github.com/Platypus27-coder/VietMedBridge.git'
                or not re.fullmatch('[0-9a-f]{40}',JOB.get('code_commit',''))):
            raise ValueError('Thư viện hoặc producer không khớp gói đã freeze.')
        print('QWEN_BENCHMARK_INPUT_READY: thư viện cũ; tối đa ba task slice, không ghi task hoàn thành.',flush=True)
    ''')
    bootstrap=gpu(0).cells[2].source
    bootstrap+='\n(WORK_DIR/\'qwen_token_batching.py\').write_text('+repr(
        (ROOT/'scripts/qwen_token_batching.py').read_text(encoding='utf-8'))+",encoding='utf-8')\n"
    bootstrap+="env['PYTHONPATH']=os.pathsep.join((str(WORK_DIR),str(CHECKOUT/'src'),str(JOB_DIR)))\n"
    run=cell('''
        #@title 3. Đo độc lập hai GPU; kết quả nhỏ trong Outputs
        output=Path('/kaggle/working/vmb_qwen_benchmark')
        output.mkdir(parents=True,exist_ok=True)
        processes=[]
        started=time.monotonic()
        try:
            for worker in (0,1):
                child=env.copy()
                child['CUDA_VISIBLE_DEVICES']=str(worker)
                command=[sys.executable,'-u',str(WORK_DIR/'qwen_token_batching.py'),
                    '--job',str(JOB_DIR),'--checkout',str(CHECKOUT),'--output',str(output),
                    '--worker-id',str(worker)]
                processes.append(subprocess.Popen(command,env=child))
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None,0) for p in processes):
                    raise RuntimeError('Benchmark worker lỗi; xem traceback, giữ model cache trong runtime.')
                if time.monotonic()-started>2400:
                    raise RuntimeError('Benchmark vượt 40 phút; kiểm tra tải model/runtime trước khi chạy corpus.')
                print('QWEN_BENCHMARK_HEARTBEAT:',int(time.monotonic()-started),'s',flush=True)
                time.sleep(15)
            if any(p.returncode!=0 for p in processes):
                raise RuntimeError('Benchmark chưa hoàn tất trên cả hai GPU.')
        finally:
            for p in processes:
                if p.poll() is None: p.terminate()
            for p in processes:
                try: p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill();p.wait(timeout=10)
        for worker in (0,1):
            report=json.loads((output/f'worker-{worker}.json').read_text())
            print('QWEN_BENCHMARK_GPU:',worker,'| native texts/s',report.get('baseline_texts_per_second'),
                '| selected texts/s',report.get('selected_texts_per_second'),
                '| sample speedup',report.get('measured_sample_speedup'),'| kernel',report.get('kernel'),flush=True)
        print('BENCHMARK_COMPLETE_NO_TASKS_COMMITTED: chỉ đo tốc độ, chưa encode hàng đợi và không đổi checkpoint Drive.',flush=True)
    ''')
    return book('''
        # 04 — Đo tốc độ Qwen trên acc 0 trong lúc acc 1/2 vẫn chạy

        Kaggle → Add Input thư viện cũ **vmb-full-inputs-d6ab9a432ac6-1792733353195313801**
        → Internet ON → T4 x2 → Save & Run All. Không cần kế hoạch GPU mới,
        không nhập/gộp/chạy lại task cũ. Không dùng GPU acc 1/2 đang chạy.

        Hai GPU đo riêng native batch 16, nhóm theo token và nhánh FP16 math SDPA
        nếu runtime hỗ trợ. Mọi lựa chọn phải nhanh hơn ít nhất 5%, vượt cosine
        >=0,99999 và sai số tuyệt đối <=0,001, rồi xác nhận bằng phép đo thứ hai.
        Nhánh nhanh không đạt sẽ bị loại. Weights NF4, text, tokenizer và pooling
        gốc giữ nguyên; lựa chọn attention trung gian được ghi rõ trong báo cáo.

        Notebook đọc tối đa ba task slice từ thư viện text đã có, không đọc Drive,
        vector cũ hoặc toàn bộ corpus. Tải model/runtime vẫn cần thiết khi phiên
        mới. Không ghi task vào hàng đợi đã hoàn thành. Outputs chỉ có hai JSON
        đo tốc độ và log; gửi các dòng QWEN_BENCHMARK_GPU để chốt cấu hình.
    ''',[preflight,cell(bootstrap),run],gpu=True)


def colab_gpu_plan_notebook():
    value = full_queue_transfer()
    value.cells[1].source = value.cells[1].source.replace(
        '#@title 1. Colab CPU — xuất TOÀN BỘ dữ liệu đã freeze',
        '#@title 1. Colab CPU — kết nối Drive và tải mã khóa')
    value.cells[1].source = value.cells[1].source.replace(
        "print('NEW DATA: CPU export/import; không chạy lại 02/03, không nạp model weights.', flush=True)",
        "print('COLAB PLAN: dùng manifest frozen trên Drive; không export corpus hay nạp model.', flush=True)")
    value.cells[0].source = dedent('''
        # Colab CPU — tạo hàng đợi embedding cho một phiên Colab, một GPU

        Chạy một lần trên Colab CPU. Kế hoạch mới giữ riêng task của Kaggle acc 2
        đang chạy trong plan 63b6fe57cbf4; phần của các acc đã nhập và task chưa
        được giao sẽ sang một Colab worker. Notebook chỉ đọc manifest/key index đã
        lưu trên Drive để lập lịch; không encode, không đọc lại corpus.

        Trước khi chạy, thêm Colab Secret `KAGGLE_JSON`, bật Notebook access,
        rồi Run all. Có thể điền handle Outputs acc 2 nếu phiên đã lưu xong.
        Kế hoạch và file thực thi khóa SHA được ghi trên Drive. Không chạy lại cell
        này sau khi worker đã bắt đầu để giữ nguyên task assignment.
    ''').strip()
    value.cells[1].source = value.cells[1].source.replace(
        'KAGGLE_ACCOUNTS = 2 #@param {type:"integer"}',
        'SOURCE_KAGGLE_ACCOUNTS = 4  # Original 63b6 plan team; Colab runs one account')
    value.cells[3].source = cell(f'''
        #@title 3. Tạo plan Colab 1×1 GPU; giữ riêng task Kaggle acc 2 đang chạy
        CONTINUE_FROM_PLAN = "{SOURCE_EXECUTION_PLAN}"
        EXPECTED_LIBRARY_JOB = "{FULL_LIBRARY_JOB}"
        EXECUTION_ACCOUNTS = 1
        EXECUTION_GPUS_PER_ACCOUNT = 1

        if KAGGLE_NOTEBOOK_OUTPUT.strip():
            from kaggle.api.kaggle_api_extended import KaggleApi
            from vietmedbridge.kaggle_outputs import output_handles
            api=KaggleApi()
            api.authenticate()
            roots=[p.relative_to(PROJECT).as_posix() for p in (PROJECT/PARENT).glob('*/embeddings_v1')]
            download_work=PROJECT/'data_staging/kaggle_output_transfers'
            download_work.mkdir(parents=True,exist_ok=True)
            for handle in output_handles(KAGGLE_NOTEBOOK_OUTPUT):
                output=pipeline.download_outputs(api,handle,download_work,project=PROJECT,import_roots=roots,
                    allow_failed=True,manifest_path='vmb_full_queue/result/new-data-result.json')
                saved=pipeline.checked(pipeline.read_json(output/'job.json'))
                result=pipeline.checked(pipeline.read_json(output/'new-data-result.json'))
                if result.get('execution_plan')!=CONTINUE_FROM_PLAN or result.get('account_id')!=2:
                    raise ValueError('Chỉ nhập Outputs acc 2 của plan đang chạy 63b6; không nhập nhầm lượt khác.')
                if saved.get('manifest_sha256')!=EXPECTED_LIBRARY_JOB:
                    raise ValueError('Outputs thuộc input library khác; giữ nguyên checkpoint.')
                old_flow=queue_flow(WORK_DIR/'cpu',accounts=4,gpu_count=2,batch_dir=saved['source_batch_dir'])
                print(json.dumps(old_flow.install(output,receipt_api=api_source),ensure_ascii=False,indent=2),flush=True)

        flow=queue_flow(WORK_DIR/'cpu',accounts=4,gpu_count=2,batch_dir=BATCH_DIR)
        pointer=flow.root/'input_library.json'
        if not pointer.is_file():
            raise FileNotFoundError('Drive chưa có full input library đã freeze; dừng để không tạo lại từ corpus.')
        library=queue._validate_library_header(queue.read_json(pointer))
        if library.get('manifest_sha256')!=EXPECTED_LIBRARY_JOB:
            raise ValueError('Drive full input library khác job đã chốt; không tạo plan mới.')
        active_path=flow.root/'active_execution_plan.json'
        active=queue.checked(queue.read_json(active_path))
        if active.get('plan')==CONTINUE_FROM_PLAN:
            target,plan=flow.create_plan(library,receipt_api=api_source,
                execution_accounts=EXECUTION_ACCOUNTS,execution_gpu_count=EXECUTION_GPUS_PER_ACCOUNT,
                continue_completed_from=CONTINUE_FROM_PLAN)
        else:
            plan=flow.saved_plan(active.get('plan'))
            if (plan.get('team')!={{'accounts':EXECUTION_ACCOUNTS,'gpu_count':EXECUTION_GPUS_PER_ACCOUNT}}
                    or plan.get('continued_from_plan')!=CONTINUE_FROM_PLAN):
                raise RuntimeError('Drive đang có plan Colab khác. Không đổi assignment khi worker có thể đã chạy.')
            target=flow.stage_plan(plan)
            print('COLAB_PLAN_REUSED: không tạo task assignment mới.',flush=True)
        print('COLAB_PLAN_READY:',plan['manifest_sha256'],flush=True)
        print('TEAM:',plan['team'],'| tasks:',plan['scheduled_tasks'],'| task IDs còn khóa Kaggle:',
            len(plan.get('reserved_task_ids',[])),'| source pending accounts:',plan.get('reservations',[]),flush=True)
        print('PLAN_ON_DRIVE:',flow.root/'plans'/(plan['manifest_sha256']+'.json'),flush=True)
        print('Dùng đúng một Colab GPU: ACCOUNT_ID=0; checkpoint riêng trên Drive.',flush=True)
    ''').source
    return value


def colab_gpu_worker_notebook():
    value = full_queue_transfer()
    value.cells[1].source = value.cells[1].source.replace(
        '#@title 1. Colab CPU — xuất TOÀN BỘ dữ liệu đã freeze',
        '#@title 1. Colab GPU — kết nối Drive và tải mã khóa')
    value.cells[1].source = value.cells[1].source.replace(
        "print('NEW DATA: CPU export/import; không chạy lại 02/03, không nạp model weights.', flush=True)",
        "print('COLAB GPU WORKER: dùng thư viện text frozen; không export hay quét corpus.', flush=True)")
    value.cells[0].source = dedent(f'''
        # Colab GPU worker — embedding phần task được giao, checkpoint trực tiếp lên Drive

        Dùng một phiên Colab GPU. Đặt `KAGGLE_ACCOUNT_ID=0` và `COLAB_PLAN_SHA`
        đúng giá trị notebook tạo plan đã in. Worker chạy toàn bộ whitelist được
        giao cho phiên này; khởi động lại cùng plan/account sẽ đọc journal trên
        Drive và tiếp tục, không tạo assignment mới.

        Đặt Colab runtime GPU (T4), bật Internet, thêm Secret `KAGGLE_JSON`, bật
        Notebook access, rồi Run all. Text library/model cache được tải vào runtime
        Colab; vector blocks, map, result manifest và task journal nằm trong
        `VietMedBridge/data_staging/colab_gpu_checkpoints` trên Drive.
        Không chạy cell tạo plan ở đây hoặc chạy worker này song song ở phiên khác.
    ''').strip()
    value.cells[1].source = value.cells[1].source.replace(
        'KAGGLE_ACCOUNTS = 2 #@param {type:"integer"}',
        'SOURCE_KAGGLE_ACCOUNTS = 4  # Original 63b6 plan team; Colab runs one account')
    value.cells[3].source = cell(f'''
        #@title 3. Chạy một worker Colab GPU; journal/checkpoint trên Drive
        import importlib
        import kagglehub
        import subprocess
        import time

        LIBRARY_HANDLE = "{FULL_LIBRARY_HANDLE}"
        EXPECTED_LIBRARY_JOB = "{FULL_LIBRARY_JOB}"
        SOURCE_PLAN = "{SOURCE_EXECUTION_PLAN}"
        KAGGLE_ACCOUNT_ID = 0 #@param {{type:"integer"}}
        COLAB_PLAN_SHA = "" #@param {{type:"string"}}
        WORKER_HOURS = 11.0 #@param {{type:"number"}}

        if KAGGLE_ACCOUNT_ID!=0 or not re.fullmatch(r'[0-9a-f]{{64}}',COLAB_PLAN_SHA):
            raise ValueError('Dùng ACCOUNT_ID=0 và COLAB_PLAN_SHA chính xác từ notebook tạo plan.')
        gpu_probe=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
            check=True,text=True,capture_output=True)
        if len([line for line in gpu_probe.stdout.splitlines() if line.strip()])!=1:
            raise RuntimeError('Chọn Colab GPU có đúng một GPU trước khi chạy worker.')
        print('Tải sealed ALL_TEXT_DATASET:',LIBRARY_HANDLE,flush=True)
        JOB_DIR=Path(kagglehub.dataset_download(LIBRARY_HANDLE))
        JOB=queue.validate_library(JOB_DIR,CHECKOUT)
        if JOB.get('manifest_sha256')!=EXPECTED_LIBRARY_JOB:
            raise ValueError('ALL_TEXT_DATASET sai manifest; không khởi tạo embedding.')
        from vietmedbridge.portable_embeddings import embedding_runtime,runtime_install_commands
        for command in runtime_install_commands(JOB['runtime']):
            print('COLAB_RUNTIME_INSTALL:',command[-1],flush=True)
            subprocess.run(command,check=True)
        subprocess.run([sys.executable,'-m','pip','install','pyarrow==21.0.0',
            'tokenizers==0.22.2','huggingface-hub==0.36.0','accelerate==1.12.0'],check=True)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Runtime khóa đã cài nhưng Colab không còn đúng một CUDA GPU.')
        if embedding_runtime()!=JOB['runtime']:
            raise RuntimeError('Không cài được runtime PyTorch/Transformers/BitsAndBytes đã khóa.')
        flow=queue_flow(WORK_DIR/'cpu',accounts=4,gpu_count=2,batch_dir=JOB['source_batch_dir'])
        PLAN=flow.saved_plan(COLAB_PLAN_SHA)
        PLAN=queue.validate_plan(PLAN,JOB)
        if (PLAN.get('team')!={{'accounts':1,'gpu_count':1}}
                or PLAN.get('continued_from_plan')!=SOURCE_PLAN):
            raise ValueError('Plan không phải continuation Colab 1×1 đã khóa; không chạy.')
        PLAN_DIR=flow.stage_plan(PLAN)
        plan_path=PLAN_DIR/'job.json'
        sys.path[:0]=[str(PLAN_DIR),str(JOB_DIR),str(CHECKOUT/'src')]
        for module_name in ('full_queue_embeddings','new_data_kaggle','qwen_token_batching'):
            sys.modules.pop(module_name,None)
        importlib.invalidate_caches()
        import full_queue_embeddings as worker
        worker.verify_executor(PLAN)
        if worker.validate_plan(plan_path,JOB)!=PLAN:
            raise ValueError('Drive plan differs from the sealed execution plan.')
        from vietmedbridge.retrieval_models import TorchDenseEncoder
        from vietmedbridge.qwen_models import TorchQwenEncoder

        RUN_ROOT=(PROJECT/'data_staging/colab_gpu_checkpoints'/COLAB_PLAN_SHA/
            f'account-{{KAGGLE_ACCOUNT_ID}}'/'vmb_full_queue')
        OUTPUT=RUN_ROOT/'result'
        LOCAL_WORK=Path('/content/vmb_full_queue_workers')/COLAB_PLAN_SHA/f'account-{{KAGGLE_ACCOUNT_ID}}'
        deadline=time.monotonic()+max(0.25,WORKER_HOURS)*3600-120
        def qwen_factory():
            owner=TorchQwenEncoder(JOB['config']['second_dense'])
            if PLAN.get('qwen_execution') in worker.QWEN_EXECUTIONS:
                from qwen_token_batching import TokenBatchOwner
                return TokenBatchOwner(owner,RUN_ROOT/'qwen-tuning'/f'worker-{{KAGGLE_ACCOUNT_ID}}.json',
                    deadline=deadline,attention_tuning=PLAN.get('qwen_execution')==worker.QWEN_EXECUTION)
            return owner

        print('COLAB WORKER:',KAGGLE_ACCOUNT_ID,'| worker_id:',KAGGLE_ACCOUNT_ID,
            '| plan:',COLAB_PLAN_SHA[:12],'| team:',PLAN['team'],flush=True)
        print('DRIVE CHECKPOINT:',RUN_ROOT,flush=True)
        try:
            WORKER_RESULT=worker.run_worker(JOB_DIR,PLAN,OUTPUT,LOCAL_WORK,
                worker_id=KAGGLE_ACCOUNT_ID,deadline=deadline,
                factories={{'bge':lambda:TorchDenseEncoder(JOB['config']['dense']),'qwen':qwen_factory}})
        finally:
            SAVED_RESULT=worker.finalize(JOB_DIR,PLAN,RUN_ROOT,KAGGLE_ACCOUNT_ID)
            print('COLAB_CHECKPOINT_SAVED_ON_DRIVE:',OUTPUT/'new-data-result.json',flush=True)
            print('COLAB_RESULT_SUMMARY:',json.dumps({{'state':SAVED_RESULT['state'],
                'account_id':SAVED_RESULT['account_id'],'completed_tasks':len(SAVED_RESULT['tasks']),
                'plan':SAVED_RESULT['execution_plan']}},ensure_ascii=False),flush=True)
    ''').source
    return value


def colab_gpu_import_notebook():
    value = full_queue_transfer()
    value.cells[1].source = value.cells[1].source.replace(
        '#@title 1. Colab CPU — xuất TOÀN BỘ dữ liệu đã freeze',
        '#@title 1. Colab CPU — kết nối Drive và tải mã khóa')
    value.cells[1].source = value.cells[1].source.replace(
        "print('NEW DATA: CPU export/import; không chạy lại 02/03, không nạp model weights.', flush=True)",
        "print('COLAB IMPORT: chỉ nhập checkpoint Drive; không export corpus hay nạp model.', flush=True)")
    value.cells[0].source = dedent('''
        # Colab CPU — nhập checkpoint Colab về Drive rồi chuẩn bị lượt Kaggle 4×2

        Chạy sau khi Colab account 0 tạo `new-data-result.json` trên Drive.
        Notebook dùng SHA/manifest gốc và native importer; chạy lại an toàn,
        receipt đã nhập sẽ không tải lại payload.
        Có thể nhập thêm Outputs Kaggle acc 2 nếu phiên đó đã hoàn tất.

        Chỉ khi receipt của worker Colab account 0 đã nhập, notebook mới tạo kế hoạch
        Kaggle 4 acc × T4×2. Task của Kaggle acc 2 cũ vẫn bị giữ riêng nếu chưa có
        Outputs được nhập; không giao task đang chạy cho Kaggle/Colab lần nữa.
    ''').strip()
    value.cells[1].source = value.cells[1].source.replace(
        'KAGGLE_ACCOUNTS = 2 #@param {type:"integer"}',
        'SOURCE_KAGGLE_ACCOUNTS = 4  # Original 63b6 plan team; Colab runs one account')
    value.cells[1].source += '\nCOLAB_PLAN_SHA = "" #@param {type:"string"}\n'
    value.cells[3].source = cell(f'''
        #@title 3. Nhập Colab receipt; sau đó cấp lượt Kaggle 4×2
        COLAB_PLAN_SHA = COLAB_PLAN_SHA.strip()
        EXPECTED_LIBRARY_JOB = "{FULL_LIBRARY_JOB}"
        if not re.fullmatch(r'[0-9a-f]{{64}}',COLAB_PLAN_SHA):
            raise ValueError('Điền COLAB_PLAN_SHA đã dùng ở cả bốn worker.')
        flow=queue_flow(WORK_DIR/'cpu',accounts=4,gpu_count=2,batch_dir=BATCH_DIR)
        pointer=flow.root/'input_library.json'
        if not pointer.is_file():
            raise FileNotFoundError('Không thấy full input library frozen trên Drive.')
        library=queue._validate_library_header(queue.read_json(pointer))
        if library.get('manifest_sha256')!=EXPECTED_LIBRARY_JOB:
            raise ValueError('Full input library khác job đã chốt.')

        # Tùy chọn: nhập acc 2 của Kaggle plan cũ nếu đã có Outputs lưu thành công.
        if KAGGLE_NOTEBOOK_OUTPUT.strip():
            from kaggle.api.kaggle_api_extended import KaggleApi
            from vietmedbridge.kaggle_outputs import output_handles
            api=KaggleApi();api.authenticate()
            roots=[p.relative_to(PROJECT).as_posix() for p in (PROJECT/PARENT).glob('*/embeddings_v1')]
            download_work=PROJECT/'data_staging/kaggle_output_transfers'
            download_work.mkdir(parents=True,exist_ok=True)
            for handle in output_handles(KAGGLE_NOTEBOOK_OUTPUT):
                output=pipeline.download_outputs(api,handle,download_work,project=PROJECT,import_roots=roots,
                    allow_failed=True,manifest_path='vmb_full_queue/result/new-data-result.json')
                result=queue.checked(queue.read_json(output/'new-data-result.json'))
                if result.get('execution_plan')!="{SOURCE_EXECUTION_PLAN}" or result.get('account_id')!=2:
                    raise ValueError('Chỉ nhập đúng Outputs acc 2 của plan nguồn 63b6.')
                saved=queue.checked(queue.read_json(output/'job.json'))
                old_flow=queue_flow(WORK_DIR/'cpu',accounts=4,gpu_count=2,batch_dir=saved['source_batch_dir'])
                print(json.dumps(old_flow.install(output,receipt_api=api_source),ensure_ascii=False,indent=2),flush=True)

        plan=flow.saved_plan(COLAB_PLAN_SHA)
        if plan.get('team')!={{'accounts':1,'gpu_count':1}}:
            raise ValueError('Colab plan sai team; không nhập.')
        imported=[];missing=[]
        for account in range(1):
            output=(PROJECT/'data_staging/colab_gpu_checkpoints'/COLAB_PLAN_SHA/
                f'account-{{account}}'/'vmb_full_queue'/'result')
            manifest=output/'new-data-result.json'
            if not manifest.is_file():
                missing.append(account)
                continue
            result=queue.checked(queue.read_json(manifest))
            if (result.get('execution_plan')!=COLAB_PLAN_SHA or result.get('account_id')!=account
                    or result.get('job')!=EXPECTED_LIBRARY_JOB):
                raise ValueError(f'Colab account {{account}} output/plan binding không khớp.')
            report=flow.install(output,receipt_api=api_source)
            imported.append({{'account_id':account,'tasks':len(report['selection_ids']),
                'result':result['manifest_sha256'],'state':result['state']}})
            print('COLAB_RESULT_IMPORTED:',json.dumps(imported[-1],ensure_ascii=False),flush=True)

        print('COLAB_IMPORT_STATUS:',json.dumps({{'imported':imported,'missing_accounts':missing,
            'coverage':flow.coverage()}},ensure_ascii=False,indent=2),flush=True)
        if missing:
            print('Chưa tạo plan Kaggle. Chạy xong/rerun đúng account còn thiếu rồi chạy cell này lại.',flush=True)
        else:
            receipts=flow.account_receipts(library)
            plan_receipts={{r['account_id'] for r in receipts if r.get('execution_plan')==COLAB_PLAN_SHA}}
            if plan_receipts!={{0}}:
                raise RuntimeError('Chưa có receipt của Colab account 0; không thể mở khóa plan tiếp theo.')
            target,next_plan=flow.create_plan(library,receipt_api=api_source,
                execution_accounts=4,execution_gpu_count=2)
            if target is None:
                print('KAGGLE_PLAN_STATUS:',json.dumps(next_plan,ensure_ascii=False,indent=2),flush=True)
            else:
                plan_handle=queue.upload_plan(flow,target,next_plan,owner)
                print('KAGGLE_PLAN_READY: https://www.kaggle.com/datasets/'+plan_handle,flush=True)
                print('KAGGLE_PLAN:',next_plan['manifest_sha256'],'| tasks:',next_plan['scheduled_tasks'],
                    '| tasks vẫn khóa bởi acc 2 cũ:',len(next_plan.get('reserved_task_ids',[])),flush=True)
    ''').source
    return value


def colab_gpu_single_account_notebook():
    plan_book = colab_gpu_plan_notebook()
    worker_book = colab_gpu_worker_notebook()

    setup = plan_book.cells[1].source.replace(
        '#@title 1. Colab CPU — kết nối Drive và tải mã khóa',
        '#@title 1. Colab GPU — kết nối Drive, Kaggle và tải mã khóa')
    setup = setup.replace('import re\n', 'import re\nimport shutil\n', 1)
    setup = setup.replace(
        "if not Path('/var/colab/hostname').is_file():\n    raise RuntimeError('Mounting Colab required')",
        "if not Path('/var/colab/hostname').is_file() or not shutil.which('nvidia-smi'):\n"
        "    raise RuntimeError('Chọn Colab GPU trước khi Run all; không chuyển runtime giữa chừng.')")
    # The source setup's guard text varies by generated transfer version.
    setup = setup.replace(
        "if not Path('/var/colab/hostname').is_file():\n    raise RuntimeError('Notebook chuyển dữ liệu này chạy trên Colab CPU. Kaggle dùng 04_new_data_account_0 hoặc 1.')",
        "if not Path('/var/colab/hostname').is_file() or not shutil.which('nvidia-smi'):\n"
        "    raise RuntimeError('Chọn Colab GPU trước khi Run all; không chuyển runtime giữa chừng.')")
    setup = setup.replace(
        "print('COLAB PLAN: dùng manifest frozen trên Drive; không export corpus hay nạp model.', flush=True)",
        "print('COLAB GPU: tạo plan, embedding và ghi checkpoint trực tiếp lên Drive.', flush=True)")
    setup = setup.replace(
        'SOURCE_KAGGLE_ACCOUNTS = 4  # Original 63b6 plan team; Colab runs one account\n', '')
    setup = setup.replace('MIGRATE_UNSTARTED_LEGACY = True #@param {type:"boolean"}\n', '')
    setup = setup.replace('KAGGLE_NOTEBOOK_OUTPUT = "" #@param {type:"string"}\n', '')
    setup = setup.replace(
        '# Để trống để xuất gói. Sau GPU: owner/notebook[,owner2/notebook2] để nhập Outputs.\n',
        '')
    old_project_block = '''PROJECT = Path('/content/drive/MyDrive/VietMedBridge')
relative = 'data_staging/btc-maximize-71be0e12-v1'
if not (PROJECT/relative/'delta_candidates.json').is_file():
    PROJECT = Path('/content/drive/.shortcut-targets-by-id/1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK/VietMedBridge')
PROJECT = PROJECT.resolve()'''
    new_project_block = dedent(f'''\
        EXPECTED_LIBRARY_JOB = "{FULL_LIBRARY_JOB}"
        DRIVE_ROOT_CANDIDATES = [
            Path('/content/drive/MyDrive/VietMedBridge'),
            Path('/content/drive/.shortcut-targets-by-id/1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK/VietMedBridge'),
        ]
        PROJECT = None
        for candidate in DRIVE_ROOT_CANDIDATES:
            scope_parent = candidate/'data_staging/btc-full-embeddings-71be0e12-v1'
            active_path = scope_parent/'active_scope.json'
            if not active_path.is_file():
                continue
            try:
                active_scope = json.loads(active_path.read_text(encoding='utf-8'))
                scope_root = (candidate/active_scope['root']).resolve()
                if scope_root.parent != scope_parent.resolve():
                    continue
                library_path = scope_root/'embeddings_v1/kaggle-2x2/input_library.json'
                if not library_path.is_file():
                    continue
                saved_library = json.loads(library_path.read_text(encoding='utf-8'))
                if saved_library.get('manifest_sha256') == EXPECTED_LIBRARY_JOB:
                    PROJECT = candidate.resolve()
                    print('DRIVE_ROOT_SELECTED:',PROJECT,'| existing full library verified',flush=True)
                    break
            except (OSError, KeyError, TypeError, ValueError):
                continue
        if PROJECT is None:
            raise FileNotFoundError(
                'Không tìm thấy full input library đã freeze trong hai vị trí Drive. '
                'Notebook dừng trước khi tạo assignment; kiểm tra Drive mount hoặc thư mục full-embeddings.'
            )''')
    if old_project_block not in setup:
        raise ValueError('Expected Drive root selection block was not found in the Colab setup.')
    setup = setup.replace(old_project_block, new_project_block)
    if 'Notebook chuyển dữ liệu này chạy trên Colab CPU' in setup:
        raise ValueError('Single-Colab setup still contains the CPU-only guard.')

    plan_source = plan_book.cells[3].source
    plan_source = plan_source.replace(
        '#@title 3. Tạo plan Colab 1×1 GPU; giữ riêng task Kaggle acc 2 đang chạy',
        '#@title 3. Phân công task tự động; giữ riêng task Kaggle acc 2 đang chạy')
    plan_source = plan_source.replace(
        "raise FileNotFoundError('Drive chưa có full input library đã freeze; dừng để không tạo lại từ corpus.')",
        "raise FileNotFoundError(f'Drive full input library chưa có tại {pointer}; chưa tạo assignment.')")
    plan_source = plan_source.replace(
        "flow=queue_flow(WORK_DIR/'cpu',accounts=4,gpu_count=2,batch_dir=BATCH_DIR)",
        "flow=queue_flow(WORK_DIR/'cpu',accounts=2,gpu_count=2,batch_dir=BATCH_DIR)")
    # The optional old Kaggle output is imported once at the end, after Colab work.
    optional_start = plan_source.index("if KAGGLE_NOTEBOOK_OUTPUT.strip():")
    flow_start = plan_source.index("\nflow=queue_flow(", optional_start) + 1
    plan_source = plan_source[:optional_start] + plan_source[flow_start:]
    active_start = plan_source.index("active_path=flow.root/'active_execution_plan.json'")
    ready_start = plan_source.index("print('COLAB_PLAN_READY:'", active_start)
    persistent_plan = dedent(f'''\
        COLAB_PLAN_POINTER=flow.root/'colab_single_gpu_plan.json'
        if COLAB_PLAN_POINTER.is_file():
            pointer_record=queue.checked(queue.read_json(COLAB_PLAN_POINTER))
            if (pointer_record.get('schema')!='colab-single-account-plan-v1'
                    or pointer_record.get('library_job')!=EXPECTED_LIBRARY_JOB
                    or pointer_record.get('source_plan')!=CONTINUE_FROM_PLAN):
                raise RuntimeError('Drive continuation pointer belongs to another library/plan.')
            plan=flow.saved_plan(pointer_record.get('plan'))
            plan=queue.validate_plan(plan,library)
            if (plan.get('team')!={{'accounts':1,'gpu_count':1}}
                    or plan.get('continued_from_plan')!=CONTINUE_FROM_PLAN):
                raise RuntimeError('Saved Colab plan is not the expected single-GPU continuation.')
            target=flow.stage_plan(plan)
            print('COLAB_PLAN_REUSED: same task assignment; no new planning.',flush=True)
        else:
            active_path=flow.root/'active_execution_plan.json'
            active=queue.checked(queue.read_json(active_path))
            if active.get('plan')==CONTINUE_FROM_PLAN:
                target,plan=flow.create_plan(library,receipt_api=api_source,
                    execution_accounts=1,execution_gpu_count=1,
                    continue_completed_from=CONTINUE_FROM_PLAN)
                if target is None:
                    raise RuntimeError('No dispatchable tasks; inspect the reserved Kaggle account status.')
            else:
                plan=flow.saved_plan(active.get('plan'))
                if (plan.get('team')!={{'accounts':1,'gpu_count':1}}
                        or plan.get('continued_from_plan')!=CONTINUE_FROM_PLAN):
                    raise RuntimeError('Drive has another active plan. Refusing to reassign possibly running tasks.')
                target=flow.stage_plan(plan)
            pointer_record={{'schema':'colab-single-account-plan-v1',
                'library_job':EXPECTED_LIBRARY_JOB,'source_plan':CONTINUE_FROM_PLAN,
                'plan':plan['manifest_sha256'],'state':'PLANNED'}}
            queue.atomic_json(COLAB_PLAN_POINTER,queue.seal(pointer_record))
        pointer_record=queue.checked(queue.read_json(COLAB_PLAN_POINTER))
        RUN_COLAB_WORKER=pointer_record.get('state')!='IMPORTED'
        RUN_COLAB_IMPORT=RUN_COLAB_WORKER
        COLAB_PLAN_SHA=plan['manifest_sha256']
    ''')
    plan_source = plan_source[:active_start] + persistent_plan + plan_source[ready_start:]
    plan_source = plan_source.replace(
        "print('Dùng đúng một Colab GPU: ACCOUNT_ID=0; checkpoint riêng trên Drive.',flush=True)",
        "print('Colab worker: account 0 | one GPU | automatic resume enabled.',flush=True)\n"
        "print('CHECKPOINT_ROOT:',PROJECT/'data_staging/colab_gpu_checkpoints'/COLAB_PLAN_SHA/'account-0',flush=True)")
    plan_source = plan_source.replace("print('COLAB_PLAN_READY:',plan['manifest_sha256'],flush=True)",
        "print('COLAB_TASK_ASSIGNMENT_LOCKED:',plan['manifest_sha256'],flush=True)")
    plan_source = plan_source.replace("print('PLAN_ON_DRIVE:',flow.root/'plans'/(plan['manifest_sha256']+'.json'),flush=True)",
        "print('ASSIGNMENT_RECORD_ON_DRIVE:',flow.root/'plans'/(plan['manifest_sha256']+'.json'),flush=True)")

    worker_source = worker_book.cells[3].source
    worker_source = worker_source.replace(
        "flow=queue_flow(WORK_DIR/'cpu',accounts=4,gpu_count=2,batch_dir=JOB['source_batch_dir'])",
        "flow=queue_flow(WORK_DIR/'cpu',accounts=2,gpu_count=2,batch_dir=JOB['source_batch_dir'])")
    worker_source = worker_source.replace(
        'KAGGLE_ACCOUNT_ID = 0 #@param {type:"integer"}\n', 'KAGGLE_ACCOUNT_ID = 0\n')
    worker_source = worker_source.replace(
        '        COLAB_PLAN_SHA = "" #@param {type:"string"}\n', '')
    worker_lines = worker_source.splitlines()
    worker_title = worker_lines[0]
    worker_title = worker_title.replace(
        '#@title 3. Chạy một worker Colab GPU;',
        '#@title 4. Chạy một worker Colab GPU;')
    worker_body = '\n'.join(worker_lines[1:])
    worker_body = worker_body.replace("        COLAB_PLAN_SHA = \"\" #@param {type:\"string\"}\n", '')
    worker_body = worker_body.replace('COLAB_PLAN_SHA = "" #@param {type:"string"}\n', '')
    worker_body = '\n'.join(line for line in worker_body.splitlines()
                             if 'COLAB_PLAN_SHA = "" #@param' not in line)
    worker_body = worker_body.replace(
        "if KAGGLE_ACCOUNT_ID!=0 or not re.fullmatch(r'[0-9a-f]{64}',COLAB_PLAN_SHA):\n"
        "    raise ValueError('Dùng ACCOUNT_ID=0 và COLAB_PLAN_SHA chính xác từ notebook tạo plan.')",
        "if KAGGLE_ACCOUNT_ID!=0 or not re.fullmatch(r'[0-9a-f]{64}',COLAB_PLAN_SHA):\n"
        "    raise ValueError('Single-account plan binding is invalid; stop without running tasks.')")
    work_hours = "WORKER_HOURS = 11.0 #@param {type:\"number\"}"
    worker_body = worker_body.replace(work_hours, '')
    worker_body = '\n'.join(line.rstrip() for line in worker_body.splitlines()).strip()
    worker_source = (worker_title + '\n' + work_hours + '\n'
        "if RUN_COLAB_WORKER:\n" + indent(worker_body, '    ') + "\nelse:\n"
        "    print('Colab worker already imported; skipping embedding.',flush=True)")

    import_source = dedent(f'''\
        #@title 5. Nhập checkpoint vào Drive; tiếp tục nếu còn task
        if RUN_COLAB_IMPORT:
            EXPECTED_LIBRARY_JOB = "{FULL_LIBRARY_JOB}"
            flow=queue_flow(WORK_DIR/'cpu',accounts=2,gpu_count=2,batch_dir=BATCH_DIR)
            pointer=flow.root/'input_library.json'
            if not pointer.is_file():
                raise FileNotFoundError('Không thấy full input library frozen trên Drive.')
            library=queue._validate_library_header(queue.read_json(pointer))
            if library.get('manifest_sha256')!=EXPECTED_LIBRARY_JOB:
                raise ValueError('Full input library khác job đã chốt.')

            plan=flow.saved_plan(COLAB_PLAN_SHA)
            if plan.get('team')!={{'accounts':1,'gpu_count':1}}:
                raise ValueError('Assignment lock sai team; không nhập.')
            output=(PROJECT/'data_staging/colab_gpu_checkpoints'/COLAB_PLAN_SHA/
                'account-0'/'vmb_full_queue'/'result')
            manifest=output/'new-data-result.json'
            if not manifest.is_file():
                raise FileNotFoundError('Worker chưa ghi receipt lên Drive; chạy lại cùng notebook để tiếp tục.')
            result=queue.checked(queue.read_json(manifest))
            if (result.get('execution_plan')!=COLAB_PLAN_SHA or result.get('account_id')!=0
                    or result.get('job')!=EXPECTED_LIBRARY_JOB):
                raise ValueError('Colab result không khớp task assignment; không nhập.')
            report=flow.install(output,receipt_api=api_source)
            print('COLAB_RESULT_IMPORTED_TO_DRIVE:',json.dumps({{
                'state':result['state'],'completed_tasks':len(report['selection_ids']),
                'assigned_tasks':len(plan['tasks']),'coverage':flow.coverage()
            }},ensure_ascii=False,indent=2),flush=True)

            if result.get('state')=='ROUND_COMPLETE':
                queue.atomic_json(COLAB_PLAN_POINTER,queue.seal({{
                    'schema':'colab-single-account-plan-v1','library_job':EXPECTED_LIBRARY_JOB,
                    'source_plan':"{SOURCE_EXECUTION_PLAN}",'plan':COLAB_PLAN_SHA,
                    'state':'IMPORTED','updated_at':queue.utc_now()}}))
                RUN_COLAB_WORKER=False
                RUN_COLAB_IMPORT=False
                print('COLAB_ACCOUNT_COMPLETE: no more Colab tasks in this assignment.',flush=True)
            else:
                print('COLAB_PARTIAL_CHECKPOINT_IMPORTED: rerun this same notebook to resume remaining assigned tasks.',flush=True)
        else:
            print('Colab account assignment already complete; skipping embedding/import.',flush=True)
    ''').strip()

    metadata_source = plan_book.cells[2].source.replace(
        '#@title 2. Kết nối manifest frozen và cache đã có — CPU, không mở vector payload',
        '#@title 2. Nạp mã và metadata — không chuyển runtime, không mở vector payload')
    value = nb.v4.new_notebook(cells=[
        nb.v4.new_markdown_cell(dedent('''
            # 04 — Colab một GPU: khóa task, embedding và nhập Drive trong cùng notebook

            Chọn một GPU Colab (T4), thêm Secret `KAGGLE_JSON`, kết nối Drive, rồi bấm **Run all**.
            Notebook tự khóa task của Kaggle acc 2 đang chạy, tạo phân công nội bộ 1×1, chạy worker 0,
            lưu vector/journal/manifest trực tiếp vào Drive, rồi nhập receipt ngay trên Drive.
            Không cần mở notebook plan/import riêng hay xuất plan dataset mới.

            Nếu Colab dừng giữa chừng, chạy lại **cùng notebook**. Plan và checkpoint Drive
            được giữ nguyên để tiếp tục; không mở thêm phiên Colab song song với account 0.
            Nếu đã tạo plan 4 account bằng notebook cũ, không khởi chạy worker cũ; notebook
            này sẽ dừng thay vì âm thầm đổi assignment đang có.
        ''').strip()),
        nb.v4.new_code_cell(setup),
        nb.v4.new_code_cell(metadata_source),
        nb.v4.new_code_cell(plan_source),
        nb.v4.new_code_cell(worker_source),
        nb.v4.new_code_cell(import_source),
    ])
    value.metadata = dict(plan_book.metadata)
    value.metadata['accelerator'] = 'GPU'
    return value


def kaggle_four_account_continuation_notebook():
    source_book = colab_gpu_plan_notebook()
    setup = source_book.cells[1].source.replace(
        '#@title 1. Colab CPU — kết nối Drive và tải mã khóa',
        '#@title 1. Colab CPU — kết nối Drive để chuẩn bị lượt Kaggle 4 acc')
    setup = setup.replace(
        'SOURCE_KAGGLE_ACCOUNTS = 4  # Original 63b6 plan team; Colab runs one account\n', '')
    setup = setup.replace('KAGGLE_NOTEBOOK_OUTPUT = "" #@param {type:"string"}\n', '')
    setup = setup.replace(
        '# Để trống để xuất gói. Sau GPU: owner/notebook[,owner2/notebook2] để nhập Outputs.\n', '')
    old_project_block = '''PROJECT = Path('/content/drive/MyDrive/VietMedBridge')
relative = 'data_staging/btc-maximize-71be0e12-v1'
if not (PROJECT/relative/'delta_candidates.json').is_file():
    PROJECT = Path('/content/drive/.shortcut-targets-by-id/1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK/VietMedBridge')
PROJECT = PROJECT.resolve()'''
    new_project_block = dedent(f'''\
        EXPECTED_LIBRARY_JOB = "{FULL_LIBRARY_JOB}"
        DRIVE_ROOT_CANDIDATES = [
            Path('/content/drive/MyDrive/VietMedBridge'),
            Path('/content/drive/.shortcut-targets-by-id/1uKKHYEcMPodnJIscKPi6jsoNBJvyhFGK/VietMedBridge'),
        ]
        PROJECT = None
        for candidate in DRIVE_ROOT_CANDIDATES:
            scope_parent = candidate/'data_staging/btc-full-embeddings-71be0e12-v1'
            active_path = scope_parent/'active_scope.json'
            if not active_path.is_file():
                continue
            try:
                active_scope = json.loads(active_path.read_text(encoding='utf-8'))
                scope_root = (candidate/active_scope['root']).resolve()
                if scope_root.parent != scope_parent.resolve():
                    continue
                library_path = scope_root/'embeddings_v1/kaggle-2x2/input_library.json'
                if not library_path.is_file():
                    continue
                saved_library = json.loads(library_path.read_text(encoding='utf-8'))
                if saved_library.get('manifest_sha256') == EXPECTED_LIBRARY_JOB:
                    PROJECT = candidate.resolve()
                    print('DRIVE_ROOT_SELECTED:',PROJECT,'| full library team 2×T4 verified',flush=True)
                    break
            except (OSError, KeyError, TypeError, ValueError):
                continue
        if PROJECT is None:
            raise FileNotFoundError(
                'Không tìm thấy full input library d6ab... trên Drive; chưa tạo plan hay đổi assignment.'
            )''')
    if old_project_block not in setup:
        raise ValueError('Expected Drive root selection block was not found in continuation setup.')
    setup = setup.replace(old_project_block, new_project_block)

    plan = dedent(f'''\
        #@title 3. Tạo plan tiếp tục 4 acc × T4×2; giữ task đang chạy của plan 63b6
        CONTINUE_FROM_PLAN = "{SOURCE_EXECUTION_PLAN}"
        EXPECTED_LIBRARY_JOB = "{FULL_LIBRARY_JOB}"
        SOURCE_LIBRARY_ACCOUNTS = 2
        EXECUTION_ACCOUNTS = 4
        EXECUTION_GPUS_PER_ACCOUNT = 2

        flow=queue_flow(WORK_DIR/'cpu',accounts=SOURCE_LIBRARY_ACCOUNTS,gpu_count=2,batch_dir=BATCH_DIR)
        pointer=flow.root/'input_library.json'
        if not pointer.is_file():
            raise FileNotFoundError(f'Full input library chưa có tại {{pointer}}; không tạo task mới.')
        library=queue._validate_library_header(queue.read_json(pointer))
        if library.get('manifest_sha256')!=EXPECTED_LIBRARY_JOB:
            raise ValueError('Full input library sai manifest; giữ nguyên task cũ.')

        active_path=flow.root/'active_execution_plan.json'
        active=queue.checked(queue.read_json(active_path))
        if active.get('plan')==CONTINUE_FROM_PLAN:
            target,plan=flow.create_plan(library,receipt_api=api_source,
                execution_accounts=EXECUTION_ACCOUNTS,execution_gpu_count=EXECUTION_GPUS_PER_ACCOUNT,
                continue_completed_from=CONTINUE_FROM_PLAN)
        else:
            plan=flow.saved_plan(active.get('plan'))
            if (plan.get('team')!={{'accounts':EXECUTION_ACCOUNTS,'gpu_count':EXECUTION_GPUS_PER_ACCOUNT}}
                    or plan.get('continued_from_plan')!=CONTINUE_FROM_PLAN):
                raise RuntimeError('Drive đang có plan khác; không đổi assignment khi task có thể đang chạy.')
            target=flow.stage_plan(plan)
            print('KAGGLE_CONTINUATION_PLAN_REUSED: cùng phân công, không tạo trùng.',flush=True)

        if target is None:
            print('KAGGLE_CONTINUATION_STATUS:',json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
            print('Chưa có task an toàn để giao; account chưa có receipt vẫn được giữ riêng.',flush=True)
        else:
            handle=queue.upload_plan(flow,target,plan,owner)
            print('KAGGLE_CONTINUATION_PLAN_READY: https://www.kaggle.com/datasets/'+handle,flush=True)
            print('PLAN:',plan['manifest_sha256'],'| team:',plan['team'],'| scheduled:',plan['scheduled_tasks'],
                '| remaining:',plan.get('remaining_tasks_after_plan'),
                '| reserved from old running accounts:',len(plan.get('reserved_task_ids',[])),flush=True)
            print('INPUT_LIBRARY: https://www.kaggle.com/datasets/{FULL_LIBRARY_HANDLE}',flush=True)
            print('Thêm đúng hai dataset trên vào 04_full_data_4acc_account_0/1/2/3; không dùng plan cũ.',flush=True)
    ''')
    value = nb.v4.new_notebook(cells=[
        nb.v4.new_markdown_cell(dedent(f'''\
            # Tiếp tục Qwen trên Kaggle — 4 acc × T4×2

            Chạy notebook này **một lần trên Colab CPU** để dùng receipt đã lưu trên Drive và xuất plan tiếp nối.
            Nó giữ riêng task chưa có receipt của plan `{SOURCE_EXECUTION_PLAN[:12]}`, không chạy embedding,
            không nạp model và không tạo lại thư viện text.

            Khi có `KAGGLE_CONTINUATION_PLAN_READY`, trên cả 4 tài khoản Kaggle Add Input cùng:
            1. thư viện text `vmb-full-inputs-d6ab9a432ac6-1792733353195313801`;
            2. dataset plan mới được in ở cuối notebook.

            Chạy `04_full_data_4acc_account_0.ipynb`, `..._1.ipynb`, `..._2.ipynb`, `..._3.ipynb` tương ứng.
            Nếu acc 2 vẫn đang chạy plan cũ, để phiên đó hoàn tất/lưu Outputs trước khi mở worker mới trên acc 2.
        ''').strip()),
        nb.v4.new_code_cell(setup),
        source_book.cells[2],
        nb.v4.new_code_cell(plan),
    ])
    value.metadata = dict(source_book.metadata)
    value.metadata.pop('accelerator', None)
    return value


def main():
    nb.write(full_queue_transfer(), ROOT/'notebooks/04_colab_export_full_data_kaggle.ipynb')
    nb.write(three_account_transfer(), ROOT/'notebooks/04_colab_prepare_3_accounts_kaggle.ipynb')
    nb.write(four_account_transfer(), ROOT/'notebooks/04_colab_prepare_4_accounts_kaggle.ipynb')
    nb.write(completed_output_import(), ROOT/'notebooks/04_colab_import_completed_kaggle.ipynb')
    nb.write(qwen_benchmark_notebook(), ROOT/'notebooks/kaggle/04_benchmark_qwen_t4.ipynb')
    nb.write(two_account_continuation(), ROOT/'notebooks/04_colab_continue_accounts_0_1.ipynb')
    nb.write(colab_gpu_plan_notebook(), ROOT/'notebooks/04_colab_full_queue_gpu_plan.ipynb')
    nb.write(colab_gpu_worker_notebook(), ROOT/'notebooks/04_colab_full_queue_gpu_worker.ipynb')
    nb.write(colab_gpu_import_notebook(), ROOT/'notebooks/04_colab_full_queue_gpu_import_and_kaggle_continue.ipynb')
    nb.write(colab_gpu_single_account_notebook(), ROOT/'notebooks/04_colab_full_queue_gpu_single_account.ipynb')
    nb.write(kaggle_four_account_continuation_notebook(), ROOT/'notebooks/04_colab_continue_kaggle_4_accounts.ipynb')
    for account in (0,1):
        nb.write(full_queue_gpu(account), ROOT/f'notebooks/kaggle/04_full_data_account_{account}.ipynb')
        nb.write(full_queue_gpu(account), ROOT/f'notebooks/kaggle/04_continue_2acc_account_{account}.ipynb')
    for account in (0,1,2):
        nb.write(full_queue_gpu(account, expected_accounts=3), ROOT/f'notebooks/kaggle/04_full_data_3acc_account_{account}.ipynb')
    for account in range(4):
        nb.write(full_queue_gpu(account, expected_accounts=4), ROOT/f'notebooks/kaggle/04_full_data_4acc_account_{account}.ipynb')


if __name__=='__main__':
    main()
