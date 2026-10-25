"""Keep all text inputs on Kaggle; subsequent rounds only upload a small plan."""
from pathlib import Path
from textwrap import dedent
import nbformat as nb

try:
    from .write_new_data_kaggle_notebooks import ROOT, book, cell, full_transfer, gpu
except ImportError:
    from write_new_data_kaggle_notebooks import ROOT, book, cell, full_transfer, gpu

SCOPE_SHA = '0310f6d51a13bdff63cb9d9dfe337d9aee8898df9f34857c84a2708d9f941721'


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
                    print('GPU_PLAN_DATASET_READY:','https://www.kaggle.com/datasets/'+plan['dataset_handle'],flush=True)
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
        SESSION_HOURS=10.0
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
            if set(PLAN.get('executor_sources',{}))!={'full_queue_embeddings.py','new_data_kaggle.py'}:
                raise ValueError('Missing locked execution scheduler sources.')
            for name,checksum in PLAN['executor_sources'].items():
                if hashlib.sha256((EXECUTOR_DIR/name).read_bytes()).hexdigest()!=checksum:
                    raise ValueError('Execution scheduler checksum changed.')
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

        Sau khi ba phiên lưu Outputs xong, điền ba handle notebook vào
        KAGGLE_NOTEBOOK_OUTPUT rồi chạy lại notebook CPU này để nhập và tạo
        lượt tiếp theo. Không tự chuyển team thư viện gốc sang 3.
    ''').strip()
    value.cells[1].source = value.cells[1].source.replace(
        'KAGGLE_ACCOUNTS = 2 #@param {type:"integer"}',
        'KAGGLE_ACCOUNTS = 2  # Thư viện gốc; giữ để tiếp tục đúng checkpoint.\n'
        'EXECUTION_ACCOUNTS = 3  # Ba acc, mỗi acc T4 x2.\n'
        'CONFIRMED_UNSTARTED_PLAN = "ab7954e4a47f3301ee6c484f696a1c312e86052dca6976e0d2e2f533a06d9ef8"')
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
        "print('GPU_PLAN_DATASET_READY:',",
        "print('GPU_PLAN_ASSIGNMENTS:',{a:sum(plan['task_workers'][t]//2==a for t in plan['tasks']) for a in range(EXECUTION_ACCOUNTS)},flush=True)\n"
        "        print('GPU_PLAN_DATASET_READY:',")
    return value


def main():
    nb.write(full_queue_transfer(), ROOT/'notebooks/04_colab_export_full_data_kaggle.ipynb')
    nb.write(three_account_transfer(), ROOT/'notebooks/04_colab_prepare_3_accounts_kaggle.ipynb')
    for account in (0,1):
        nb.write(full_queue_gpu(account), ROOT/f'notebooks/kaggle/04_full_data_account_{account}.ipynb')
    for account in (0,1,2):
        nb.write(full_queue_gpu(account, expected_accounts=3), ROOT/f'notebooks/kaggle/04_full_data_3acc_account_{account}.ipynb')


if __name__=='__main__':
    main()
