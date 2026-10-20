"""Generate the isolated later-shard CPU notebook, reusing the parent runtime lock."""
from pathlib import Path
from textwrap import dedent
import json
import nbformat as nb

from write_isolated_intake_notebook import make_cells, FULL_SCOPE, RESUME_TOOLS

ROOT=Path(__file__).resolve().parents[1]
EXTRA=(ROOT/'scripts/intake_00008_extension.py').read_text(encoding='utf-8')


def notebook():
    old=make_cells()
    setup=old[1]['source'].replace('CAMPAIGN = "btc-new-71be0e12-20261019-v1"',
        'PARENT_ROOT = PROJECT.resolve() / "data_staging/btc-new-71be0e12-20261019-v1"\n'
        'CAMPAIGN = "btc-new-71be0e12-00008-20261020-v1"')
    setup=setup.replace('lock = json.loads(lock_path.read_text()) if lock_path.is_file() else {}',
        'parent_lock = PARENT_ROOT / "intake_code_lock.json"\n'
        'if not parent_lock.is_file():\n'
        '    raise RuntimeError("Thiếu code lock của campaign 9 nguồn đã hoàn tất; chưa tự chọn code khác.")\n'
        'lock = json.loads((lock_path if lock_path.is_file() else parent_lock).read_text())')
    setup=setup.replace('if runtime_path.is_file():\n    pinned = json.loads(runtime_path.read_text())',
        'pinned_runtime_path = runtime_path if runtime_path.is_file() else PARENT_ROOT / "intake_runtime_lock.json"\n'
        'if not pinned_runtime_path.is_file():\n'
        '    raise RuntimeError("Thiếu runtime lock của parent campaign.")\n'
        'if pinned_runtime_path.is_file():\n    pinned = json.loads(pinned_runtime_path.read_text())')
    setup=setup.replace('if not lock:\n    atomic_json(lock_path',
        'if not lock_path.is_file():\n    atomic_json(lock_path')
    setup=setup.replace('/content/vmb_intake_work','/content/vmb_intake_00008_work')
    setup=setup.replace('/content/VietMedBridge-intake','/content/VietMedBridge-intake-00008')
    setup=setup.replace('#@title 1. Kết nối Drive và cài code riêng cho lượt nhập mới',
        '#@title 1. CPU: dùng code/runtime lock của campaign đã xong; output riêng cho 00008')
    assert 'parent_lock' in setup and 'pinned_runtime_path' in setup
    assert 'reference = lock.get("commit") or "main"' in setup
    # Only the resource-loader definitions are reused; do not construct/run the
    # old nine-source controller or append to its immutable source list.
    loaders=old[2]['source'][old[2]['source'].index('LAZY_RESOURCES = {}'):]
    loaders=loaders.replace('shared = INTAKE_ROOT / "references', 'shared = PARENT_ROOT / "references')
    prepare=RESUME_TOOLS+'\n'+EXTRA+'\n'+dedent('''
        from vietmedbridge.artifacts import verify_file, publish_file
        from vietmedbridge.isolated_intake import DriveSource, fetch_reference
        from vietmedbridge.chunks import load_bge_tokenizer, ChunkConfig
        from vietmedbridge.golden import run_golden_suite
        from huggingface_hub import hf_hub_download
        RUN_MODE = "sequential"
        REFERENCE_DIR = INTAKE_ROOT / "references"
    ''')+'\n'+loaders+'\nFULL_SCOPE = json.loads('+repr(json.dumps(FULL_SCOPE,ensure_ascii=False))+')\n'+dedent('''
        PLAN, PRIOR_MAPPINGS, PARENT_STATUS = extension_00008_plan(
            PARENT_ROOT, INTAKE_ROOT, WORK_DIR, FULL_SCOPE, load_reader)
        CONFIG = ChunkConfig(**PLAN["chunking"])
        TOKENIZER_SPEC = {"model_id":"BAAI/bge-m3", "revision":PLAN["tokenizer_revision"], "special_tokens":False}
        CONTROLLER = IntakeResume(PLAN, root=INTAKE_ROOT, work_dir=WORK_DIR,
            tokenizer_spec=TOKENIZER_SPEC, chunk_config=CONFIG, shard_size=2048,
            prior_mappings=PRIOR_MAPPINGS)
        print("Parent giữ nguyên:", PARENT_STATUS["completed_sources"], "/", PARENT_STATUS["total_sources"],flush=True)
        print(json.dumps(CONTROLLER.status(),ensure_ascii=False,indent=2),flush=True)
    ''')
    run=dedent('''
        #@title 3. Nhập riêng 00008: bỏ qua ID đã hợp lệ, chia chunk phần mới, giữ lỗi để xử lý sau
        def source_metadata(source,target):
            return load_00008_metadata(load_reader(),source,target,INTAKE_ROOT)
        def continue_00008():
            return CONTROLLER.run(load_reader=load_reader,load_links=load_links,
                load_baseline=load_baseline,load_tokenizer=load_tokenizer,load_metadata=source_metadata)
        try:
            RESULT = continue_00008()
        except OSError as error:
            if error.errno not in (103,107):
                raise
            print("Drive ngắt kết nối; giữ checkpoint và gắn lại một lần.",flush=True)
            drive.mount("/content/drive",force_remount=True)
            RESULT = continue_00008()
        atomic_json(INTAKE_ROOT / "intake_status.json",RESULT)
        print(json.dumps(RESULT,ensure_ascii=False,indent=2),flush=True)
        if RESULT["state"] == "INTAKE_COMPLETE":
            print("00008 CPU COMPLETE. Báo cáo:", INTAKE_ROOT / "intake_status.json",flush=True)
            print("Chưa embedding/freeze. Parent 9 nguồn và candidate submit hiện tại giữ nguyên.",flush=True)
    ''')
    book=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(dedent('''
        # 02 — Nhập riêng data mới 00008 (Colab CPU)

        **CPU → Run all trên một tài khoản riêng. Không dừng luồng submit.**
        Chỉ nhận `vibiomir_retain_shard_00008.tar` mới tìm thấy trong AI Guru.
        Tự kiểm descriptor, snapshot và SHA frontier qua Drive API ranges trước khi khóa plan.
        Không tải cả archive 2,50 GB; metadata cần xử lý và kết quả được lưu trên Drive.

        Dùng lại mapping hợp lệ của baseline và 9 nguồn đã hoàn tất để bỏ qua phần trùng.
        Chín nguồn cũ chỉ đọc receipt nhỏ khi nhận handoff; SHA của mapping được kiểm
        khi thực sự dùng để chọn ID mới. Không chạy lại import/chunk của chúng.

        Output: `VietMedBridge/data_staging/btc-new-71be0e12-00008-20261020-v1/`.
        Không đổi active candidate, snapshot hoặc model cache của luồng submit.
        Ngắt phiên: mở lại đúng notebook, Run all với cùng folder; giữ plan và checkpoint.
        `INTAKE_COMPLETE` chỉ là import/quality/chunk CPU cho nguồn này, chưa phải embedding,
        index, submission hoặc chứng nhận đã đủ toàn bộ ID của BTC.
    ''').strip()),nb.v4.new_code_cell(setup),nb.v4.new_code_cell('#@title 2. Nhận 9 receipt cũ và audit nguồn 00008\n'+prepare),
        nb.v4.new_code_cell(run)])
    book.metadata={'colab':{'name':'02_colab_import_new_00008.ipynb'},
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}}
    return book


if __name__=='__main__':
    target=ROOT/'notebooks/02_colab_import_new_00008.ipynb'
    nb.write(notebook(),target)
    print(target)
