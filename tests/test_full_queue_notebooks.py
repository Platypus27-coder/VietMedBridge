"""The full library is checked before model setup, including unplanned assets."""
import ast
import json
from pathlib import Path

import pytest

from test_new_data_kaggle import project
from scripts.full_queue_embeddings import FullQueueKaggle
from scripts.kaggle_input_preflight import inspect_full_queue_inputs
from scripts.write_full_queue_notebooks import full_queue_transfer, full_queue_gpu, ROOT, SCOPE_SHA
from vietmedbridge.content_embeddings import seal


def rebind(value, **changes):
    return seal({**{k:v for k,v in value.items() if k!='manifest_sha256'}, **changes})


def test_generated_cpu_embeds_current_helpers_and_exports_all_inputs():
    notebook=full_queue_transfer()
    for c in notebook.cells:
        if c.cell_type=='code': compile(c.source,'<full-queue-cpu>','exec')
    source=notebook.cells[2].source
    nodes=ast.parse(source).body
    assignment=next(n.value for n in nodes if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets))
    embedded=json.loads(ast.literal_eval(assignment.args[0]))
    assert all(embedded[name]==(ROOT/'scripts'/name).read_text(encoding='utf-8') for name in embedded)
    writes=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text'
            and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)]
    assert (ROOT/'scripts/full_queue_embeddings.py').read_text(encoding='utf-8') in [n.args[0].value for n in writes]
    assert (ROOT/'scripts/kaggle_output_transfer.py').read_text(encoding='utf-8') in [n.args[0].value for n in writes]
    assert 'flow.export_inputs(' in notebook.cells[3].source
    assert 'flow.create_plan(' in notebook.cells[3].source
    assert 'flow.export(' not in notebook.cells[3].source
    assert 'supersede_unstarted_legacy=MIGRATE_UNSTARTED_LEGACY' in notebook.cells[3].source
    assert 'allow_failed=True' in notebook.cells[3].source
    assert "manifest_path='vmb_full_queue/result/new-data-result.json'" in notebook.cells[3].source
    assert 'sys.modules.pop(module_name,None)' in source


def test_gpu_notebooks_gate_the_complete_library_before_loading_models():
    for account in (0,1):
        notebook=full_queue_gpu(account)
        for c in notebook.cells:
            if c.cell_type=='code': compile(c.source,'<full-queue-gpu>','exec')
        first=notebook.cells[1].source
        assert f'KAGGLE_ACCOUNT_ID={account}' in first
        assert 'inspect_full_queue_inputs(' in first
        assert SCOPE_SHA in first
        assert 'nvidia-smi' not in first and 'pip' not in first
        assert 'nvidia-smi' in notebook.cells[2].source
        assert "JOB_DIR/'full_queue_embeddings.py'" in notebook.cells[3].source
        assert "'--plan',str(PLAN_PATH)" in notebook.cells[3].source
        assert (ROOT/'scripts/kaggle_output_guard.py').read_text(encoding='utf-8') in notebook.cells[3].source
        assert 'GPU_RESULT=run_with_bounded_output(' in notebook.cells[3].source
        assert 'subprocess.run(command,env=env,check=True)' not in notebook.cells[3].source


def test_stdlib_inventory_checks_a_shared_file_not_scheduled_in_this_plan(project,tmp_path,monkeypatch):
    directory,identities,checkout=project
    flow=FullQueueKaggle(directory,tmp_path/'cpu',identities,accounts=2)
    target,library=flow.export_inputs(checkout,config={},runtime={},code_commit='a'*40)
    _,plan=flow.create_plan(library,output_bytes=2100)
    scope={'official_ids':802028,'source_builds':13,'scope_manifest_sha256':SCOPE_SHA}
    library=rebind(library,coverage_scope=scope,frozen_sources=[{'build_run':str(i)} for i in range(13)])
    plan=rebind(plan,library_job=library['manifest_sha256'],coverage_scope=scope)
    original=Path.read_bytes
    def metadata_only(path):
        if path.suffix=='.parquet': pytest.fail('Preflight read a text payload')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',metadata_only)
    summary=inspect_full_queue_inputs(target,library,plan,expected_scope=SCOPE_SHA,report=lambda *a:None)
    assert summary['input_export_complete'] and summary['all_exported_tasks']==len(library['tasks'])
    assert summary['this_gpu_plan_tasks']<summary['all_exported_tasks']
    assert summary['semantic_quality']=='NOT_CERTIFIED_BY_FILE_INVENTORY'
    scheduled={t['asset'] for t in library['tasks'] if t['id'] in plan['tasks']}
    not_scheduled=set(library['assets'])-scheduled
    assert not_scheduled
    (target/next(iter(not_scheduled))).unlink()
    with pytest.raises(FileNotFoundError,match='missing 1 shared text files'):
        inspect_full_queue_inputs(target,library,plan,expected_scope=SCOPE_SHA,report=lambda *a:None)
