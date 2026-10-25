"""A completed GPU run is bound before any vector transfer can start."""
import pytest

from test_new_data_kaggle import project
from test_kaggle_output_paged import setup_remote
from scripts.new_data_kaggle import download_outputs
from scripts.write_full_queue_notebooks import completed_output_import


@pytest.mark.parametrize('field,value', [
    ('manifest_sha256', 'f'*64), ('job', 'f'*64),
    ('execution_plan', 'f'*64), ('account_id', 2), ('completed_tasks', 918),
])
def test_wrong_saved_run_stops_before_listing_or_vector_download(project, tmp_path, monkeypatch, field, value):
    flow, remote, result, work, api, listed, downloads, hub, pinned, relative = setup_remote(project, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='expected completed GPU run'):
        download_outputs(api, 'owner/saved/2', work, allow_failed=True,
                         manifest_path=relative, expected_result={field:value})
    assert not listed and not downloads and not flow.imported()


def test_matching_run_imports_once_and_cold_import_skips_vectors(project, tmp_path, monkeypatch):
    flow, remote, result, work, api, listed, downloads, hub, pinned, relative = setup_remote(project, tmp_path, monkeypatch)
    expected={k:result[k] for k in ('manifest_sha256','job','execution_plan','account_id')}
    expected['completed_tasks']=len(result['tasks'])
    output=download_outputs(api,'owner/saved/2',work,allow_failed=True,
                            manifest_path=relative,expected_result=expected)
    report=flow.install(output)
    assert report['state']=='NEW_DATA_KAGGLE_IMPORTED'
    before=len(downloads)
    roots=[flow.view.output.relative_to(project[0]).as_posix()]
    output=download_outputs(api,'owner/saved/2',tmp_path/'new-runtime',project=project[0],
        import_roots=roots,allow_failed=True,manifest_path=relative,expected_result=expected)
    assert len(downloads)==before
    assert flow.install(output)['payloads_reopened'] is False


def test_import_notebook_is_preconfigured_and_does_not_create_a_plan():
    notebook=completed_output_import()
    for c in notebook.cells:
        if c.cell_type=='code': compile(c.source,'<completed-import>','exec')
    assert 'KAGGLE_NOTEBOOK_OUTPUT = "kaisercoding/impact-2"' in notebook.cells[1].source
    source=notebook.cells[3].source
    assert 'completed_tasks\':918' in source
    assert "download_work=PROJECT/'data_staging/kaggle_output_transfers'" in source
    assert 'expected_result=expected' in source
    assert '.create_plan(' not in source and '.export_inputs(' not in source and '.upload' not in source
