from pathlib import Path

import pytest

from src.utils.config import load_config, prepare_directories


@pytest.fixture
def config(tmp_path):
    values = load_config(output_dir=tmp_path/'output')
    values['dataset'].update(query_id_column='query_id', gold_doc_ids_column='gold_doc_ids')
    values['crawler'].update(concurrent_requests=4, concurrent_requests_per_domain=2, download_timeout=0.5,
                             retry_times=1, max_response_bytes=100000, warn_response_bytes=50000)
    values['crawler']['autothrottle']['enabled'] = False
    values['crawler']['rate_limit']['default_cooldown_seconds'] = 0.1
    values['extraction']['workers'] = 2
    values['extraction']['max_tasks_per_child'] = 2
    prepare_directories(values)
    return values
