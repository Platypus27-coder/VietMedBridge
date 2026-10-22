"""Read frozen recovery manifests; share the existing sparse Kaggle worker/importer."""
from pathlib import Path
import re

from vietmedbridge.artifacts import atomic_json, digest_json, read_json
from vietmedbridge.content_embeddings import checked, seal
from vietmedbridge.scale_vectors import bound

CAMPAIGN = 'data_staging/btc-maximize-71be0e12-v1'


def load_recovery_batch(project, batch_dir=None):
    project = Path(project).resolve()
    parent = project / CAMPAIGN
    if batch_dir is None:
        frozen = checked(read_json(parent / 'delta_candidates.json'))
        root = parent / 'embedding_batches' / frozen['manifest_sha256'][:16]
        pointer = root / 'frozen_deltas.json'
        if pointer.is_file():
            if checked(read_json(pointer)) != frozen:
                raise ValueError('Recovery handoff already exists with different manifests.')
        else:
            atomic_json(pointer, frozen)
    else:
        root = bound(project, batch_dir)
        if root.parent != parent / 'embedding_batches' or not re.fullmatch('[0-9a-f]{16}', root.name):
            raise ValueError('Unexpected recovery embedding batch directory.')
        frozen = checked(read_json(root / 'frozen_deltas.json'))
        if root.name != frozen['manifest_sha256'][:16]:
            raise ValueError('Recovery handoff directory checksum mismatch.')
    if frozen.get('submission_candidate_changed') is not False or not frozen.get('builds'):
        raise ValueError('A nonempty immutable recovery handoff is required.')
    sources, builds, names = [], [], set()
    for item in frozen['builds']:
        name = item['build_run']
        if (item.get('state') != 'DELTA_READY_FOR_EMBEDDING'
                or not isinstance(item.get('data_root'), str) or item['data_root'].replace('\\', '/') != CAMPAIGN
                or not re.fullmatch(r'(metadata-text-v1|raw-text-v1|raw-ready-[0-9a-f]{16})', name)
                or name in names or item.get('submission_candidate_changed') is not False):
            raise ValueError('Recovery source coverage or producer scope changed.')
        names.add(name)
        build = parent / 'processed' / name
        expected_name = 'candidate-' + item['candidate_manifest_sha256'][:16] + '.json'
        if item['candidate_name'] != expected_name:
            raise ValueError('Unexpected frozen candidate path.')
        candidate = read_json(build / expected_name)
        expected = {k: v for k, v in candidate.items() if k != 'candidate_manifest_sha256'}
        if (digest_json(expected) != item['candidate_manifest_sha256']
                or candidate.get('candidate_manifest_sha256') != item['candidate_manifest_sha256']
                or candidate.get('state') != 'FROZEN_CANDIDATE'
                or candidate.get('snapshot_sha256') != item['snapshot_sha256']
                or candidate.get('counts', {}).get('documents') != item['documents']
                or not candidate.get('golden', {}).get('passed')
                or candidate.get('health', {}).get('integrity', {}).get('official_membership') != 'VERIFIED'
                or not candidate.get('health', {}).get('integrity', {}).get('passed')):
            raise ValueError('Recovery candidate is not the validated frozen snapshot.')
        inputs = checked(read_json(build / 'index_inputs/units.json'))
        if (inputs.get('state') != 'COMPLETE' or inputs['manifest_sha256'] != item['index_inputs_manifest_sha256']
                or inputs['candidate_manifest_sha256'] != item['candidate_manifest_sha256']
                or inputs['input_count'] != item['unique_dense_inputs']):
            raise ValueError('Recovery model-input binding changed.')
        record = {**item, 'build_dir': build.relative_to(project).as_posix()}
        cursor = 0
        for part in inputs['parts']:
            if (part['start'] != cursor or part['rows'] != min(4096, inputs['input_count']-cursor)
                    or part['signature'] != inputs['signature']):
                raise ValueError('Recovery input part coverage changed.')
            bound(build / 'index_inputs', part['path'])
            sources.append({'ordinal': len(sources), 'build': record, 'inputs': inputs, 'part': part})
            cursor += part['rows']
        if cursor != inputs['input_count']:
            raise ValueError('Recovery model-input range is incomplete.')
        builds.append(record)
    batch = seal({'state': 'RECOVERY_INPUTS_READY', 'frozen_deltas': frozen['manifest_sha256'],
                  'builds': builds, 'documents': sum(b['documents'] for b in builds),
                  'input_parts': len(sources),
                  'input_rows_before_cross_batch_dedup': sum(b['unique_dense_inputs'] for b in builds),
                  'scope': 'Frozen recovery deltas only; corpus embedding coverage is separate.'})
    return root, batch, sources
