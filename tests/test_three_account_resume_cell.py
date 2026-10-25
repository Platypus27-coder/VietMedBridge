"""Execute the delivered CPU cell on new-plan, waiting and drained-queue paths."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/04_colab_prepare_3_accounts_kaggle.ipynb'


def generated_three_account_cell():
    """Evaluate the actual generator using notebook-shaped stdlib objects."""
    baseline = json.loads((ROOT/'notebooks/04_colab_export_full_data_kaggle.ipynb').read_text(encoding='utf-8'))
    def base():
        return SimpleNamespace(cells=[SimpleNamespace(source=''.join(c['source'])) for c in copy.deepcopy(baseline['cells'])])
    tree = ast.parse((ROOT/'scripts/write_full_queue_notebooks.py').read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'three_account_transfer')
    scope = {'dedent': dedent, 'full_queue_transfer': base}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<three-account-generator>', 'exec'), scope)
    return scope['three_account_transfer']().cells[3].source


class ResumeCellTests(unittest.TestCase):
    def execute(self, plan, *, new=False, generated=False):
        source = generated_three_account_cell() if generated else ''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][3]['source'])
        events = []
        library = {'scheduled_tasks': 36491, 'unscheduled_tasks': 0}
        class Flow:
            def export_inputs(self, *args, **kwargs):
                events.append('reuse-library')
                return Path('library'), library
            def create_plan(self, *args, **kwargs):
                self_outer.assertEqual(kwargs['execution_accounts'], 3)
                self_outer.assertEqual(kwargs['replace_unstarted_plan'], 'confirmed-old-plan')
                events.append('resume-plan')
                return (Path('new-plan') if new else None), plan
        def upload_plan(*args):
            events.append('upload-new-plan')
            return 'owner/new-plan'
        self_outer = self
        scope = {'KAGGLE_NOTEBOOK_OUTPUT': '', 'KAGGLE_ACCOUNTS': 2,
            'EXECUTION_ACCOUNTS': 3, 'CONFIRMED_UNSTARTED_PLAN': 'confirmed-old-plan',
            'MIGRATE_UNSTARTED_LEGACY': True, 'WORK_DIR': ROOT/'.unused-cpu-cell-fixture',
            'CHECKOUT': ROOT/'.unused-checkout', 'config': {}, 'runtime': {},
            'lock': {'commit': 'producer'}, 'api_source': object(), 'owner': 'owner',
            'json': json, 'queue_flow': lambda *a, **k: Flow(),
            'queue': SimpleNamespace(ELASTIC_OUTPUT_BYTES=18000000000,
                upload_library=lambda *a: 'owner/existing-text-library', upload_plan=upload_plan)}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(source, '<delivered-cpu-cell>', 'exec'), scope)
        return output.getvalue(), events

    def test_waiting_plan_is_reused_without_task_list_or_reupload(self):
        plan = {'state': 'WAITING_FOR_KAGGLE_OUTPUTS', 'dataset_handle': 'owner/already-uploaded-plan',
                'imported_accounts': [], 'required_accounts': [0, 1, 2]}
        for generated in (False, True):
            with self.subTest(generated=generated):
                output, events = self.execute(plan, generated=generated)
                self.assertIn('CURRENT_PLAN_WAITING_FOR_OUTPUTS: https://www.kaggle.com/datasets/owner/already-uploaded-plan', output)
                self.assertNotIn('GPU_PLAN_DATASET_READY:', output)
                self.assertNotIn('GPU_PLAN_ASSIGNMENTS:', output)
                self.assertNotIn('upload-new-plan', events)

    def test_new_plan_reports_all_six_workers_once(self):
        plan = {'tasks': ['a', 'b', 'c', 'd', 'e', 'f'],
                'task_workers': dict(zip(['a', 'b', 'c', 'd', 'e', 'f'], range(6)))}
        for generated in (False, True):
            with self.subTest(generated=generated):
                output, events = self.execute(plan, new=True, generated=generated)
                self.assertIn('GPU_PLAN_ASSIGNMENTS: {0: 2, 1: 2, 2: 2}', output)
                self.assertEqual(events.count('upload-new-plan'), 1)
                self.assertEqual(output.count('GPU_PLAN_DATASET_READY:'), 1)

    def test_finished_queue_does_not_access_missing_dataset_handle(self):
        coverage = {f: {'completed_parts': 3024, 'total_parts': 3024} for f in ('bge', 'qwen')}
        for generated in (False, True):
            with self.subTest(generated=generated):
                output, events = self.execute({'state': 'NO_PENDING_KAGGLE_TASKS', 'coverage': coverage}, generated=generated)
                self.assertIn('FULL_DATA_EMBEDDINGS_COMPLETE', output)
                self.assertNotIn('GPU_PLAN_DATASET_READY:', output)
                self.assertNotIn('upload-new-plan', events)

    def test_incomplete_maps_are_never_reported_complete(self):
        coverage = {'qwen': {'completed_parts': 204, 'total_parts': 3024}}
        for generated in (False, True):
            with self.subTest(generated=generated), self.assertRaises(RuntimeError):
                self.execute({'state': 'NO_PENDING_KAGGLE_TASKS', 'coverage': coverage}, generated=generated)


if __name__ == '__main__':
    unittest.main()
