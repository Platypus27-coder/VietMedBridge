"""Real CPU FAISS/source contracts; model scores are explicitly test doubles."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from test_retrieval import Encoder, catalog
from test_retrieval_cascade import CascadeReranker, Tokenizer, Translator
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
from vietmedbridge.competition_pilot import MASTER_PLAN, bind_pilot_run, finish_pilot
from vietmedbridge.embeddings import embed_units, embedding_matrix
from vietmedbridge.document_dense import document_units
from vietmedbridge.full_plan_runtime import reviewed_calibration
from vietmedbridge.medical_lexical import MedicalAnalyzer
from vietmedbridge.query_expansion import (PICO_FIELDS, cached_expansions, expand_queries,
    is_complex, validate_expansion)
from vietmedbridge.query_translation import translate_queries
from vietmedbridge.qwen_models import (pack_pair, pair_windows, review_spec, RERANKER_ID, RERANKER_REVISION,
    TorchQwenEncoder, TorchQwenReranker, cached_qwen_embeddings, EMBEDDING_ID, EMBEDDING_REVISION,
    EMBEDDING_DIMENSION, QUERY_INSTRUCTION)
from vietmedbridge.reranker_training import mine_hard_negatives, ranking_metrics, validate_query_split
from vietmedbridge.retrieval_diagnostics import diagnose, run_dev_ablations
from vietmedbridge.sharded_dense import build_dense_shards, ShardedDenseIndex
from vietmedbridge.source_parents import derive_parents
from vietmedbridge.strong_retrieval import StrongConfig, StrongIndex, auxiliary_units, predict_strong
from vietmedbridge.training_workflow import run_training_workflow, _selected_adapter


class ExpansionTranslator(Translator):
    def translate(self, query, prompt):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('simulated disconnect')
        return json.dumps({'original_vi':query, 'pico':{k:[] for k in PICO_FIELDS},
                           'subqueries':[query], 'hyde_en':query})


def strong_pipeline(tmp_path, catalog, queries=None):
    tokenizer = Tokenizer()
    source = derive_parents(catalog, tokenizer)
    queries = queries or [{'id':123,'query':'HbA1c điều trị'}, {'id':900,'query':'病毒感染'}]
    encoder = Encoder()
    cm = embed_units(source.units, encoder, tmp_path/'corpus', part_size=1)
    qm = embed_units([{'id':q['id'],'text':q['query']} for q in queries], encoder, tmp_path/'qvectors')
    translations, _ = translate_queries(queries, Translator(), tmp_path/'translations')
    expansions = expand_queries(queries, ExpansionTranslator(), tmp_path/'expansions', translations=translations)
    aux = auxiliary_units(queries, expansions)
    secondary = Encoder()
    secondary.identity = {'model':'SECOND_DENSE_CPU_TEST_DOUBLE', 'input_role':'corpus'}
    sm = embed_units(source.units, secondary, tmp_path/'secondary', part_size=1)
    secondary.identity = {**secondary.identity, 'input_role':'query'}
    am = embed_units(aux, secondary, tmp_path/'aux')
    docs=document_units(source,tokenizer)
    dm=embed_units(docs,encoder,tmp_path/'documents')
    index = StrongIndex(source, embedding_matrix(tmp_path/'corpus', cm), cm, tmp_path/'index', tokenizer,
        secondary_vectors=embedding_matrix(tmp_path/'secondary', sm), secondary_manifest=sm,
        auxiliary_vectors=embedding_matrix(tmp_path/'aux', am), auxiliary_manifest=am,
        auxiliary_inputs=aux, queries=queries, expansions=expansions, analyzer=MedicalAnalyzer(segmentation=False),
        document_vectors=embedding_matrix(tmp_path/'documents',dm),document_manifest=dm,document_inputs=docs)
    config = StrongConfig(doc_candidate_k=3, sparse_top_k=3, detail_doc_k=3, doc_top_k=3,
        doc_min_k=1, chunk_top_k=3, chunk_min_k=1, children_per_doc=1)
    return index, queries, qm, embedding_matrix(tmp_path/'qvectors', qm), translations, config


def test_strong_all_1200_queries_resume_source_export_and_diagnostics(tmp_path, catalog):
    queries = [{'id':100001+i*7,'query':f'HbA1c xét nghiệm số {i}'} for i in range(1200)]
    index, queries, qm, vectors, translations, policy = strong_pipeline(tmp_path, catalog, queries)
    reranker = CascadeReranker()
    first, report = predict_strong(index, queries, vectors, reranker, tmp_path/'pred',
        query_embedding_manifest=qm, translations=translations, config=policy, max_new_queries=17)
    assert len(first) == 17 and report['state'] == 'IN_PROGRESS'
    fresh = CascadeReranker()
    records, report = predict_strong(index, queries, vectors, fresh, tmp_path/'pred',
        query_embedding_manifest=qm, translations=translations, config=policy)
    assert report['state'] == 'COMPLETE' and fresh.calls == 2*(1200-17)
    repeat = CascadeReranker()
    again, repeated_report = predict_strong(index, queries, vectors, repeat, tmp_path/'pred',
        query_embedding_manifest=qm, translations=translations, config=policy)
    assert again == records and repeated_report == report and repeat.calls == 0
    diagnostic = diagnose(records, queries, index, tmp_path/'diagnostic.json')
    assert diagnostic['candidate_branch_occurrences']['dense_qwen'] > 0
    assert diagnostic['candidate_branch_occurrences']['dense_document'] > 0
    assert diagnostic['quality_state'] == 'NOT_EVALUATED_NO_REFERENCE_LABELS'
    for record in records:
        assert record['prediction']['relevant_chunks']
        assert all(p['parent_id'].startswith('source-parent-') for p in record['provenance'])
    plan = tmp_path/MASTER_PLAN
    plan.write_text('CPU fixture, not competition inference', encoding='utf-8')
    contract = bind_pilot_run(tmp_path/'run', plan_path=plan, catalog=index.catalog, queries=queries,
        config={'expected_queries':1200, 'architecture':'CPU_FULL_ARCHITECTURE_TEST_DOUBLE'}, code_commit='test')
    _, ready = finish_pilot(records, queries, index.catalog, report, tmp_path/'run', contract=contract,
        tokenizer=index.tokenizer, evidence={'inference_scope':'CPU_TEST_DOUBLES_DO_NOT_SUBMIT'})
    assert ready['query_count'] == 1200 and ready['official_score'] is None
    # Stage evidence must remain attached to the same model, query run and pairs.
    stage = tmp_path/f"pred/query-{queries[0]['id']}-children.done.json"
    saved = read_json(stage)
    saved['signature'] = 'other-run'
    saved['record_sha256'] = digest_json({k:v for k,v in saved.items() if k != 'record_sha256'})
    atomic_json(stage, saved)
    with pytest.raises(ValueError, match='evidence'):
        predict_strong(index, queries, vectors, repeat, tmp_path/'pred', query_embedding_manifest=qm,
            translations=translations, config=policy)


def test_strong_stage_interruption_and_policy_or_id_changes(tmp_path, catalog):
    index, queries, qm, vectors, translations, config = strong_pipeline(tmp_path, catalog)
    with pytest.raises(RuntimeError, match='disconnect'):
        predict_strong(index, queries, vectors, CascadeReranker(fail_at=2), tmp_path/'pred',
            query_embedding_manifest=qm, translations=translations, config=config)
    resumed = CascadeReranker()
    records, _ = predict_strong(index, queries, vectors, resumed, tmp_path/'pred',
        query_embedding_manifest=qm, translations=translations, config=config)
    assert resumed.calls == 3 and len(records) == 2
    with pytest.raises(ValueError, match='changed'):
        predict_strong(index, queries, vectors, resumed, tmp_path/'pred', query_embedding_manifest=qm,
            translations=translations, config=replace(config, hyde_weight=.4))
    with pytest.raises(ValueError, match='unique integer'):
        predict_strong(index, [queries[0],queries[0]], vectors, resumed, tmp_path/'duplicate',
            query_embedding_manifest=qm, translations=translations, config=config)
    with pytest.raises(ValueError, match='limit'):
        predict_strong(index, queries, vectors, resumed, tmp_path/'negative', query_embedding_manifest=qm,
            translations=translations, config=config, max_new_queries=-1)


def test_exact_source_adaptive_parents_keep_anchor_and_original_catalog(catalog):
    text = ' '.join(f'word{i}' for i in range(1300))
    doc = catalog.documents[583]
    doc.update(source_text=text, source_text_sha256=hashlib.sha256(text.encode()).hexdigest())
    tokens = Tokenizer()(text)['offset_mapping']
    a, b = tokens[550][0], tokens[729][1]
    catalog.children['child-583'].update(text=text[a:b], start_char=a, end_char=b, source_text_sha256=doc['source_text_sha256'])
    derived = derive_parents(catalog, Tokenizer())
    assert 'source_parent_alternatives' not in catalog.children['child-583']
    for budget, pid in derived.children['child-583']['source_parent_alternatives'].items():
        parent = derived.parents[pid]
        assert parent['text'] == text[parent['start_char']:parent['end_char']]
        assert parent['start_char'] <= a < b <= parent['end_char']
        assert len(Tokenizer()(parent['text'])['input_ids']) <= int(budget)
    assert derived.identity['derived_parent_policy_sha256']


def test_source_parent_retokenization_boundary_overflow_trims_context(catalog):
    text = ' '.join(f'word{i}' for i in range(1300))
    catalog.documents[583]['source_text'] = text
    offsets = Tokenizer()(text)['offset_mapping']
    a, b = offsets[550][0], offsets[729][1]
    catalog.children['child-583'].update(text=text[a:b], start_char=a, end_char=b)
    class BoundaryTokenizer(Tokenizer):
        def __call__(self, value, **kwargs):
            encoded = super().__call__(value, **kwargs)
            if 450 < len(encoded['input_ids']) < 1300:
                encoded['input_ids'] += ['boundary-token']*20
            return encoded
    tokenizer = BoundaryTokenizer()
    result = derive_parents(catalog, tokenizer)
    for budget, pid in result.children['child-583']['source_parent_alternatives'].items():
        assert len(tokenizer(result.parents[pid]['text'])['input_ids']) <= int(budget)
        assert text[a:b] in result.parents[pid]['text']


def test_adaptive_complex_query_uses_declared_640_parent(tmp_path, catalog):
    queries = [{'id':321,'query':'So sánh HbA1c và diabetes trong nhóm người bệnh dài hạn có nhiều yếu tố liên quan đến hiệu quả can thiệp và kết quả xét nghiệm tại bệnh viện'}]
    index, queries, qm, vectors, translations, config = strong_pipeline(tmp_path, catalog, queries)
    records, _ = predict_strong(index, queries, vectors, CascadeReranker(), tmp_path/'pred',
        query_embedding_manifest=qm, translations=translations, config=config)
    assert records[0]['provenance']
    assert all(index.catalog.parents[p['parent_id']]['token_budget'] == 640 for p in records[0]['provenance'])


def test_expansion_additive_guard_complex_only_resume_and_cache(tmp_path):
    query = 'So sánh ảnh hưởng của HbA1c > 7 ở người mang thai không dùng thuốc và theo dõi kết quả trong quần thể nghiên cứu dài hạn tại bệnh viện'
    assert is_complex(query)
    english = 'Compare HbA1c > 7 effects during pregnancy without medication and follow up long term outcomes at the hospital'
    raw = {'original_vi':query, 'pico':{k:[] for k in PICO_FIELDS}, 'subqueries':[english], 'hyde_en':english.replace('7','8')}
    result = validate_expansion(query, json.dumps(raw))
    assert result['subqueries'] == [english] and result['hyde_en'] is None
    assert result['rejections']
    raw['pico']['population'] = ['invented source']
    assert validate_expansion(query, json.dumps(raw))['rejections'] == ['SCHEMA_OR_SOURCE_ANCHOR']
    queries = [{'id':1,'query':'HbA1c'}, {'id':2,'query':query}, {'id':3,'query':query+' thêm'}]
    translations, _ = translate_queries(queries, Translator(), tmp_path/'translations')
    with pytest.raises(RuntimeError, match='disconnect'):
        expand_queries(queries, ExpansionTranslator(fail_at=2), tmp_path/'expansion', translations=translations)
    resumed = ExpansionTranslator()
    records = expand_queries(queries, resumed, tmp_path/'expansion', translations=translations)
    assert resumed.calls == 1 and not records[0]['active']
    assert cached_expansions(queries, translations, {}, tmp_path/'expansion') == records
    changed = deepcopy(records[0]); changed['variants']['hyde_en'] = 'invented'
    atomic_json(tmp_path/'expansion/query-1.done.json', changed)
    with pytest.raises(ValueError, match='mismatch'):
        cached_expansions(queries, translations, {}, tmp_path/'expansion')


class EncodeTokenizer(Tokenizer):
    def encode(self, text, **kwargs):
        return self(text)['input_ids']


def test_qwen_chat_budget_includes_prefix_query_suffix_and_retains_tail():
    tokenizer = EncodeTokenizer()
    text = 'head '*300+'tail evidence'
    windows = pair_windows(tokenizer, 'long query '*20, text, 160)
    assert 'tail evidence' in windows[-1]['text']
    assert all(text[w['start_char']:w['end_char']] == w['text'] for w in windows)
    assert all(len(pack_pair(tokenizer, 'long query '*20, w['text'], 160)) <= 160 for w in windows)
    with pytest.raises(ValueError, match='budget'):
        pack_pair(tokenizer, 'q '*200, text, 160)
    with pytest.raises(ValueError, match='reviewed pinned'):
        review_spec({'model_id':'private/unreviewed','revision':'main','quantization':'nf4','max_length':1024}, 'reranker')


class FakeTensor(np.ndarray):
    """CPU-only tensor double; .to(cuda) records intent without using a GPU."""
    def __new__(cls, values):
        return np.asarray(values).view(cls)
    def to(self, device):
        assert device == 'cuda'
        return self
    def float(self):
        return self.astype(np.float32)
    def cpu(self):
        return self
    def numpy(self):
        return np.asarray(self)


class BatchTokenizer(EncodeTokenizer):
    def pad(self, data, **kwargs):
        rows = data['input_ids']
        width = max(map(len, rows))
        return {'input_ids':FakeTensor([[0]*(width-len(r))+[1]*len(r) for r in rows]),
                'attention_mask':FakeTensor([[0]*(width-len(r))+[1]*len(r) for r in rows])}
    def __call__(self, value, **kwargs):
        if isinstance(value, list):
            self.last_texts = value
            return self.pad({'input_ids':[super(BatchTokenizer,self).__call__(t)['input_ids'] for t in value]})
        return super().__call__(value, **kwargs)


def test_actual_qwen_batch_methods_last_token_pooling_and_raw_yes_no_logits():
    tokenizer = BatchTokenizer()
    fake_torch = SimpleNamespace(inference_mode=nullcontext, all=np.all,
        nn=SimpleNamespace(functional=SimpleNamespace(normalize=lambda x,p,dim:FakeTensor(x/np.linalg.norm(x,axis=dim,keepdims=True)))))
    encoder = object.__new__(TorchQwenEncoder)
    encoder.tokenizer, encoder.torch = tokenizer, fake_torch
    encoder.spec, encoder.input_role = {'max_length':512}, 'query'
    class Model:
        def __call__(self, **inputs):
            self.inputs = inputs
            n, width = inputs['input_ids'].shape
            hidden = np.zeros((n,width,3),np.float32)
            hidden[:,-1] = [[0,3,4],[1,0,0]]
            return SimpleNamespace(last_hidden_state=FakeTensor(hidden), logits=FakeTensor([[[0,2,9,0]],[[0,5,2,0]]]))
    model = Model(); encoder.model = model
    vectors = encoder._batch(['HbA1c','virus query longer'])
    assert np.allclose(vectors, [[0,.6,.8],[1,0,0]])
    assert all(t.startswith('Instruct: '+QUERY_INSTRUCTION) for t in tokenizer.last_texts)
    encoder.input_role = 'corpus'
    encoder._batch(['HbA1c','virus query longer'])
    assert tokenizer.last_texts == ['HbA1c','virus query longer']
    reranker = object.__new__(TorchQwenReranker)
    reranker.tokenizer, reranker.torch, reranker.model = tokenizer, fake_torch, model
    reranker.spec, reranker.yes, reranker.no = {'max_length':512}, 1, 2
    assert reranker._batch([('q','p'),('q longer','p longer')]).tolist() == [-7,3]
    assert model.inputs['logits_to_keep'] == 1 and model.inputs['use_cache'] is False


def test_qwen_vector_cache_binds_pooling_role_and_original_producer(tmp_path):
    from vietmedbridge import qwen_models
    spec = {'model_id':EMBEDDING_ID,'revision':EMBEDDING_REVISION,'max_length':1536,'quantization':'fp16'}
    encoder = Encoder()
    encoder.dimension = EMBEDDING_DIMENSION
    encoder.identity = {**spec,'input_role':'corpus','query_instruction':QUERY_INSTRUCTION,'fine_tuned':False,
        'pooling':'last-attended-token-l2','truncation':False,'role':'second_dense','dimension':EMBEDDING_DIMENSION,
        'precision':'torch.float16','device_class':'cuda','inference_code_sha256':sha256_file(qwen_models.__file__)}
    encoder.encode = lambda texts, batch_size: np.pad(np.array([[1,0,0]]*len(texts),dtype=np.float32),((0,0),(0,EMBEDDING_DIMENSION-3)))
    units = [{'id':1,'text':'HbA1c'}]
    manifest = embed_units(units, encoder, tmp_path)
    assert cached_qwen_embeddings(tmp_path, units, spec, 'corpus')[1] == manifest
    with pytest.raises(ValueError, match='mismatch'):
        cached_qwen_embeddings(tmp_path, units, spec, 'query')
    broken = deepcopy(manifest); broken['encoder']['pooling'] = 'mean'
    broken['manifest_sha256'] = digest_json({k:v for k,v in broken.items() if k != 'manifest_sha256'})
    atomic_json(tmp_path/'embeddings.json', broken)
    with pytest.raises(ValueError, match='mismatch'):
        cached_qwen_embeddings(tmp_path, units, spec, 'corpus')


def test_real_vi_zh_segmentation_keeps_original_medical_tokens():
    pytest.importorskip('pyvi'); pytest.importorskip('jieba')
    analyzer = MedicalAnalyzer()
    vi = analyzer.analyze('HbA1c xét nghiệm tại bệnh viện','vi')
    zh = analyzer.analyze('病毒感染检测 HbA1c','zh')
    assert {'hba1c','xét_nghiệm','bệnh_viện'} <= set(vi)
    assert {'hba1c','病毒'} <= set(zh)


def test_reviewed_glossary_is_soft_and_keeps_original_tokens(tmp_path):
    glossary = tmp_path/'glossary.json'
    atomic_json(glossary, {'reviewed':True,'entries':[{'aliases':['HbA1c','glycated hemoglobin'], 'source':'reviewed fixture'}]})
    analyzer = MedicalAnalyzer(segmentation=False, glossary_path=glossary)
    assert 'hba1c' in analyzer.analyze('HbA1c > 7')
    assert 'glycated hemoglobin' in analyzer.aliases('HbA1c > 7')
    assert analyzer.identity['aliases'] == 1
    atomic_json(glossary, {'reviewed':False,'entries':[]})
    with pytest.raises(ValueError, match='reviewed'):
        MedicalAnalyzer(segmentation=False, glossary_path=glossary)


def test_dense_shards_resume_match_flat_faiss_and_detect_corruption(tmp_path):
    import faiss
    units = [{'id':i,'text':('HbA1c' if i % 2 else 'virus')} for i in range(13)]
    manifest = embed_units(units, Encoder(), tmp_path/'vectors', part_size=3)
    partial = build_dense_shards(tmp_path/'vectors', manifest, tmp_path/'shards', rows_per_shard=4, max_new_shards=1)
    assert partial['state'] == 'IN_PROGRESS' and len(partial['parts']) == 1
    with pytest.raises(ValueError, match='Complete every'):
        ShardedDenseIndex(tmp_path/'shards')
    complete = build_dense_shards(tmp_path/'vectors', manifest, tmp_path/'shards', rows_per_shard=4)
    assert complete['state'] == 'COMPLETE' and [p['start'] for p in complete['parts']] == [0,4,8,12]
    vectors = embedding_matrix(tmp_path/'vectors', manifest)
    flat = faiss.IndexFlatIP(3); flat.add(vectors)
    shard = ShardedDenseIndex(tmp_path/'shards')
    queries = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
    values, ids = shard.search(queries, 7)
    assert np.allclose(values, flat.search(queries,7)[0])
    assert np.allclose(np.sum(vectors[ids]*queries[:,None,:], axis=2), values)
    (tmp_path/'shards'/complete['parts'][0]['file']).write_bytes(b'broken')
    with pytest.raises(ValueError, match='changed artifact'):
        shard.search(queries, 7)


def labeled(query_id, query, doc=583, text='HbA1c'):
    return {'id':query_id,'query':query,'relevant_docs':[doc], 'relevant_chunks':[{'doc_id':doc,'chunk_text':text}]}


def test_training_query_leakage_missing_labels_and_metric_ties(tmp_path):
    train = {'reviewed':True,'split':'train','queries':[labeled(100,'independent train')]}
    dev = {'reviewed':True,'split':'dev','queries':[labeled(200,'independent dev')]}
    validate_query_split(train, dev, [{'id':1,'query':'contest'}])
    with pytest.raises(ValueError, match='Contest'):
        validate_query_split(train, dev, [{'id':1,'query':'independent train'}])
    with pytest.raises(ValueError, match='overlap'):
        validate_query_split(train, {**dev,'queries':train['queries']}, [])
    status = run_training_workflow(tmp_path, tmp_path/'no-checkout-required')
    assert status['state'] == 'WAITING_FOR_INDEPENDENT_REVIEWED_TRAIN_DEV_LABELS'
    assert status['fine_tuned'] is False and not list(tmp_path.rglob('*.safetensors'))
    metrics = ranking_metrics([1,1], [{'query_id':1,'label':0},{'query_id':1,'label':1}])
    assert metrics['mrr_at_10'] == .5


def test_mining_long_gold_parent_positive_and_only_reviewed_negatives(catalog):
    tokenizer = Tokenizer()
    text = ' '.join(f'token{i}' for i in range(600))
    catalog.documents[583]['source_text'] = text
    catalog.children['positive'] = {'doc_id':583,'text':' '.join(f'token{i}' for i in range(180))}
    rows = [{'child_id':'positive'}]
    for idx in range(8):
        cid = f'negative{idx}'
        catalog.children[cid] = {'doc_id':991,'text':f'virus unrelated {idx}'}
        rows.append({'child_id':cid})
    query = labeled(20,'independent biomedical query',text=text)
    labels = {'split':'train','reviewed':True, 'queries':[query]}
    record = {'prediction':{'id':20}, 'query_sha256':digest_json({'id':20,'query':query['query']}), 'ranking':{'children':rows}}
    none = mine_hard_negatives([record], labels, catalog, tokenizer, [])
    assert none['state'] != 'READY' and len(none['evaluation_pairs']) == 1
    query['negative_child_ids'] = [f'negative{i}' for i in range(7)]
    bundle = mine_hard_negatives([record], labels, catalog, tokenizer, [])
    assert bundle['state'] == 'READY' and len(bundle['groups'][0]['pairs']) == 8
    assert bundle['groups'][0]['pairs'][0]['child_id'] == 'positive'
    assert len(bundle['evaluation_pairs']) == 8  # unknown negative8 is not gold.
    query['relevant_chunks'][0]['chunk_text'] = 'invented gold'
    with pytest.raises(ValueError, match='exact source'):
        mine_hard_negatives([record], labels, catalog, tokenizer, [])


def test_selected_adapter_and_calibration_bind_exact_files_policy_and_queries(tmp_path, catalog):
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    for name in ('adapter_config.json','adapter_model.safetensors'):
        (checkpoint/name).write_bytes(b'CPU fixture no actual model')
    from vietmedbridge.reranker_training import write_adapter_manifest
    write_adapter_manifest(checkpoint, labels={'reviewed_fixture':True}, training_contract={'split':'dev'})
    from vietmedbridge.reranker_training import verify_training_checkpoint
    assert verify_training_checkpoint(checkpoint, {'split':'dev'})['fine_tuned']
    with pytest.raises(ValueError, match='contract'):
        verify_training_checkpoint(checkpoint, {'split':'changed'})
    selected = _selected_adapter(checkpoint, tmp_path/'selected_adapter')
    queries = [{'id':1,'query':'contest'}]
    spec = {'model_id':RERANKER_ID,'revision':RERANKER_REVISION}
    calibration = {'split':'dev','reviewed':True,'catalog':catalog.identity,'reranker_spec':spec,
        'contest_queries_sha256':digest_json(queries),'adapter_manifest_sha256':selected['manifest_sha256'],
        'calibration_context':{'fixture':'context'},'dev_query_ids':[2], 'dev_query_texts_sha256':[digest_json('independent')],
        'policy':{'doc_top_k':10,'chunk_top_k':8,'doc_score_margin':None,'chunk_score_margin':None}}
    calibration['manifest_sha256'] = digest_json(calibration)
    atomic_json(tmp_path/'calibrated_policy.json', calibration)
    assert reviewed_calibration(tmp_path, catalog, queries, spec, context={'fixture':'context'})[1] == tmp_path/'selected_adapter'
    with pytest.raises(ValueError, match='match'):
        reviewed_calibration(tmp_path, catalog, queries, spec, context={'fixture':'changed'})
    (checkpoint/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed artifact'):
        _selected_adapter(checkpoint, tmp_path/'second')


@pytest.mark.parametrize('recall', [None, float('nan'), -1., 1.])
def test_embedding_training_waits_for_measured_recall_bottleneck(recall, tmp_path):
    from vietmedbridge.embedding_training import train_embedding_qlora
    with pytest.raises(ValueError, match='measured candidate-recall'):
        train_embedding_qlora(None, {}, {}, tmp_path, recall_report={'fused_document_pool_recall_macro':recall})


def test_dev_ablations_run_all_profiles_and_forbid_contest_tuning(tmp_path, catalog):
    index, queries, qm, vectors, translations, policy = strong_pipeline(tmp_path, catalog)
    labels = {'split':'dev','reviewed':True,'queries':[labeled(q['id'],q['query'],583,index.catalog.parents['parent-583']['text']) for q in queries]}
    results = run_dev_ablations(index, queries, vectors, qm, translations, CascadeReranker(), policy,
        labels, [], tmp_path/'ablations')
    assert len(results) == 8 and all('combined_f2' in r for r in results)
    with pytest.raises(ValueError, match='Contest'):
        run_dev_ablations(index, queries, vectors, qm, translations, CascadeReranker(), policy, labels, queries, tmp_path/'leak')


def test_full_notebook_training_is_separate_no_action_and_pinned_models():
    root = Path(__file__).resolve().parents[1]
    for name in ('04_colab_retrieval_baseline.ipynb','05_colab_supervised_training.ipynb'):
        notebook = json.loads((root/'notebooks'/name).read_text(encoding='utf-8'))
        source = '\n'.join(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code')
        assert '.[notebook,retrieval,strong]' in source
        expected_api = 'full-master-plan-strong-v6-large-baseline' if name.startswith('04') else 'full-master-plan-supervised-v6-shared-cache-review'
        assert expected_api in source and 'git' in source
        assert 'ACTION =' not in source
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                assert cell['outputs'] == [] and cell['execution_count'] is None
                compile(cell['source'], name, 'exec')
    config = read_json(root/'configs/retrieval_full.json')
    registry = read_json(root/'configs/strong_model_manifest.json')
    for key in ('second_dense','reranker'):
        assert any(m['model_id'] == config[key]['model_id'] and m['revision'] == config[key]['revision']
                   and m['parameters'] <= 15_000_000_000 and m['public_release'] < '2026-08-01'
                   for m in registry['models'])
    from vietmedbridge.qwen_models import review_model_registry
    assert review_model_registry(config, registry) == digest_json(registry)
    bad = deepcopy(registry); bad['models'][0]['parameters'] = 16_000_000_000
    with pytest.raises(ValueError, match='registry'):
        review_model_registry(config, bad)


def test_main_colab_entrypoint_wiring_all_queries_and_cache_resume(tmp_path, catalog, monkeypatch):
    """Exercise the exact notebook entrypoint; all model loaders are CPU doubles."""
    from vietmedbridge import full_plan_runtime as runtime, qwen_models, translation_model, retrieval_models
    from vietmedbridge.qwen_models import RoleEncoder
    import transformers
    root = Path(__file__).resolve().parents[1]
    checkout, data = tmp_path/'checkout', tmp_path/'data'
    checkout.mkdir(); (checkout/'configs').mkdir()
    (checkout/MASTER_PLAN).write_text('CPU runtime wiring fixture, never submit its predictions',encoding='utf-8')
    config = read_json(root/'configs/retrieval_full.json')
    config['lexical_segmentation'] = False
    config['retrieval'].update(doc_candidate_k=3, sparse_top_k=3, detail_doc_k=3, doc_top_k=3)
    atomic_json(checkout/'configs/retrieval_full.json', config)
    atomic_json(checkout/'configs/strong_model_manifest.json', read_json(root/'configs/strong_model_manifest.json'))
    catalog.build_config = {'official_links_sha256':'official-fixture'}
    catalog.candidate['golden'] = {'tokenizer':config['dense']}
    atomic_json(data/'raw/snapshot.json', {'files':{'links_corpus.parquet':{'sha256':'official-fixture'}}})
    queries = [{'id':50001+i*3,'query':f'HbA1c xét nghiệm {i}'} for i in range(1200)]
    monkeypatch.setattr(runtime, 'load_catalog', lambda *a, **k:catalog)
    monkeypatch.setattr(runtime, 'inference_batches', lambda: {'embedding':2,'reranker':2})
    monkeypatch.setattr(runtime, 'load_queries', lambda *a, **k:queries)
    monkeypatch.setattr(runtime, 'parquet_path', lambda *a:Path('unused-CPU-fixture'))
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k:Tokenizer())
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda:True, get_device_name=lambda *a:'CPU_TENSOR_DOUBLE_NOT_GPU')))
    calls = {'primary':0,'translation':0,'secondary':0,'reranking':0}

    class Dense(Encoder):
        def __init__(self, spec):
            super().__init__()
            self.identity = {**spec, 'pooling':'cls-l2-v1','precision':'torch.float16', 'device_class':'cuda',
                'dimension':self.dimension,'truncation':False, 'query_instruction':None,'inference_code_sha256':sha256_file(retrieval_models.__file__),
                'test_double':True}
        def encode(self, texts, **kwargs):
            calls['primary'] += 1
            return super().encode(texts, **kwargs)
        def close(self):
            pass

    class Secondary(Encoder):
        dimension = EMBEDDING_DIMENSION
        def __init__(self, spec):
            super().__init__()
            self.identity = {**spec, 'pooling':'last-attended-token-l2','precision':'torch.float16', 'device_class':'cuda',
                'truncation':False,'query_instruction':QUERY_INSTRUCTION,'fine_tuned':False,'role':'second_dense',
                'dimension':EMBEDDING_DIMENSION,'inference_code_sha256':sha256_file(qwen_models.__file__), 'test_double':True}
            self.oom_backoffs = 0
        def for_role(self, role):
            return RoleEncoder(self, role)
        def encode(self, texts, **kwargs):
            calls['secondary'] += 1
            return np.pad(super().encode(texts, **kwargs),((0,0),(0,EMBEDDING_DIMENSION-3)))
        def close(self):
            pass

    class Translation(Translator):
        def __init__(self, spec):
            super().__init__()
            self.identity = {**spec,'inference_code_sha256':sha256_file(translation_model.__file__), 'test_double':True}
        def translate(self, query, prompt):
            calls['translation'] += 1
            return super().translate(query, prompt)
        def close(self):
            pass

    class Ranking(CascadeReranker):
        def __init__(self, spec, adapter_path=None):
            super().__init__()
            assert adapter_path is None
            self.spec, self.tokenizer, self.oom_backoffs = spec, EncodeTokenizer(), 0
            self.identity = {**spec,'fine_tuned':False,'test_double':True}
        def windows(self, query, text, overlap):
            return pair_windows(self.tokenizer, query, text, self.spec['max_length'], overlap)
        def score(self, pairs, **kwargs):
            calls['reranking'] += 1
            return super().score(pairs, **kwargs)
        def close(self):
            pass

    monkeypatch.setattr(runtime,'TorchDenseEncoder', Dense)
    monkeypatch.setattr(runtime,'TorchQwenEncoder', Secondary)
    monkeypatch.setattr(runtime,'TorchQueryTranslator', Translation)
    monkeypatch.setattr(runtime,'TorchQwenReranker', Ranking)
    result = runtime.run_full_pipeline(data, checkout, code_commit='CPU-test', work_dir=tmp_path/'work')
    assert result['ready']['query_count'] == result['status']['query_count'] == 1200
    assert result['ready']['official_score'] is None
    assert result['status']['model_parameter_budget']['total_parameters'] == 20_346_066_432
    assert result['status']['fine_tuning'] == 'WAITING_FOR_INDEPENDENT_REVIEWED_TRAIN_DEV_LABELS'
    before = dict(calls)
    repeated = runtime.run_full_pipeline(data, checkout, code_commit='CPU-test', work_dir=tmp_path/'work')
    assert repeated['ready']['zip_sha256'] == result['ready']['zip_sha256']
    assert calls == before  # all encode/LLM/score checkpoints reused by actual entrypoint.
