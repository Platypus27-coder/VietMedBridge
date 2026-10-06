"""Source/query review, resume and mining without pretending CPU doubles are gold."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from test_retrieval_cascade import Tokenizer
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
from vietmedbridge.negative_review import export_candidate_review, import_candidate_review
from vietmedbridge.reranker_training import mine_hard_negatives
from vietmedbridge.retrieval_data import Catalog
from vietmedbridge.training_data import (bind_source_plan, export_source_review,
    generate_training_drafts, plan_training_sources, prepare_training_data,
    publish_reviewed_labels, validate_draft)


@pytest.fixture
def sources():
    documents, children, parents, units, aliases = {}, {}, {}, [], {}
    for d in range(10, 18):
        text = f'Dữ liệu kiểm thử {d}: HbA1c được theo dõi trong nhóm {d}. Đoạn giả lập này chỉ kiểm hợp đồng phần mềm, không dùng làm kết luận y khoa.'
        checksum = hashlib.sha256(text.encode()).hexdigest()
        child_id, parent_id = f'child-{d}', f'parent-{d}'
        doc = {'doc_id':d,'title':f'Kiểm thử {d}','source_text':text,'source_text_sha256':checksum}
        span = {'chunk_id':child_id,'doc_id':d,'text':text,'source_text_sha256':checksum,
            'start_char':0,'end_char':len(text),'parent_id':parent_id}
        documents[d], children[child_id], parents[parent_id] = doc, span, {**span,'chunk_id':parent_id}
        units.append({'id':str(d),'text':text}); aliases[str(d)] = [child_id]
    return Catalog({'snapshot_sha256':'CPU-source-fixture','candidate_manifest_sha256':'CPU-source-fixture'},
        documents,children,parents,units,aliases,{'official_links_sha256':'CPU-official-fixture'})


@pytest.fixture
def policy():
    return {'train_samples':4,'dev_samples':2,'dev_fraction':.25,'max_children_per_document':1,
        'min_source_chars':20,'max_source_chars':1600,'min_train_reviewed':1,'min_dev_reviewed':1,
        'seed':42,'teacher_max_new_tokens':384}


class Teacher:
    def __init__(self, spec, fail_at=None):
        self.identity = {**spec,'test_double':True}
        self.calls, self.fail_at = 0, fail_at
    def translate(self, payload, prompt):
        assert set(payload) == {'passage'}
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('simulated interruption')
        number = payload['passage'].split('kiểm thử ')[1].split(':')[0]
        return json.dumps({'question_vi':f'HbA1c được theo dõi trong nhóm kiểm thử {number} như thế nào?',
                          'evidence_quote':payload['passage'][:70]},ensure_ascii=False)
    def close(self):
        pass


def reviewed_setup(tmp_path, sources, policy):
    plan = plan_training_sources(sources, [], policy)
    bind_source_plan(tmp_path, plan)
    spec = {'model_id':'CPU-TEACHER-DOUBLE','revision':'fixture','max_new_tokens':384}
    generate_training_drafts(tmp_path, plan, spec, [], Teacher(spec))
    path = export_source_review(tmp_path, plan)
    review = read_json(path)
    for item in review['items']:
        item['review'].update(decision='ACCEPT',reviewer='CPU-review-fixture-not-real-gold',
            query=f'Chỉ số HbA1c trong nhóm kiểm thử {item["doc_id"]} được mô tả thế nào?',
            evidence_quote=item['text'][:70],independently_written=item['split']=='dev')
    atomic_json(path, review)
    return plan, review


def test_duplicate_sources_share_fold_before_questions(sources, policy):
    duplicate = deepcopy(sources.documents[10]); duplicate['doc_id'] = 99
    sources.documents[99] = duplicate
    sources.children['child-99'] = {**sources.children['child-10'],'chunk_id':'child-99','doc_id':99}
    plan = plan_training_sources(sources, [], policy)
    assert plan['source_splits']['10'] == plan['source_splits']['99']
    train = {s['source_group'] for s in plan['samples'] if s['split']=='train'}
    dev = {s['source_group'] for s in plan['samples'] if s['split']=='dev'}
    assert not train & dev
    assert len({s['text'] for s in plan['samples']}) == len(plan['samples'])


def test_draft_disconnect_resume_dev_never_sent_to_teacher_and_tampering(tmp_path, sources, policy):
    plan = plan_training_sources(sources, [], policy)
    bind_source_plan(tmp_path, plan)
    spec = {'model_id':'CPU-TEACHER-DOUBLE','revision':'fixture'}
    with pytest.raises(RuntimeError,match='interruption'):
        generate_training_drafts(tmp_path, plan, spec, [], Teacher(spec,fail_at=2))
    teacher = Teacher(spec)
    result = generate_training_drafts(tmp_path, plan, spec, [], teacher)
    assert result['completed'] == 4 and teacher.calls == 3 and result['reviewed'] is False
    assert generate_training_drafts(tmp_path, plan, spec, [], None)['written_this_call'] == 0
    sample = next(s for s in plan['samples'] if s['split']=='train')
    path = tmp_path/'drafts'/(sample['item_id']+'.json')
    value = read_json(path); value['raw'] = 'changed'
    atomic_json(path,value)
    with pytest.raises(ValueError,match='integrity'):
        generate_training_drafts(tmp_path,plan,spec,[],None)


def test_bad_quote_or_contest_query_never_becomes_a_valid_draft():
    source = {'text':'HbA1c được theo dõi trong tài liệu kiểm thử phần mềm này.'}
    raw = json.dumps({'question_vi':'HbA1c được theo dõi như thế nào?','evidence_quote':source['text']})
    assert validate_draft(raw,source,[{'id':1,'query':'HBA1C được theo dõi như thế nào!!!'}])['state']=='DRAFT_REJECTED'
    wrong = json.dumps({'question_vi':'HbA1c được theo dõi như thế nào?','evidence_quote':'Đoạn bịa đặt không có trong source.'})
    assert validate_draft(wrong,source,[])['reason']=='QUESTION_OR_EXACT_SOURCE_QUOTE'


def test_pending_reviews_never_create_labels_and_immutable_source_is_checked(tmp_path,sources,policy):
    plan = plan_training_sources(sources,[],policy); bind_source_plan(tmp_path,plan)
    spec = {'model_id':'CPU-TEACHER-DOUBLE','revision':'fixture'}
    generate_training_drafts(tmp_path,plan,spec,[],Teacher(spec))
    path = export_source_review(tmp_path,plan)
    status = publish_reviewed_labels(tmp_path,plan,sources,[],tmp_path/'labels')
    assert status['pending']==6 and not (tmp_path/'labels/retrieval_train.json').exists()
    value = read_json(path); value['items'][0]['review']['notes']='keep this team edit'
    atomic_json(path,value); export_source_review(tmp_path,plan)
    assert read_json(path)['items'][0]['review']['notes']=='keep this team edit'
    value['items'][0]['text']='changed source'; atomic_json(path,value)
    with pytest.raises(ValueError,match='immutable'):
        publish_reviewed_labels(tmp_path,plan,sources,[],tmp_path/'labels')


def test_reviewed_publication_requires_independent_dev_and_tracks_provenance(tmp_path,sources,policy):
    plan, review = reviewed_setup(tmp_path,sources,policy)
    dev = next(i for i in review['items'] if i['split']=='dev')
    dev['review']['independently_written']=False
    atomic_json(tmp_path/'source_review.json',review)
    with pytest.raises(ValueError,match='independently written'):
        publish_reviewed_labels(tmp_path,plan,sources,[],tmp_path/'labels')
    dev['review']['independently_written']=True; atomic_json(tmp_path/'source_review.json',review)
    status = publish_reviewed_labels(tmp_path,plan,sources,[],tmp_path/'labels')
    train, dev_doc = read_json(status['train_path']),read_json(status['dev_path'])
    assert train['reviewed'] and dev_doc['reviewed'] and not train['exhaustive_chunks']
    assert all(q['provenance']['label_kind']=='HUMAN_REVIEWED_SYNTHETIC_TRAIN' for q in train['queries'])
    assert all(q['provenance']['label_kind']=='INDEPENDENT_HUMAN_DEV' for q in dev_doc['queries'])
    assert all(not q['negative_child_ids'] for q in train['queries'])


def test_exact_notebook_preparation_entrypoint_resume_and_no_reload_after_review(tmp_path,sources,policy,monkeypatch):
    from vietmedbridge import dataset,retrieval_data,translation_model
    import transformers
    project = Path(__file__).resolve().parents[1]
    checkout,data = tmp_path/'checkout',tmp_path/'data'
    for name in ('retrieval_full.json','strong_model_manifest.json'):
        atomic_json(checkout/'configs'/name,read_json(project/'configs'/name))
    atomic_json(checkout/'configs/training_data.json',policy)
    atomic_json(data/'raw/snapshot.json',{'files':{'links_corpus.parquet':{'sha256':'CPU-official-fixture'}}})
    monkeypatch.setattr(dataset,'parquet_path',lambda *a:Path('CPU-unused'))
    monkeypatch.setattr(retrieval_data,'load_queries',lambda *a,**k:[])
    monkeypatch.setattr(retrieval_data,'load_catalog',lambda *a,**k:sources)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    created=[]
    def teacher(spec):
        model=Teacher(spec);created.append(model);return model
    monkeypatch.setattr(translation_model,'TorchQueryTranslator',teacher)
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError,match='sample limit'):
            prepare_training_data(data,checkout,max_new_samples=invalid)
    paused=prepare_training_data(data,checkout,max_new_samples=0)
    assert paused['state']=='DRAFT_GENERATION_PARTIAL' and not created
    first=prepare_training_data(data,checkout)
    second=prepare_training_data(data,checkout)
    assert first['state']==second['state']=='WAITING_FOR_SOURCE_QUERY_REVIEW'
    assert len(created)==1 and created[0].calls==4
    assert first['model_parameter_budget']['total_parameters']==13_374_547_456
    review=read_json(first['review_path'])
    for item in review['items']:
        item['review'].update(decision='ACCEPT',reviewer='CPU-review-double',
            query=f'HbA1c trong nhóm kiểm thử {item["doc_id"]} được theo dõi thế nào?',
            evidence_quote=item['text'][:70],independently_written=item['split']=='dev')
    atomic_json(first['review_path'],review)
    assert prepare_training_data(data,checkout)['state']=='READY_FOR_HARD_NEGATIVE_REVIEW'
    assert prepare_training_data(data,checkout)['state']=='READY_FOR_HARD_NEGATIVE_REVIEW'
    assert len(created)==1


def negative_setup(tmp_path,sources,policy):
    plan,_=reviewed_setup(tmp_path/'source',sources,policy)
    status=publish_reviewed_labels(tmp_path/'source',plan,sources,[],tmp_path/'labels')
    train,dev=read_json(status['train_path']),read_json(status['dev_path'])
    records=[{'prediction':{'id':q['id']},'query_sha256':digest_json({'id':q['id'],'query':q['query']}),
        'ranking':{'children':[{'child_id':c} for c in sources.children]}} for d in (train,dev) for q in d['queries']]
    path=export_candidate_review(tmp_path/'run',records,train,dev,sources)
    return train,dev,records,path


def test_unknown_candidates_stay_unknown_then_review_makes_complete_groups(tmp_path,sources,policy):
    train,dev,records,path=negative_setup(tmp_path,sources,policy)
    t,d,status=import_candidate_review(tmp_path/'run',train,dev,sources,Tokenizer(),[])
    assert status['state']=='WAITING_FOR_CANDIDATE_REVIEW' and t==train and d==dev
    assert mine_hard_negatives(records,train,sources,Tokenizer(),[])['state']!='READY'
    review=read_json(path)
    positives={q['id']:q['provenance']['child_id'] for doc in (train,dev) for q in doc['queries']}
    for item in review['items']:
        item['review'].update(judgment='SKIP' if item['child_id']==positives[item['query_id']] else 'NEGATIVE',reviewer='CPU-review-double')
    atomic_json(path,review)
    t,d,status=import_candidate_review(tmp_path/'run',train,dev,sources,Tokenizer(),[])
    assert status['state']=='CANDIDATE_REVIEW_APPLIED'
    for doc in (t,d):
        mined=mine_hard_negatives(records,doc,sources,Tokenizer(),[])
        assert len(mined['groups'])==len(doc['queries']) and not mined['holds']
        assert all(len(g['pairs'])==8 for g in mined['groups'])
    changed=next(i for i in review['items'] if i['review']['judgment']=='NEGATIVE')
    changed['review']['judgment']='PENDING';atomic_json(path,review)
    t2,d2,status=import_candidate_review(tmp_path/'run',t,d,sources,Tokenizer(),[])
    affected=next(q for doc in (t2,d2) for q in doc['queries'] if q['id']==changed['query_id'])
    assert changed['child_id'] not in affected['negative_child_ids']


def test_candidate_source_tamper_and_negative_positive_conflict_are_rejected(tmp_path,sources,policy):
    train,dev,records,path=negative_setup(tmp_path,sources,policy)
    review=read_json(path);original=deepcopy(review)
    review['items'][0]['text']='edited source';atomic_json(path,review)
    with pytest.raises(ValueError,match='immutable'):
        import_candidate_review(tmp_path/'run',train,dev,sources,Tokenizer(),[])
    query=train['queries'][0]
    item=next(i for i in original['items'] if i['query_id']==query['id'] and i['child_id']==query['provenance']['child_id'])
    item['review'].update(judgment='NEGATIVE',reviewer='CPU-review-double');atomic_json(path,original)
    with pytest.raises(ValueError,match='conflicts'):
        import_candidate_review(tmp_path/'run',train,dev,sources,Tokenizer(),[])


def test_candidate_additional_positive_cannot_cross_reserved_dev_sources(tmp_path,sources,policy):
    train,dev,records,path=negative_setup(tmp_path,sources,policy)
    review=read_json(path)
    item=next(i for i in review['items'] if i['split']=='train' and train['source_splits'][str(i['doc_id'])]=='dev')
    item['review'].update(judgment='POSITIVE',reviewer='CPU-review-double');atomic_json(path,review)
    with pytest.raises(ValueError,match='reserved'):
        import_candidate_review(tmp_path/'run',train,dev,sources,Tokenizer(),[])


def test_exact_training_entrypoint_review_roundtrip_reuses_inference_before_training(tmp_path,sources,policy,monkeypatch):
    """Actual FAISS/mining/review path; external model/training calls are doubles."""
    from test_retrieval import Encoder
    from test_retrieval_cascade import CascadeReranker, Translator
    from test_full_plan import EncodeTokenizer
    from vietmedbridge import dataset, training_workflow as workflow, qwen_models, retrieval_models, translation_model
    from vietmedbridge.qwen_models import EMBEDDING_DIMENSION, QUERY_INSTRUCTION, RoleEncoder, pair_windows
    import transformers

    project=Path(__file__).resolve().parents[1]
    checkout,data=tmp_path/'checkout',tmp_path/'data'
    config=read_json(project/'configs/retrieval_full.json')
    config['lexical_segmentation']=False
    config['expansion_enabled']=False
    config['retrieval'].update(doc_candidate_k=8,sparse_top_k=8,detail_doc_k=8,doc_top_k=8)
    atomic_json(checkout/'configs/retrieval_full.json',config)
    atomic_json(checkout/'configs/strong_model_manifest.json',read_json(project/'configs/strong_model_manifest.json'))
    atomic_json(data/'raw/snapshot.json',{'files':{'links_corpus.parquet':{'sha256':'CPU-official-fixture'}}})
    plan,_=reviewed_setup(data/'preparation',sources,policy)
    publish_reviewed_labels(data/'preparation',plan,sources,[],data/'labels')
    monkeypatch.setattr(dataset,'parquet_path',lambda *a:Path('CPU-unused'))
    monkeypatch.setattr(workflow,'load_queries',lambda *a,**k:[])
    monkeypatch.setattr(workflow,'load_catalog',lambda *a,**k:sources)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    calls={'dense':0,'secondary':0,'translation':0,'ranking':0,'training':0}

    class Dense(Encoder):
        def __init__(self,spec):
            super().__init__()
            self.identity={**spec,'dimension':self.dimension,'pooling':'cls-l2-v1','truncation':False,
                'query_instruction':None,'precision':'torch.float16','device_class':'cuda',
                'inference_code_sha256':sha256_file(retrieval_models.__file__),'test_double':True}
        def encode(self,texts,**kwargs):
            calls['dense']+=1
            return super().encode(texts,**kwargs)
        def close(self):
            pass

    class Secondary(Encoder):
        dimension=EMBEDDING_DIMENSION
        def __init__(self,spec):
            super().__init__()
            self.identity={**spec,'dimension':self.dimension,'pooling':'last-attended-token-l2',
                'query_instruction':QUERY_INSTRUCTION,'truncation':False,'fine_tuned':False,
                'role':'second_dense','precision':'torch.float16','device_class':'cuda',
                'inference_code_sha256':sha256_file(qwen_models.__file__),'test_double':True}
        def for_role(self,role):
            return RoleEncoder(self,role)
        def encode(self,texts,**kwargs):
            calls['secondary']+=1
            return np.pad(super().encode(texts,**kwargs),((0,0),(0,self.dimension-3)))
        def close(self):
            pass

    class Translation(Translator):
        def __init__(self,spec):
            super().__init__()
            self.identity={**spec,'inference_code_sha256':sha256_file(translation_model.__file__),'test_double':True}
        def translate(self,query,prompt):
            calls['translation']+=1
            return super().translate(query,prompt)
        def close(self):
            pass

    class Ranking(CascadeReranker):
        def __init__(self,spec):
            super().__init__()
            self.spec,self.tokenizer=spec,EncodeTokenizer()
            self.identity={**spec,'fine_tuned':False,'test_double':True}
        def windows(self,query,text,overlap):
            return pair_windows(self.tokenizer,query,text,self.spec['max_length'],overlap)
        def score(self,pairs,**kwargs):
            calls['ranking']+=1
            return super().score(pairs,**kwargs)
        def close(self):
            pass

    class TrainingReached(Exception):
        pass
    def train(reranker,train_bundle,dev_bundle,output_dir,**kwargs):
        calls['training']+=1
        assert not train_bundle['holds'] and not dev_bundle['holds']
        assert all(len(g['pairs'])==8 for b in (train_bundle,dev_bundle) for g in b['groups'])
        assert kwargs['model_budget']['total_parameters']==13_374_547_456
        assert Path(output_dir).name==digest_json([read_json(data/'labels/retrieval_train.json'),read_json(data/'labels/retrieval_dev.json')])[:10]
        raise TrainingReached('CPU test stops before real GPU QLoRA')
    monkeypatch.setattr(workflow,'TorchDenseEncoder',Dense)
    monkeypatch.setattr(workflow,'TorchQwenEncoder',Secondary)
    monkeypatch.setattr(workflow,'TorchQueryTranslator',Translation)
    monkeypatch.setattr(workflow,'TorchQwenReranker',Ranking)
    monkeypatch.setattr(workflow,'train_qlora',train)

    first=workflow.run_training_workflow(data,checkout,work_dir=tmp_path/'work')
    assert first['state']=='MINING_REQUIRES_MORE_GOLD_COVERAGE_OR_REVIEWED_NEGATIVES'
    assert calls['training']==0 and len(first['train_holds'])==4 and len(first['dev_holds'])==2
    before=dict(calls)
    review=read_json(first['review_path'])
    positives={q['id']:q['provenance']['child_id'] for split in ('train','dev')
        for q in read_json(data/f'labels/retrieval_{split}.json')['queries']}
    for item in review['items']:
        item['review'].update(judgment='SKIP' if item['child_id']==positives[item['query_id']] else 'NEGATIVE',reviewer='CPU-review-double')
    atomic_json(first['review_path'],review)
    with pytest.raises(TrainingReached):
        workflow.run_training_workflow(data,checkout,work_dir=tmp_path/'work')
    assert {k:v for k,v in calls.items() if k!='training'}=={k:v for k,v in before.items() if k!='training'}
    assert calls['training']==1
    assert len(list((data/'training/stage-a-qlora-v2-15b/experiments').iterdir()))==1
