"""Source/query review, resume and mining without pretending CPU doubles are gold."""
from copy import deepcopy
import hashlib
import json
import struct
import sys
from types import SimpleNamespace
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
    publish_reviewed_labels, validate_draft, install_ai_pilot_review, reuse_completed_review_plan)


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
            evidence_quote=item['text'][:70],independently_written=item['split']!='train')
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


def test_explicit_ai_pilot_publishes_only_reviewed_subset_without_claiming_human_gold(tmp_path,sources,policy):
    policy = {**policy,'heldout_samples':2,'heldout_fraction':.25,'min_heldout_reviewed':1}
    plan,review = reviewed_setup(tmp_path,sources,policy)
    review.update(review_mode='AI_ASSISTED_PILOT',review_authorization='USER_DELEGATED_TO_CODEX_2026_10_06')
    chosen = set()
    for item in review['items']:
        if item['split'] in chosen:
            item['review'].update(decision='PENDING',reviewer='',independently_written=False)
        else:
            chosen.add(item['split'])
            item['review'].update(reviewer='AI CPU review fixture',reviewer_type='AI',query_author_type='AI',independently_written=False)
    atomic_json(tmp_path/'source_review.json',review)
    result = publish_reviewed_labels(tmp_path,plan,sources,[],tmp_path/'labels')
    assert result['state']=='READY_FOR_HARD_NEGATIVE_REVIEW'
    assert result['accepted']=={'train':1,'dev':1,'heldout':1}
    assert result['deferred_source_items']==5 and result['human_validated'] is False
    for split in ('train','dev','heldout'):
        label = read_json(tmp_path/'labels'/f'retrieval_{split}.json')
        assert len(label['queries'])==1 and label['human_validated'] is False
        assert label['review_mode']=='AI_ASSISTED_PILOT'
        assert label['queries'][0]['provenance']['label_kind']==f'AI_REVIEWED_PILOT_{split.upper()}'
    assert sum(i['review']['decision']=='PENDING' for i in read_json(tmp_path/'source_review.json')['items'])==5


def test_ai_pilot_rejects_false_human_claims_and_still_requires_minimums_and_exact_quotes(tmp_path,sources,policy):
    plan,review = reviewed_setup(tmp_path,sources,policy)
    first = review['items'][0]
    first['review']['reviewer_type']='AI'
    atomic_json(tmp_path/'source_review.json',review)
    with pytest.raises(ValueError,match='explicit AI_ASSISTED_PILOT'):
        publish_reviewed_labels(tmp_path,plan,sources,[],tmp_path/'labels')
    review.update(review_mode='AI_ASSISTED_PILOT',review_authorization='USER_DELEGATED_TO_CODEX_2026_10_06')
    for item in review['items']:
        item['review'].update(reviewer_type='AI',query_author_type='AI',independently_written=False)
    first['review']['independently_written']=True
    atomic_json(tmp_path/'source_review.json',review)
    with pytest.raises(ValueError,match='independent human authorship'):
        publish_reviewed_labels(tmp_path,plan,sources,[],tmp_path/'labels')
    first['review']['independently_written']=False
    first['review']['evidence_quote']='A fabricated quote not present in the source.'
    atomic_json(tmp_path/'source_review.json',review)
    with pytest.raises(ValueError,match='query/quote'):
        publish_reviewed_labels(tmp_path,plan,sources,[],tmp_path/'labels')
    for item in review['items']:
        item['review']['decision']='PENDING'
    atomic_json(tmp_path/'source_review.json',review)
    assert publish_reviewed_labels(tmp_path,plan,sources,[],tmp_path/'labels')['state']=='WAITING_FOR_SOURCE_QUERY_REVIEW'
    assert not (tmp_path/'labels/retrieval_train.json').exists()


@pytest.mark.parametrize("preparation_name",["stage-a-source-training-v2-heldout","stage-a-source-training-v2-heldout-new-corpus"])
def test_ai_review_upload_keeps_backup_sources_and_conflicting_team_edits(tmp_path,sources,policy,preparation_name):
    data = tmp_path/'data'
    review_dir = data/'labels/preparation'/preparation_name
    plan, original = reviewed_setup(review_dir,sources,policy)
    incoming = deepcopy(original)
    incoming.update(review_mode='AI_ASSISTED_PILOT',review_authorization='USER_DELEGATED_TO_CODEX_2026_10_06')
    for item in incoming['items']:
        item['review'].update(reviewer_type='AI',query_author_type='AI',independently_written=False)
    for item in original['items']:
        item['review'].update(decision='PENDING',reviewer='',independently_written=False)
    target = review_dir/'source_review.json'
    atomic_json(target,original)
    upload = tmp_path/'source_review_ai_pilot.json'
    atomic_json(upload,incoming)
    result = install_ai_pilot_review(upload,data)
    assert read_json(result['backup_path'])==original and read_json(target)==incoming
    assert result['accepted']=={'train':4,'dev':2,'heldout':0}
    assert install_ai_pilot_review(upload,data)['accepted']==result['accepted']
    changed = deepcopy(incoming)
    changed['items'][0]['review']['notes']='team already edited this decision'
    atomic_json(target,changed)
    with pytest.raises(ValueError,match='conflict'):
        install_ai_pilot_review(upload,data)
    assert read_json(target)==changed
    atomic_json(target,original)
    incoming['items'][0]['text']='tampered source'
    atomic_json(upload,incoming)
    with pytest.raises(ValueError,match='immutable'):
        install_ai_pilot_review(upload,data)
    assert read_json(target)==original


def test_review_only_upgrade_reuses_complete_sealed_drafts_but_checks_every_source_contract(tmp_path,sources,policy):
    plan,review = reviewed_setup(tmp_path,sources,policy)
    review.update(review_mode='AI_ASSISTED_PILOT',review_authorization='USER_DELEGATED_TO_CODEX_2026_10_06')
    atomic_json(tmp_path/'source_review.json',review)
    proposed = {k:v for k,v in plan.items() if k!='sha256'}
    proposed['code_sha256']='different-review-code-version'
    proposed['sha256']=digest_json(proposed)
    assert reuse_completed_review_plan(tmp_path,proposed)==plan
    spec={'model_id':'CPU-TEACHER-DOUBLE','revision':'fixture','max_new_tokens':384}
    assert generate_training_drafts(tmp_path,reuse_completed_review_plan(tmp_path,proposed),spec,[],None)['written_this_call']==0
    changed = deepcopy(proposed)
    changed['prompt_sha256']='changed-teacher-prompt'
    with pytest.raises(ValueError,match='source/prompt/policy changed'):
        reuse_completed_review_plan(tmp_path,changed)
    sample=next(s for s in plan['samples'] if s['split']=='train')
    (tmp_path/'drafts'/(sample['item_id']+'.json')).unlink()
    with pytest.raises(ValueError,match='drafts to be complete'):
        reuse_completed_review_plan(tmp_path,proposed)


def test_three_source_folds_keep_heldout_out_of_teacher_and_training_labels(tmp_path,sources,policy):
    policy={**policy,'heldout_samples':2,'heldout_fraction':.25,'min_heldout_reviewed':1}
    plan,review=reviewed_setup(tmp_path,sources,policy)
    groups={s:{r['source_group'] for r in plan['samples'] if r['split']==s} for s in ('train','dev','heldout')}
    assert not (groups['train']&groups['dev'] or groups['train']&groups['heldout'] or groups['dev']&groups['heldout'])
    assert len(list((tmp_path/'drafts').iterdir()))==4
    assert all(i['draft'] is None for i in review['items'] if i['split']=='heldout')
    result=publish_reviewed_labels(tmp_path,plan,sources,[],tmp_path/'labels')
    heldout=read_json(tmp_path/'labels/retrieval_heldout.json')
    assert result['accepted']=={'train':4,'dev':2,'heldout':2}
    assert all(q['provenance']['label_kind']=='INDEPENDENT_HUMAN_HELD_OUT' for q in heldout['queries'])
    from vietmedbridge.heldout import validate_heldout_split
    train,dev=read_json(result['train_path']),read_json(result['dev_path'])
    heldout['queries'][0]['id']=train['queries'][0]['id']
    with pytest.raises(ValueError,match='overlap'):
        validate_heldout_split(heldout,train,dev,[],sources)


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
    assert first['model_parameter_budget']['total_parameters']==20_346_066_432
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
    # Keep nested finalist/adapter namespaces below Windows MAX_PATH in CPU tests.
    tmp_path=tmp_path.parent/'tw'
    tmp_path.mkdir()
    from test_retrieval import Encoder
    from test_retrieval_cascade import CascadeReranker, Translator
    from test_full_plan import EncodeTokenizer
    from vietmedbridge import dataset, training_workflow as workflow, qwen_models, retrieval_models, translation_model
    from vietmedbridge.qwen_models import EMBEDDING_DIMENSION, QUERY_INSTRUCTION, RoleEncoder, pair_windows
    from vietmedbridge.reranker_training import write_adapter_manifest
    import transformers

    project=Path(__file__).resolve().parents[1]
    checkout,data=tmp_path/'checkout',tmp_path/'data'
    config=read_json(project/'configs/retrieval_full.json')
    config['lexical_segmentation']=False
    config['expansion_enabled']=False
    config['retrieval'].update(doc_candidate_k=8,sparse_top_k=8,detail_doc_k=8,doc_top_k=8)
    sources.candidate['golden']={'tokenizer':config['dense']}
    contest=[{'id':900000+i,'query':f'Official CPU fixture question {i}'} for i in range(1200)]
    atomic_json(checkout/'configs/retrieval_full.json',config)
    atomic_json(checkout/'configs/strong_model_manifest.json',read_json(project/'configs/strong_model_manifest.json'))
    atomic_json(data/'raw/snapshot.json',{'files':{'links_corpus.parquet':{'sha256':'CPU-official-fixture'}}})
    policy={**policy,'heldout_samples':2,'heldout_fraction':.25,'min_heldout_reviewed':1}
    plan,_=reviewed_setup(data/'preparation',sources,policy)
    publish_reviewed_labels(data/'preparation',plan,sources,[],data/'labels')
    monkeypatch.setattr(dataset,'parquet_path',lambda *a:Path('CPU-unused'))
    monkeypatch.setattr(workflow,'load_queries',lambda *a,**k:contest)
    monkeypatch.setattr(workflow,'load_catalog',lambda *a,**k:sources)
    monkeypatch.setattr(workflow,'inference_batches',lambda: {'embedding':2,'reranker':2})
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
            self.oom_backoffs=0
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
        def __init__(self,spec,adapter_path=None):
            super().__init__()
            self.spec,self.tokenizer=spec,EncodeTokenizer()
            self.oom_backoffs=0
            self.identity={**spec,'fine_tuned':False,'test_double':True}
            if adapter_path is not None:
                self.identity.update(fine_tuned=True,adapter_manifest_sha256=read_json(Path(adapter_path)/'adapter_manifest.json')['manifest_sha256'])
        def windows(self,query,text,overlap):
            return pair_windows(self.tokenizer,query,text,self.spec['max_length'],overlap)
        def score(self,pairs,**kwargs):
            calls['ranking']+=1
            return super().score(pairs,**kwargs)
        def close(self):
            pass

    class TrainingReached(Exception):
        pass
    stop_before_training=True
    def train(reranker,train_bundle,dev_bundle,output_dir,**kwargs):
        calls['training']+=1
        assert not train_bundle['holds'] and not dev_bundle['holds']
        assert all(len(g['pairs'])==8 for b in (train_bundle,dev_bundle) for g in b['groups'])
        assert kwargs['model_budget']['total_parameters']==20_346_066_432
        assert not {g['query_id'] for b in (train_bundle,dev_bundle) for g in b['groups']} & {q['id'] for q in read_json(data/'labels/retrieval_heldout.json')['queries']}
        assert Path(output_dir).name==digest_json([read_json(data/'labels/retrieval_train.json'),read_json(data/'labels/retrieval_dev.json')])[:10]
        if stop_before_training:
            raise TrainingReached('CPU test stops before real GPU QLoRA')
        adapter=Path(output_dir)/'best-dev-adapter';adapter.mkdir(parents=True,exist_ok=True)
        header=json.dumps({'lora.weight':{'shape':[2,7],'dtype':'F32','data_offsets':[0,56]}}).encode()
        (adapter/'adapter_model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+bytes(56))
        atomic_json(adapter/'adapter_config.json',{'test_double':True})
        write_adapter_manifest(adapter,labels=train_bundle,training_contract={'test_double':True})
        return {'adapter':str(adapter),'test_double':True}
    monkeypatch.setattr(workflow,'TorchDenseEncoder',Dense)
    monkeypatch.setattr(workflow,'TorchQwenEncoder',Secondary)
    monkeypatch.setattr(workflow,'TorchQueryTranslator',Translation)
    monkeypatch.setattr(workflow,'TorchQwenReranker',Ranking)
    monkeypatch.setattr(workflow,'train_qlora',train)

    first=workflow.run_training_workflow(data,checkout,work_dir=tmp_path/'work')
    assert first['state']=='MINING_REQUIRES_MORE_GOLD_COVERAGE_OR_REVIEWED_NEGATIVES'
    assert calls['training']==0 and len(first['train_holds'])==4 and len(first['dev_holds'])==2
    before=dict(calls)
    assert calls['ranking']==0
    pending=workflow.run_training_workflow(data,checkout,work_dir=tmp_path/'work')
    assert pending['gpu_models_loaded_this_call']==0 and pending['mining_reused'] is True
    assert calls==before
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
    assert len(list((data/'training/stage-a-qlora-v3-per-model-15b/experiments').iterdir()))==1
    stop_before_training=False
    pilot=workflow.run_training_workflow(data,checkout,work_dir=tmp_path/'work')
    assert pilot['research_ablations']=='DEFERRED_UNTIL_FIRST_MEASURED_RESULT'
    assert not (data/'training/stage-a-qlora-v3-per-model-15b/ablation_summary.json').exists()
    completed=workflow.run_training_workflow(data,checkout,work_dir=tmp_path/'work',run_ablations=True)
    assert completed['state']=='TRAINED_DEV_SELECTED_HELD_OUT_EVALUATED_LOCAL_PROXY'
    assert completed['model_parameter_budget']['adapter_parameters']==14
    assert completed['heldout_evaluation']['used_for_checkpoint_or_cutoff_selection'] is False
    run=data/'training/stage-a-qlora-v3-per-model-15b'
    assert len(read_json(run/'ablation_summary.json')['trials'])==8
    assert (run/'selected_adapter/adapter_manifest.json').is_file()
    heldout_ids={q['id'] for q in read_json(data/'labels/retrieval_heldout.json')['queries']}
    assert heldout_ids=={q['id'] for q in completed['heldout_evaluation']['evaluation']['per_query']}
    for path in run.glob('experiments/*/mining_queries/query-*.done.json'):
        record=read_json(path)
        if 'prediction' in record:
            assert record['prediction']['id'] not in heldout_ids
    from vietmedbridge.full_plan_runtime import reviewed_calibration,calibration_context
    from vietmedbridge.medical_lexical import MedicalAnalyzer
    from vietmedbridge.source_parents import derive_parents
    selected,adapter=reviewed_calibration(run,derive_parents(sources,Tokenizer()),contest,config['reranker'],
        context=calibration_context(config,MedicalAnalyzer(segmentation=False)))
    assert selected['manifest_sha256']==read_json(run/'calibrated_policy.json')['manifest_sha256']
    assert adapter==run/'selected_adapter'
    before_complete=dict(calls)
    repeated=workflow.run_training_workflow(data,checkout,work_dir=tmp_path/'work',run_ablations=True)
    assert repeated==completed
    assert {k:v for k,v in calls.items() if k!='training'}=={k:v for k,v in before_complete.items() if k!='training'}
    # Notebook 04 consumes the actual selected manifest/policy and emits all fixture IDs.
    from vietmedbridge import full_plan_runtime as runtime
    from vietmedbridge.competition_pilot import MASTER_PLAN
    (checkout/MASTER_PLAN).write_text('CPU wiring fixture, not a scored medical submission',encoding='utf-8')
    monkeypatch.setattr(runtime,'load_catalog',lambda *a,**k:sources)
    monkeypatch.setattr(runtime,'inference_batches',lambda: {'embedding':2,'reranker':2})
    monkeypatch.setattr(runtime,'load_queries',lambda *a,**k:contest)
    monkeypatch.setattr(runtime,'parquet_path',lambda *a:Path('CPU-unused'))
    monkeypatch.setattr(runtime,'TorchDenseEncoder',Dense)
    monkeypatch.setattr(runtime,'TorchQwenEncoder',Secondary)
    monkeypatch.setattr(runtime,'TorchQueryTranslator',Translation)
    monkeypatch.setattr(runtime,'TorchQwenReranker',Ranking)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,
        get_device_name=lambda *a:'CPU_DOUBLE_NOT_GPU')))
    submission=runtime.run_full_pipeline(data,checkout,code_commit='CPU-double',work_dir=tmp_path/'work')
    assert submission['ready']['query_count']==1200 and submission['ready']['official_score'] is None
    assert submission['status']['fine_tuning']=='DEV_SELECTED_HELD_OUT_EVALUATED_LOCAL_PROXY'
    assert submission['status']['model_parameter_budget']['adapter_parameters']==14
