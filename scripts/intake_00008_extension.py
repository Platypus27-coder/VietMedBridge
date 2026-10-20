"""Audit/import the later 00008 archive without changing the completed campaign."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tarfile

from vietmedbridge.artifacts import atomic_json, digest_json, read_json
from vietmedbridge.external_import import MEMBERS
from vietmedbridge.isolated_intake import _inside

SOURCE_00008 = {'drive_file_id':'1lqL46iCpfD7qnub-8ih1avKj6K3Dporv',
    'archive':'vibiomir_retain_shard_00008.tar','archive_bytes':2503065600,
    'inner_tar':None,'job_id':'new-00008','mode':'NEW_SNAPSHOT_SOURCE',
    'snapshot_id':'5586f423916005c11ffa8afb96895017d672af51d68aee8a88d28dd2e0dc39c6',
    'frontier':'data/crawl_shards/shard_00008.parquet'}


def _checked(value):
    if value.get('manifest_sha256')!=digest_json({k:v for k,v in value.items() if k!='manifest_sha256'}):
        raise ValueError('Extension audit/manifest digest changed.')
    return value


def completed_parent_mappings(parent,scope):
    """Read nine commit receipts and file sizes; payload SHA is deferred to use."""
    parent=Path(parent)
    report=read_json(parent/'full_intake_status.json')
    groups=[('.',scope['base_plan']['sources']),('additional-frontiers',scope['additional_sources']),
        ('retained-recovery',[scope['retained']])]
    expected={s['job_id'] for _,sources in groups for s in sources}
    if (report.get('state')!='FULL_INTAKE_COMPLETE' or report.get('completed_sources')!=len(expected)
        or report.get('total_sources')!=len(expected) or report.get('pending_sources')
        or report.get('scope_sha256')!=digest_json(scope)
        or {b['build_run'] for b in report.get('builds',[])}!=expected):
        raise ValueError('The nine-source parent campaign is incomplete or differs from the audited scope.')
    refs,counts,seen=[],{},set()
    for phase,sources in groups:
        root=_inside(parent,phase)
        plan=read_json(root/'intake_plan.json')
        pinned={s['job_id']:s for s in plan['sources']}
        if set(pinned)!={s['job_id'] for s in sources}:
            raise ValueError('Parent phase sources changed.')
        for source in sources:
            job=source['job_id'];target=_inside(root,'sources/'+job)
            saved=read_json(target/'complete.json')
            if (saved.get('state')!='COMPLETE' or saved.get('source')!=pinned[job]
                or saved.get('plan_sha256')!=digest_json(plan)
                or any(saved['source'].get(k)!=source[k] for k in ('drive_file_id','snapshot_id','job_id'))
                or saved.get('build_run')!=job or job in seen):
                raise ValueError('Parent source receipt changed; no automatic replay.')
            item=saved['files']['added_ids']
            path=_inside(target,item['path'])
            if (item['sha256']!=saved['added_ids_sha256'] or path.stat().st_size!=item['bytes']
                or not re.fullmatch('[0-9a-f]{64}',item['sha256'])):
                raise ValueError('Parent mapping receipt/file size changed.')
            refs.append((path,item['sha256']));seen.add(job)
            for name,value in saved['counts'].items():
                counts[name]=counts.get(name,0)+value
    if counts!=report['counts'] or sum(counts.values())!=scope['audited_coverage']['union_native_ids']:
        raise ValueError('Parent committed source counts do not match the campaign.')
    return refs,report


def audit_00008(root,load_reader,*,source=SOURCE_00008):
    """Seek TAR headers, read descriptor/frontier only; never copy the full TAR."""
    root=Path(root);marker=root/'source_audit.json'
    if marker.is_file():
        saved=_checked(read_json(marker))
        frontier=saved.get('metadata',{}).get('frontier',{}).get('sha256','')
        if (saved.get('request')!=source or not re.fullmatch('[0-9a-f]{64}',frontier)
            or saved.get('source')!=dict(source,frontier_sha256=frontier)):
            raise ValueError('Extension source changed; retain this campaign and create a new one.')
        return saved['source']
    reader=load_reader();remote=reader.metadata(source['drive_file_id'])
    if remote.get('name')!=source['archive'] or int(remote['size'])!=source['archive_bytes']:
        raise ValueError('00008 archive identity/size changed; re-audit before import.')
    wanted={MEMBERS['dataset']:'dataset',source['frontier']:'frontier'}
    found={}
    with reader.open(source['drive_file_id'],remote) as stream, tarfile.open(fileobj=stream,mode='r:') as archive:
        for index,member in enumerate(archive):
            if index>=128:
                raise ValueError('00008 required metadata missing from bounded TAR header scan.')
            kind=wanted.get(member.name)
            if kind is None:
                continue
            if kind in found or not member.isfile() or not 0<member.size<=512*2**20:
                raise ValueError('Invalid/duplicate 00008 metadata member.')
            if kind=='dataset' and member.size>2**20:
                raise ValueError('Dataset descriptor exceeds its bounded JSON size.')
            sha=hashlib.sha256();pieces=[];received=0
            with archive.extractfile(member) as incoming:
                for block in iter(lambda:incoming.read(2**20),b''):
                    sha.update(block)
                    received+=len(block)
                    if kind=='dataset':
                        pieces.append(block)
            checksum=sha.hexdigest()
            if received!=member.size:
                raise EOFError('Truncated 00008 metadata member.')
            expected=member.pax_headers.get('vibiomir.sha256')
            if expected is not None and checksum!=expected:
                raise ValueError('00008 member SHA differs from its TAR metadata.')
            found[kind]={'sha256':checksum,'bytes':member.size,'path':member.name}
            if kind=='dataset':
                descriptor=json.loads(b''.join(pieces))
                if descriptor.get('snapshot_id')!=source['snapshot_id']:
                    raise ValueError('00008 belongs to a different BTC snapshot.')
            if len(found)==2:
                break
    if set(found)!={'dataset','frontier'}:
        raise ValueError('00008 descriptor/frontier not found; do not guess its source layout.')
    if reader.metadata(source['drive_file_id'])!=remote:
        raise ValueError('00008 archive changed during audit.')
    bound=dict(source,frontier_sha256=found['frontier']['sha256'])
    saved={'request':source,'source':bound,'remote':remote,'metadata':found,
        'scope':'Descriptor/frontier identity only; row membership and all extraction bindings checked during intake'}
    saved['manifest_sha256']=digest_json(saved)
    atomic_json(marker,saved)
    print('00008 SOURCE_AUDITED: snapshot/frontier bound; full archive not downloaded',flush=True)
    return bound


def load_00008_metadata(reader,source,target,audit_root):
    from vietmedbridge.isolated_intake import stage_metadata
    audit=_checked(read_json(Path(audit_root)/'source_audit.json'))
    if source!=audit['source'] or reader.metadata(source['drive_file_id'])!=audit['remote']:
        raise ValueError('00008 archive version changed after audit; do not change the saved campaign.')
    return stage_metadata(reader,source,target)


def extension_00008_plan(parent,root,work,scope,load_reader,*,source=SOURCE_00008):
    parent,root,work=Path(parent).resolve(),Path(root).resolve(),Path(work).resolve()
    if (root==parent or root.is_relative_to(parent) or parent.is_relative_to(root)
        or work.is_relative_to(parent) or parent.is_relative_to(work)
        or work.is_relative_to(root) or root.is_relative_to(work)):
        raise ValueError('00008 must use a separate sibling campaign and runtime work directory.')
    refs,report=completed_parent_mappings(parent,scope)
    if (source['snapshot_id']!=scope['base_plan']['required_snapshot_id']
        or source['snapshot_id'] not in scope['base_plan']['snapshots']):
        raise ValueError('Extra source differs from the parent BTC snapshot.')
    source=audit_00008(root,load_reader,source=source)
    if source['frontier_sha256'] in {s['frontier_sha256'] for s in scope['base_plan']['sources']+scope['additional_sources']}:
        raise ValueError('00008 repeats an already-committed frontier.')
    plan=deepcopy(scope['base_plan'])
    plan.update(sources=[source],campaign='btc-new-71be0e12-00008-20261020-v1',
        parent_scope_sha256=digest_json(scope),prior_mapping_sha256=[checksum for _,checksum in refs])
    return plan,refs,report
