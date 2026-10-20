"""Resume an already-loaded submission adapter without cyclic vector-cache reads.

This small overlay changes transport order only. It does not replace producer
source, sampling positions, vector values, index policies, or existing caches.
"""
from pathlib import Path
import threading


def install_cache_first_io(io_type):
    """Consume resident vectors before downloads can evict unread residents."""
    if getattr(io_type.stage, '_vmb_cache_first', False):
        return
    original = io_type.stage

    def cache_first(self, items, label, *, consume=None, reserve=None, workers=4):
        items = list(items)
        if label != 'Vector cache' or len(items) < 2:
            return original(self,items,label,consume=consume,reserve=reserve,workers=workers)
        present, missing, local_bytes = [],[],0
        for number,item in enumerate(items):
            path = Path(item[1])
            if path.is_file():
                present.append((number,item))
                local_bytes += path.stat().st_size
            else:
                missing.append((number,item))
        ordered = present+missing
        self.progress(f'Vector cache plan: {len(present):,}/{len(items):,} local files'
            f' ({local_bytes/2**30:.2f} GiB); {len(missing):,} missing files'
            ' | consume local cache first; preserve original result positions')
        def consume_original(number,path):
            if consume is not None:
                consume(ordered[number][0],path)
        return original(self,[item for _,item in ordered],label,
            consume=consume_original,reserve=reserve,workers=workers)

    cache_first._vmb_cache_first = True
    io_type.stage = cache_first


def resume_cache_first(namespace):
    """Run in the same Colab VM after its current submission cell has returned."""
    if any(t.name=='submission-progress' and t.is_alive() for t in threading.enumerate()):
        raise RuntimeError('Dừng riêng cell xử lý trước; giữ nguyên runtime và cache Colab.')
    required=('VerifiedSubmissionIO','submission_gpu_progress','prepare_drive_receipt_lookup',
        'ensure_submission_mount','DRIVE_RECEIPTS','DATA_ROOT','CHECKOUT','WORK_DIR','PREPARATION','COMMIT')
    if any(name not in namespace for name in required) or namespace.get('DRIVE_RECEIPTS') is None:
        raise RuntimeError('Cần đúng runtime đã chạy bản tối ưu submission; không khởi tạo lại preparation.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Giữ runtime GPU của phiên submission; chưa tự đổi runtime.')
    install_cache_first_io(namespace['VerifiedSubmissionIO'])
    expected=namespace['PREPARATION']['candidate_manifest_sha256']
    root=namespace['ensure_submission_mount'](namespace['DATA_ROOT'],expected)
    namespace['DATA_ROOT']=root
    source=namespace['DRIVE_RECEIPTS']
    source.verify_candidate(expected)
    print('Cache-first resume: giữ mapping/vector hiện có; không chạy CPU preparation hoặc encode corpus.',flush=True)
    from vietmedbridge.submission_workflow import run_current_baseline
    with namespace['submission_gpu_progress'](root,namespace['WORK_DIR'],label='baseline',receipt_api=source) as progress:
        namespace['prepare_drive_receipt_lookup'](source,namespace['CHECKOUT'],namespace['WORK_DIR'],progress=progress.report)
        result=run_current_baseline(root,namespace['CHECKOUT'],namespace['WORK_DIR'],code_commit=namespace['COMMIT'])
    namespace['BASELINE']=result
    import json
    print(json.dumps(result['ready'],ensure_ascii=False,indent=2),flush=True)
    if result['ready']['state']=='READY_FOR_MANUAL_UPLOAD':
        print('ZIP đã kiểm tra trên Drive; xem retrieval/latest_submission.json.',flush=True)
    return result


if __name__=='__main__':
    resume_cache_first(globals())
