"""One Colab GPU notebook to measure frozen Qwen inference before Kaggle runs."""
import json
from pathlib import Path
from textwrap import dedent

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def notebook():
    def cell(source):
        return nb.v4.new_code_cell(dedent(source).strip())
    sources = {name:(ROOT/'scripts'/name).read_text(encoding='utf-8') for name in
               ('new_data_drive_api.py','qwen_token_batching.py','colab_qwen_benchmark.py')}
    setup = cell('''
        #@title 1. Colab GPU T4 — Drive API, thư viện frozen và runtime Kaggle
        import copy,hashlib,importlib,importlib.metadata,json,os,re,shutil,subprocess,sys,time
        from pathlib import Path
        if not Path('/var/colab/hostname').is_file() or not shutil.which('nvidia-smi'):
            raise RuntimeError('Chọn Colab GPU T4 ngay từ đầu rồi Run all; không chạy giai đoạn CPU trước.')
        subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],check=True)
        WORK=Path('/content/vmb_qwen_colab_benchmark')
        WORK.mkdir(parents=True,exist_ok=True)
        FORCE_REMEASURE=False #@param {type:"boolean"}
        sys.path.insert(0,str(WORK))
        from google.colab import auth
        auth.authenticate_user()
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        credentials,_=google.auth.default()
    ''')
    setup.source += '\nSOURCES=json.loads('+repr(json.dumps(sources,ensure_ascii=False))+')\n'
    setup.source += dedent(r'''
        for name,source in SOURCES.items():
            (WORK/name).write_text(source,encoding='utf-8')
            sys.modules.pop(Path(name).stem,None)
        importlib.invalidate_caches()
        from new_data_drive_api import DriveReceiptSource
        api=DriveReceiptSource(lambda:AuthorizedSession(copy.copy(credentials)),WORK,WORK,
            data_folder_id='1NJ9LtdOSRLZtCvBuML7O8UYppiJuIy4g')
        JOB_PATH=WORK/'job.json'
        def digest(value):
            return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        def read_library():
            value=json.loads(JOB_PATH.read_text(encoding='utf-8'))
            if (value.get('manifest_sha256')!='d6ab9a432ac64e5aadb39a826b7dc2176094cc66d4f4e3d9e17c121e071c0711'
                or digest({k:v for k,v in value.items() if k!='manifest_sha256'})!=value['manifest_sha256']):
                raise ValueError('Thư viện benchmark không đúng job đã freeze.')
            return value
        if not JOB_PATH.is_file():
            print('BENCHMARK_SETUP: read one saved library JSON; no corpus/vector inventory',flush=True)
            metadata=api.relative('input_library.json')
            if metadata is None: raise FileNotFoundError('Tài khoản Google chưa đọc được thư viện trong VietMedBridge.')
            api.download(metadata,WORK/'job.partial',max_bytes=32*1024*1024)
            (WORK/'job.partial').replace(JOB_PATH)
        JOB=read_library()
        if (JOB.get('repo_url')!='https://github.com/Platypus27-coder/VietMedBridge.git'
            or not re.fullmatch('[0-9a-f]{40}',JOB.get('code_commit',''))):
            raise ValueError('Producer repo/commit không hợp lệ.')
        CHECKOUT=Path('/content/VietMedBridge-qwen-benchmark')
        if not (CHECKOUT/'.git').is_dir():
            subprocess.run(['git','clone','--no-checkout',JOB['repo_url'],str(CHECKOUT)],check=True)
        if subprocess.check_output(['git','-C',str(CHECKOUT),'remote','get-url','origin'],text=True).strip()!=JOB['repo_url']:
            raise ValueError('Checkout thuộc repo khác.')
        subprocess.run(['git','-C',str(CHECKOUT),'fetch','--depth','1','origin',JOB['code_commit']],check=True)
        subprocess.run(['git','-C',str(CHECKOUT),'checkout','--detach','FETCH_HEAD'],check=True)
        runtime=JOB['runtime']
        if (set(runtime)!={'torch','transformers','bitsandbytes'}
            or any(not re.fullmatch('[0-9][A-Za-z0-9.+_-]*',v) for v in runtime.values())):
            raise ValueError('Runtime lock không hợp lệ.')
        try: installed=importlib.metadata.version('torch')
        except importlib.metadata.PackageNotFoundError: installed=None
        if installed!=runtime['torch']:
            match=re.search(r'\+(cu[0-9]+)$',runtime['torch'])
            if match is None: raise ValueError('Cần bản torch CUDA gốc của Kaggle.')
            print('BENCHMARK_SETUP: install original CUDA torch',runtime['torch'],flush=True)
            subprocess.run([sys.executable,'-m','pip','install','torch=='+runtime['torch'],
                '--index-url','https://download.pytorch.org/whl/'+match[1]],check=True)
            subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchvision','torchaudio'],check=True)
        print('BENCHMARK_SETUP: install model-only dependencies; no crawl/index packages',flush=True)
        subprocess.run([sys.executable,'-m','pip','install','numpy','tqdm','pyarrow==21.0.0',
            'transformers=='+runtime['transformers'],'bitsandbytes=='+runtime['bitsandbytes'],
            'tokenizers==0.22.2','huggingface-hub==0.36.0','accelerate==1.12.0',
            'google-api-python-client'],check=True)
        env=os.environ.copy()
        env['PYTHONPATH']=os.pathsep.join((str(WORK),str(CHECKOUT/'src')))
        env['HF_HOME']=str(WORK/'hf_cache')
        env['TOKENIZERS_PARALLELISM']='false'
        env['CUDA_VISIBLE_DEVICES']='0'
        probe="import torch; from transformers import AutoModel; print('BENCHMARK_CUDA:',torch.__version__,torch.cuda.is_available(),flush=True); assert torch.cuda.is_available()"
        checked_probe=subprocess.run([sys.executable,'-u','-c',probe],env=env,text=True,capture_output=True)
        if checked_probe.returncode and any(x in checked_probe.stderr.lower() for x in ('torchvision','torchaudio','torchvision::nms')):
            subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchvision','torchaudio'],check=True)
            checked_probe=subprocess.run([sys.executable,'-u','-c',probe],env=env,text=True,capture_output=True)
        print(checked_probe.stdout,flush=True)
        if checked_probe.returncode:
            print(checked_probe.stderr,flush=True)
            raise RuntimeError('CUDA/runtime chưa sẵn sàng; chưa nạp model hoặc chạy benchmark.')
        sys.path.insert(0,str(CHECKOUT/'src'))
    ''')
    sample = cell('''
        #@title 2. Lấy mẫu nhỏ từ tối đa ba file text trên Drive
        importlib.invalidate_caches()
        from colab_qwen_benchmark import prepare_sample,validate_job
        validate_job(JOB)
        SAMPLE_PATH,SAMPLE=prepare_sample(api,JOB,WORK/'sample')
        print('BENCHMARK_SAMPLE_READY: calibration',len(SAMPLE['calibration']),
              '| held-out evaluation',len(SAMPLE['evaluation']),'| assets',SAMPLE['assets_read'],flush=True)
    ''')
    run = cell('''
        #@title 3. Đo một GPU: native batch 16 so với bộ tự chọn Qwen v2
        REPORT_PATH=WORK/'benchmark.json'
        command=[sys.executable,'-u',str(WORK/'colab_qwen_benchmark.py'),
            '--job',str(JOB_PATH),'--sample',str(SAMPLE_PATH),'--checkout',str(CHECKOUT),'--output',str(REPORT_PATH)]
        if FORCE_REMEASURE: command.append('--force')
        started=time.monotonic()
        process=subprocess.Popen(command,env=env)
        try:
            while process.poll() is None:
                if time.monotonic()-started>2400:
                    raise RuntimeError('Benchmark vượt 40 phút kể cả tải model; giữ runtime để kiểm tra cache.')
                print('COLAB_BENCHMARK_HEARTBEAT:',int(time.monotonic()-started),'s',flush=True)
                time.sleep(15)
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait(timeout=10)
        if not REPORT_PATH.is_file():
            raise RuntimeError('Chưa có báo cáo benchmark. Xem traceback; không có task/corpus nào bị thay đổi.')
        result=json.loads(REPORT_PATH.read_text(encoding='utf-8'))
        print('BENCHMARK_STATE:',result['state'],'| tasks committed',result['tasks_committed'],flush=True)
        if process.returncode and result['state']!='BENCHMARK_FAILED':
            raise RuntimeError('Tiến trình lỗi; không dùng báo cáo cũ để kết luận lượt này.')
    ''')
    save = cell('''
        #@title 4. Lưu báo cáo nhỏ lên Drive — có thể chạy lại riêng cell này
        from colab_qwen_benchmark import publish_report
        DRIVE_REPORT=publish_report(api,REPORT_PATH)
        result=json.loads(REPORT_PATH.read_text(encoding='utf-8'))
        if result['state']=='BENCHMARK_COMPLETE_NO_TASKS_COMMITTED':
            e=result['evaluation']
            print('BENCHMARK_SUMMARY: native',round(e['native_texts_per_second'],2),'texts/s | candidate',
                round(e['candidate_texts_per_second'],2),'texts/s | held-out speedup',round(e['held_out_speedup'],2),
                '| accepted',e['candidate_accepted'],'| T4 match',result['kaggle_t4_hardware_match'],flush=True)
        else:
            print('BENCHMARK_FAILED: báo cáo lỗi đã lưu Drive; chưa kết luận tốc độ.',flush=True)
    ''')
    value = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(dedent('''
        # 04 — Thử hiệu năng Qwen 8B trên Colab trước lượt Kaggle qua đêm

        Chọn **Colab GPU T4 ngay từ đầu → Run all**. Dùng tài khoản Google
        đọc được folder VietMedBridge và ghi được data_staging. Không cần
        KAGGLE_JSON, Add Input hay notebook 02/03. Giữ phiên Kaggle đang chạy.

        Notebook dùng **Google Drive API**, không mount Drive để đọc hàng nghìn
        file. Chỉ đọc thư viện JSON đã freeze và tối đa **ba file missing-text**,
        lấy tối đa 288 text rồi tách mẫu chọn batch và mẫu đo độc lập theo ID.
        Qwen 8B/NF4, tokenizer, max_length, pooling và runtime khóa như Kaggle.

        Sau tải model và warm-up, đo hai lượt native batch 16 và hai lượt bộ
        tự chọn v2, đảo thứ tự đo, đồng bộ CUDA và kiểm sai số trên mẫu độc lập.
        Báo cáo có texts/s, tokens/s, VRAM, OOM khi tuning và cấu hình được chọn.
        [Cách đo CUDA theo PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html).

        Model/file tạm dùng ổ đĩa runtime Colab; báo cáo nhỏ được lưu qua API
        vào **VietMedBridge/data_staging/qwen_performance_benchmarks**. Không
        tải model/corpus về máy tính cá nhân, không ghi vector vào model_cache,
        không sửa task journal, candidate, kế hoạch hoặc coverage.

        Nếu Colab cấp GPU khác T4, báo cáo ghi T4 match=False; không suy ra tốc
        độ Kaggle T4 từ GPU đó. Đây là phép đo trên mẫu, chưa phải chứng nhận
        chất lượng hoặc tốc độ toàn corpus. Chưa tự thay cấu hình Kaggle.

        Sau khi có BENCHMARK_SAVED_ON_DRIVE, báo cáo đã lưu để đọc và chốt lượt
        tiếp theo. Chạy lại cell 4 chỉ lưu báo cáo, không chạy lại model. Cùng
        runtime có thể dùng lại kết quả đo; bật FORCE_REMEASURE để chủ động đo mới.
    ''').strip()),setup,sample,run,save])
    value.metadata={'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                    'language_info':{'name':'python'},'colab':{'provenance':[],'gpuType':'T4'}}
    return value


if __name__ == '__main__':
    nb.write(notebook(),ROOT/'notebooks/04_colab_benchmark_qwen_t4.ipynb')
