"""Build the Colab runner notebook (queue-polling execution loop).

Emits colab_runner.ipynb into the Google Drive queue folder. The notebook:
mounts Drive, installs deps, unpacks div-prune-src.zip from Drive (git clone
as fallback), then polls queue/*.json and executes jobs one by one, moving
finished jobs to processed/ and mirroring results back to Drive.
"""

import json
from pathlib import Path

DRIVE_ROOT = Path(r"G:/我的云端硬盘/colab-divprune")

MD_INTRO = """# div-prune Colab runner

任务队列执行器。本地把任务 JSON 放进 `colab-divprune/queue/`,这里逐个执行,
结果写回 `colab-divprune/results/`。

**使用方法**
1. 菜单 代码执行程序 - 更改运行时类型,选 **T4 GPU**
2. **只需一次**: 点左侧文件面板的「挂载云端硬盘」图标并同意授权。以后重开本笔记本 Drive 会自动挂好,不再弹授权
3. 点击 运行全部,保持浏览器标签页打开即可。若第 2 步没做,本单元会退回手动授权并给出提示
4. **队列空闲 20 分钟后 runner 会自动停止**,不再无限空转。之后追加了新任务,重开本笔记本运行全部即可
5. 想在任务跑完后立刻叫停: 在 `colab-divprune/` 里建一个空的 `STOP` 文件,下次启动会自动清理
6. 状态看 `status.json`,`current=stopped` 表示正常跑完收工。任务进度看 `processed/` 文件夹
"""

CODE_GPU = """import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if not torch.cuda.is_available():
    print('警告: 未检测到 GPU!菜单 代码执行程序-更改运行时类型 选 T4 GPU 后重跑本单元')
"""

CODE_MOUNT = """# 优先复用 Colab 的 notebook 级自动挂载(侧边栏点过一次「挂载云端硬盘」后,
# 重开笔记本会自动挂好),已挂载则跳过 OAuth,不再需要每次手动点同意。
from pathlib import Path
from google.colab import drive

MNT = Path('/content/drive')
if not (MNT / 'MyDrive').exists():
    print('Drive 尚未挂载,执行 drive.mount 手动授权(仅首次需要)')
    drive.mount('/content/drive')
else:
    print('Drive 已由 Colab 自动挂载,跳过授权')

ROOT = MNT / 'MyDrive' / 'colab-divprune'
for sub in ('queue', 'processed', 'results', 'work'):
    (ROOT / sub).mkdir(parents=True, exist_ok=True)
print('queue root:', ROOT)
print('queue jobs pending:', len(list((ROOT / 'queue').glob('*.json'))))
"""

CODE_DEPS = """!pip install -q torch-geometric hydra-core omegaconf tqdm scipy pandas sentencepiece sacrebleu
import torch_geometric, hydra, omegaconf, sentencepiece
print('torch_geometric', torch_geometric.__version__)
"""

CODE_SRC = """# 代码来源: Drive 里的 src zip(本地打包上传),无则退回 git clone
import shutil, subprocess, os
from pathlib import Path

ROOT = Path('/content/drive/MyDrive/colab-divprune')
REPO = Path('/content/div-prune')
SRC_ZIP = ROOT / 'work' / 'div-prune-src.zip'

if SRC_ZIP.exists():
    shutil.rmtree(REPO, ignore_errors=True)
    REPO.mkdir(parents=True)
    subprocess.run(['unzip', '-q', str(SRC_ZIP), '-d', str(REPO)], check=True)
    print('code unpacked from Drive zip')
else:
    if not REPO.exists():
        subprocess.run(['git', 'clone', '-q', 'https://github.com/HuKejie/div-prune', str(REPO)], check=True)
    print('code from git clone')

subprocess.run(['pip', 'install', '-q', '-e', str(REPO), '--no-deps'], check=True)
print(open(REPO / 'src' / 'divprune' / 'data_module' / 'dataset' / 'amazon.py').readline())
"""

CODE_RUNNER = """# ============ 任务队列主循环 ============
# 逐个执行 queue/*.json,完成后移到 processed/,结果与日志写回 results/
import json, os, shutil, subprocess, time
from pathlib import Path

ROOT = Path('/content/drive/MyDrive/colab-divprune')
QUEUE, PROCESSED = ROOT / 'queue', ROOT / 'processed'
RESULTS, WORK = ROOT / 'results', ROOT / 'work'
REPO = Path('/content/div-prune')
MIRROR = WORK / 'outputs_mirror'
SESSION = time.strftime('%Y%m%d-%H%M%S')
IDLE_SLEEP = 30
RUN_TIMEOUT_DEFAULT = 3600
IDLE_GRACE_S = 20 * 60       # 队列持续空闲超过 20 分钟,runner 自动停止
STOP_FILE = ROOT / 'STOP'    # 在 Drive 文件夹创建空文件可立即叫停(下次启动自动清理)


def heartbeat(current, note=''):
    (ROOT / 'status.json').write_text(json.dumps({
        'session': SESSION, 'current': current, 'note': note,
        'ts': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False))


def mirror_restore():
    \"\"\"断点续跑: 把 Drive 镜像的 outputs 拷回仓库,继续追加同一个 results.csv\"\"\"
    if not MIRROR.exists():
        return
    for out_dir in MIRROR.iterdir():
        if not out_dir.is_dir():
            continue
        target = REPO / out_dir.name
        shutil.copytree(out_dir, target, dirs_exist_ok=True)
    print('restored outputs mirror:', [p.name for p in MIRROR.iterdir()])


def mirror_sync(output_dirs):
    \"\"\"任务结束后把 outputs 目录镜像回 Drive\"\"\"
    for d in output_dirs:
        src = REPO / d
        if src.exists():
            dst = MIRROR / d
            for f in src.rglob('*'):
                if f.is_file():
                    rel = f.relative_to(src)
                    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst / rel)


def run_job(job):
    jid = job['job_id']
    out_dir = RESULTS / jid
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd, cwd = job['cmd'], job.get('cwd', str(REPO))
    timeout = int(job.get('timeout_s', RUN_TIMEOUT_DEFAULT))
    print(f'>> [{jid}] {cmd}', flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines, killed = [], False
    try:
        while True:
            line = proc.stdout.readline()
            if line:
                lines.append(line)
                print('   ' + line.rstrip()[:160], flush=True)
            if proc.poll() is not None:
                break
            if time.time() - t0 > timeout:
                proc.kill()
                killed = True
                break
            time.sleep(0.2)
    finally:
        tail = ''.join(lines[-400:])
    rc = proc.returncode if not killed else -9
    elapsed = time.time() - t0
    (out_dir / 'log.txt').write_text(tail, encoding='utf-8')
    status = 'done' if rc == 0 else 'timeout' if killed else 'failed'
    (out_dir / 'job.json').write_text(json.dumps(job, ensure_ascii=False, indent=2))
    # 收集结果文件
    copied = []
    for pat in job.get('results_patterns', []):
        p = REPO / pat
        if p.exists():
            shutil.copy2(p, out_dir / p.name)
            copied.append(p.name)
    summary = {'job_id': jid, 'status': status, 'rc': rc, 'elapsed_s': round(elapsed),
               'results': copied, 'session': SESSION}
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    mirror_sync(job.get('output_dirs', []))
    print(f'<< [{jid}] {status} in {elapsed:.0f}s, results: {copied}', flush=True)
    return status


mirror_restore()
STOP_FILE.unlink(missing_ok=True)   # 每次启动从干净状态开始
n_done = len(list(PROCESSED.glob('*.json')))
print(f'=== runner {SESSION} 就绪,已完成 {n_done} 个任务 ===', flush=True)
print(f'=== 队列空闲超过 {IDLE_GRACE_S} 秒将自动停止 ===', flush=True)
idle_since = None
while True:
    if STOP_FILE.exists():
        heartbeat('stopped', 'STOP file found, manual stop')
        print('=== 检测到 STOP 文件,runner 停止 ===', flush=True)
        break
    jobs = sorted(QUEUE.glob('*.json'), key=lambda p: p.name)
    if jobs:
        idle_since = None
        for jq in jobs:
            try:
                job = json.loads(jq.read_text(encoding='utf-8'))
            except Exception as e:
                jq.rename(PROCESSED / (jq.stem + '.invalid.json'))
                print(f'!! invalid job {jq.name}: {e}', flush=True)
                continue
            heartbeat(job['job_id'], jq.name)
            status = run_job(job)
            jq.rename(PROCESSED / f'{jq.stem}.{status}.json')
        continue   # 一轮跑完立刻再扫,期间新入队的任务也能被接住
    # 队列为空: 进入有限宽限期,不再无限空转
    if idle_since is None:
        idle_since = time.time()
        print(f'队列已空,进入 {IDLE_GRACE_S} 秒宽限期,期间新任务仍会自动执行', flush=True)
    waited = time.time() - idle_since
    heartbeat('idle', f'queue empty, auto-stop in {int(IDLE_GRACE_S - waited)}s, '
                      f'{len(list(PROCESSED.glob("*.json")))} processed')
    if waited >= IDLE_GRACE_S:
        heartbeat('stopped', f'queue empty for {int(waited)}s, graceful stop, '
                             f'{len(list(PROCESSED.glob("*.json")))} processed')
        print('=== 队列持续为空,runner 自动停止 ===', flush=True)
        print('=== 之后追加新任务时,重开本笔记本运行全部即可 ===', flush=True)
        break
    time.sleep(IDLE_SLEEP)
"""

CODE_STATUS = """# 快速状态面板(空闲时手动运行本单元)
import json
from pathlib import Path
ROOT = Path('/content/drive/MyDrive/colab-divprune')
if (ROOT / 'status.json').exists():
    print('status:', json.loads((ROOT / 'status.json').read_text()))
print('pending:', sorted(p.name for p in (ROOT/'queue').glob('*.json')))
print('processed:', len(list((ROOT/'processed').glob('*.json'))), 'jobs')
import collections
c = collections.Counter(p.suffix.split('.')[1] if '.' in p.name else '?' for p in (ROOT/'processed').glob('*.json'))
print('by status:', dict(c))
"""


def make_cell(cell_type, source):
    src = source.strip('\n')
    lines = src.split('\n')
    src_list = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": cell_type, "metadata": {}, "source": src_list}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": [
        make_cell("markdown", MD_INTRO),
        make_cell("code", CODE_GPU),
        make_cell("code", CODE_MOUNT),
        make_cell("code", CODE_DEPS),
        make_cell("code", CODE_SRC),
        make_cell("code", CODE_RUNNER),
        make_cell("code", CODE_STATUS),
    ],
}

out = DRIVE_ROOT / "colab_runner.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("written:", out)
