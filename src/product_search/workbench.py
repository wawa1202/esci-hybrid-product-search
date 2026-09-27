"""桌面面板的资源和进程管理，复用 CLI，不复制搜索/训练算法。"""
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
from uuid import uuid4


def read_object(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('文件内容不是 JSON 对象')
    return data


def discover_resources(root):
    """只发现正式目录；真正加载时仍由业务模块执行完整哈希校验。"""
    root = Path(root)
    result = {'datasets': [], 'releases': [], 'reports': []}
    searches = [('datasets', 'data/processed', 'manifest.json'),
                ('releases', 'artifacts', 'release.json'),
                ('reports', 'reports', 'ranking-results.json')]
    for kind, parent, filename in searches:
        base = root / parent
        for directory in base.iterdir() if base.exists() else []:
            if not directory.is_dir() or directory.name.startswith('.'):
                continue
            path = directory / filename
            try:
                value = read_object(path)
                if kind == 'datasets' and (value.get('status') != 'prepared' or
                        value.get('identity', {}).get('project', {}).get('profile') != 'dev'):
                    continue
                if kind == 'releases' and (not value.get('release_id') or not value.get('files')
                        or not value.get('dataset_id', '').startswith('dev-')):
                    continue
                if kind == 'reports' and (value.get('partition') != 'validation' or 'ranking' not in value):
                    continue
                result[kind].append(path)
            except (OSError, ValueError, TypeError):
                continue
        result[kind].sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return result


def report_rows(report):
    """把原始报告变成人可读的表格；未定义值不假装是零。"""
    names = {'bm25': '关键词召回', 'dense': '语义召回', 'rrf': '两路融合', 'ltr': '学习排序'}
    rows = []
    def value(section, method, metric, percent=False):
        item = report.get(section, {}).get(method, {}).get(metric, {}).get('mean')
        if not isinstance(item, (float, int)) or not math.isfinite(item):
            return '—'
        return f'{item:.2%}' if percent else f'{item:.4f}'
    for method, title in names.items():
        rows.append((title, value('ranking', method, 'ndcg'), value('ranking', method, 'mrr_e'),
                     value('retrieval', method, 'known_e_recall', True),
                     value('retrieval', method, 'judged_fraction', True)))
    return rows


@dataclass(frozen=True)
class Experiment:
    root: Path
    dataset: Path
    name: str

    @classmethod
    def create(cls, root, dataset):
        root, dataset = Path(root).resolve(), Path(dataset).resolve()
        if not dataset.is_relative_to(root / 'data' / 'processed') or not (dataset / 'manifest.json').is_file():
            raise ValueError('请选择本项目 data/processed 中的正式 dev 数据集')
        return cls(root, dataset, datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:6])

    @property
    def index_dir(self):
        return self.root / 'artifacts' / ('panel-indexes-' + self.name)

    @property
    def model_dir(self):
        return self.root / 'artifacts' / ('panel-ltr-' + self.name)

    @property
    def report_dir(self):
        return self.root / 'reports' / ('panel-validation-' + self.name)

    def command(self, step):
        common = ['--project-root', str(self.root), '--config', 'configs/dev.toml']
        if step == 'build':
            if self.index_dir.exists():
                raise FileExistsError('本轮索引目录已存在，请开始新一轮实验')
            return ['build-indexes', '--dataset', str(self.dataset), '--output', str(self.index_dir), *common]
        if step == 'train':
            if not (self.index_dir / 'release.json').is_file():
                raise ValueError('先完成第 1 步：建立索引')
            if self.model_dir.exists():
                raise FileExistsError('本轮模型目录已存在，请开始新一轮实验')
            return ['train-ranker', '--dataset', str(self.dataset), '--release', str(self.index_dir),
                    '--output', str(self.model_dir), *common]
        if step == 'evaluate':
            if not (self.model_dir / 'release.json').is_file():
                raise ValueError('先完成第 2 步：训练排序模型')
            if self.report_dir.exists():
                raise FileExistsError('本轮评估目录已存在，请开始新一轮实验')
            return ['evaluate', '--dataset', str(self.dataset), '--release', str(self.model_dir),
                    '--output', str(self.report_dir), '--partition', 'validation', *common]
        raise ValueError('面板只提供 build/train/evaluate，不运行 test')


class JobRunner:
    """一个运行槽位，只管理自己创建的进程；输出经队列交给 Tk 主线程。"""
    def __init__(self, root, events: queue.Queue):
        self.root = Path(root)
        self.events = events
        self._lock = threading.Lock()
        self._process = None
        self._busy = False
        self.log_path = None

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def start(self, command, *, label):
        with self._lock:
            if self._busy:
                raise RuntimeError('当前任务还在运行，请等待或先停止')
            logdir = self.root / 'logs'
            logdir.mkdir(parents=True, exist_ok=True)
            self.log_path = logdir / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8] + '.log')
            env = os.environ.copy()
            env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
            self._process = subprocess.Popen(command, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            self._busy = True
            process = self._process
        self.events.put(('log', f'开始：{label}\n日志：{self.log_path}\n'))
        threading.Thread(target=self._collect, args=(process, self.log_path), daemon=True).start()

    def _collect(self, process, path):
        code = -1
        try:
            with path.open('w', encoding='utf-8') as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    self.events.put(('log', line))
            code = process.wait()
        except Exception as exc:
            self.events.put(('log', f'读取日志失败：{exc}\n'))
            if process.poll() is None:
                self.stop()
        finally:
            if process.stdout is not None:
                process.stdout.close()
            with self._lock:
                self._busy = False
            self.events.put(('done', code))

    def stop(self):
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            if os.name == 'nt':
                # venv 的 Windows Python 启动器可能有子进程；只停止此实例拥有的进程树。
                try:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=10, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
            else:
                process.terminate()
