"""产物身份、哈希和发布校验；防止数据、索引、模型互相错配。"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from product_search.settings import Settings


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('JSON 顶层必须是对象')
    return value


def write_json(path: str | Path, value: object) -> None:
    """不覆盖历史实验；大产物应先在临时目录完成再整体发布。"""
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content + '\n')


def runtime_signature(settings: Settings) -> dict:
    """仅纳入影响召回/特征语义的配置，不因端口或批量大小改变身份。"""
    embedding = asdict(settings.embedding)
    for name in ('device', 'batch_size'):
        embedding.pop(name)
    return {
        'pipeline_version': 'search-v1',
        'tokenizer_version': 'casefold-before-filter-v2',
        'locale': settings.project.locale,
        'text': asdict(settings.text),
        'bm25': asdict(settings.bm25),
        'embedding': embedding,
        'retrieval': asdict(settings.retrieval),
    }


def verify_release(directory: str | Path, settings: Settings | None = None) -> dict:
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'release.json')
    for name in ('release_id', 'dataset_id', 'runtime_signature', 'files'):
        if name not in manifest:
            raise ValueError(f'release.json 缺少 {name}')
    if not isinstance(manifest['files'], dict) or not manifest['files']:
        raise ValueError('发布清单没有文件')
    for name, expected in manifest['files'].items():
        path = (directory / name).resolve()
        if path.parent != directory or name != path.name:
            raise ValueError('发布清单只能引用本目录内的文件')
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'产物 SHA256 校验失败：{name}')
    if settings is not None and manifest['runtime_signature'] != runtime_signature(settings):
        raise ValueError('当前配置与发布产物不兼容，请使用建索引时的配置')
    return manifest


def verify_dataset(directory: str | Path) -> dict:
    """数据的深层关系由 audit_store 校验；这里检查文件身份。"""
    from product_search.data.audit_store import audit_dataset
    audit_dataset(Path(directory))
    return read_json(Path(directory) / 'manifest.json')
