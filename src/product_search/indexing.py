"""离线构建发布包：商品快照、BM25、向量及可追溯清单。"""

import json
import shutil
import uuid
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import numpy as np

from product_search.artifacts import fingerprint, runtime_signature, sha256, verify_dataset, verify_release, write_json
from product_search.data.store import iter_products, open_database
from product_search.retrieval.bm25 import BM25Index
from product_search.retrieval.embedder import Embedder
from product_search.retrieval.vector_index import VectorIndex


def build_indexes(settings, dataset_dir, output_dir, *, embedder=None) -> Path:
    """先构建隐藏 staging；所有校验通过后才发布，失败不冒充成功。"""
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'发布目录已存在：{output_dir}')
    manifest = verify_dataset(dataset_dir)
    if manifest['identity']['project'] != asdict(settings.project):
        raise ValueError('配置与数据集的 profile、locale 或 seed 不符')
    selection = {key: getattr(settings.data, key) for key in manifest['identity']['selection']}
    if selection != manifest['identity']['selection']:
        raise ValueError('配置与数据集的抽样规则不符')
    embedder = embedder or Embedder(settings)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f'.{output_dir.name}.pending-{uuid.uuid4().hex}'
    stage.mkdir()
    try:
        shutil.copy2(dataset_dir / 'catalog.sqlite', stage / 'catalog.sqlite')
        shutil.copy2(dataset_dir / 'manifest.json', stage / 'dataset_manifest.json')
        with open_database(stage / 'catalog.sqlite', read_only=True) as catalog:
            print('构建 BM25 索引…', flush=True)
            with BM25Index.build(iter_products(catalog), stage / 'bm25.sqlite',
                                 text_version=settings.text.version, max_tokens=settings.text.max_tokens,
                                 k1=settings.bm25.k1, b=settings.bm25.b):
                pass
            count = catalog.execute('SELECT COUNT(*) FROM products').fetchone()[0]
            vectors = np.lib.format.open_memmap(stage / 'embeddings.npy', mode='w+', dtype=np.float32,
                                                shape=(count, embedder.dimension))
            try:
                products = iter_products(catalog)
                offset = 0
                with (stage / 'embedding_ids.jsonl').open('x', encoding='utf-8') as stream:
                    while batch := list(islice(products, settings.embedding.batch_size)):
                        encoded = embedder.encode_products(batch)
                        vectors[offset:offset + len(batch)] = encoded
                        for product in batch:
                            stream.write(json.dumps([product.locale, product.product_id]) + '\n')
                        offset += len(batch)
                        if offset % (settings.embedding.batch_size * 50) == 0 or offset == count:
                            print(f'编码商品 {offset}/{count}', flush=True)
                if offset != count:
                    raise ValueError('商品编码数量发生变化')
                vectors.flush()
            finally:
                vectors._mmap.close()
        VectorIndex.finalize(stage, signature=embedder.signature)
        files = {p.name: sha256(p) for p in sorted(stage.iterdir()) if p.is_file()}
        identity = {'dataset_id': manifest['dataset_id'], 'runtime_signature': runtime_signature(settings), 'files': files}
        write_json(stage / 'release.json', {
            **identity, 'release_id': 'index-' + fingerprint(identity)[:16],
            'kind': 'indexes', 'catalog_scope': settings.data.catalog_scope,
            'settings_snapshot': asdict(settings),
        })
        verify_release(stage, settings)
        # Windows rename 在目标已存在时失败，不覆盖历史目录。
        stage.rename(output_dir)
        return output_dir
    except BaseException as exc:
        if stage.exists() and not (stage / 'failure.json').exists():
            write_json(stage / 'failure.json', {'status': 'failed', 'type': type(exc).__name__, 'message': str(exc)})
        raise
