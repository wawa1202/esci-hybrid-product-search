"""校验模型发布身份，按与训练相同的特征和消融列进行稳定排序。"""

from pathlib import Path

import numpy as np

from product_search.artifacts import read_json, runtime_signature, sha256, verify_release
from product_search.contracts import ProductKey, get_product_key
from product_search.ranking.features import FEATURE_NAMES, _active_features, build_features
from product_search.schemas import Product, Query, SearchResult
from product_search.settings import Settings


class Ranker:
    """加载完整发布中的 LightGBM 模型；不打开标签数据库。"""

    def __init__(self, release_dir: str | Path, settings: Settings):
        self.release_dir = Path(release_dir).resolve()
        self.settings = settings
        self.release = verify_release(self.release_dir, settings)
        if not {'ranker.txt', 'model_manifest.json'} <= self.release['files'].keys():
            raise ValueError('release 缺少模型及 model_manifest 文件哈希')
        manifest = read_json(self.release_dir / 'model_manifest.json')
        if manifest.get('runtime_signature') != runtime_signature(settings):
            raise ValueError('模型 runtime signature 不匹配')
        if manifest.get('dataset_id') != self.release['dataset_id']:
            raise ValueError('模型 dataset_id 不匹配')
        if not isinstance(manifest.get('drop_features'), list):
            raise ValueError('model manifest 缺少 drop_features 特征列表')
        self.feature_names = _active_features(manifest['drop_features'])
        if manifest.get('feature_names') != list(self.feature_names):
            raise ValueError('模型 feature_names 特征顺序或消融列不匹配')
        if manifest.get('model_sha256') != sha256(self.release_dir / 'ranker.txt'):
            raise ValueError('模型 SHA256 校验失败')
        iteration = manifest.get('best_iteration')
        if type(iteration) is not int or iteration < 1:
            raise ValueError('模型 best_iteration 无效')

        import lightgbm as lgb

        self.model = lgb.Booster(model_file=str(self.release_dir / 'ranker.txt'))
        if self.model.feature_name() != list(self.feature_names):
            raise ValueError('LightGBM 模型 feature_names 不匹配')
        if self.model.current_iteration() != iteration:
            raise ValueError('模型 best_iteration 与树数量不一致')
        self.best_iteration = iteration
        self.model_manifest = manifest
        self._columns = [FEATURE_NAMES.index(name) for name in self.feature_names]

    def predict(
        self, query: Query, products: list[Product], *,
        bm25_scores: dict[ProductKey, float], dense_scores: dict[ProductKey, float],
        k: int | None = None,
    ) -> list[SearchResult]:
        """按分数降序排列，同分按完整商品键升序，rank 从 1 开始。"""
        if k is not None and (type(k) is not int or k < 1):
            raise ValueError('k 必须为正整数或 None')
        X = build_features(query, products, bm25_scores=bm25_scores,
                           dense_scores=dense_scores, settings=self.settings)
        if not products:
            return []
        scores = np.asarray(self.model.predict(
            X[:, self._columns], num_iteration=self.best_iteration, num_threads=4,
        ))
        if scores.shape != (len(products),) or not np.isfinite(scores).all():
            raise ValueError('模型预测分数形状或 finite 数值无效')
        order = sorted(range(len(products)),
                       key=lambda i: (-float(scores[i]), get_product_key(products[i])))[:k]
        return [SearchResult(query.query_id, products[i].product_id, products[i].locale,
                             rank, float(scores[i])) for rank, i in enumerate(order, 1)]
