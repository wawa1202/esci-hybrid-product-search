"""使用 train/validation 训练 LambdaRank，并创建独立的新发布目录。"""

from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from product_search.artifacts import fingerprint, sha256, verify_release, write_json
from product_search.ranking.dataset import _verify_training_inputs, build_training_data
from product_search.ranking.features import FEATURE_NAMES, _active_features
from product_search.retrieval.embedder import Embedder
from product_search.settings import Settings


def train_ranker(
    settings: Settings, dataset_dir: str | Path, release_dir: str | Path,
    output_dir: str | Path, *, embedder=None, drop_features=(),
) -> Path:
    """固定 CPU 四线程和增益，验证集早停；原 release 始终只读。

    feature_names 保存实际保留列，drop_features 保存消融列。
    release.json 最后写入，只有它存在且校验通过才视为完整发布。
    """
    active = _active_features(drop_features)
    dropped = tuple(name for name in FEATURE_NAMES if name not in active)
    output = Path(output_dir)
    if not output.is_absolute():
        raise ValueError('output_dir 必须是 CLI 指定的绝对路径')
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'不覆盖已有输出目录：{output}')
    dataset_dir, release_dir = Path(dataset_dir).resolve(), Path(release_dir).resolve()
    release = _verify_training_inputs(settings, dataset_dir, release_dir)
    if embedder is None:
        embedder = Embedder(settings)
    train = build_training_data(settings, dataset_dir, release_dir, embedder, partition='train')
    valid = build_training_data(settings, dataset_dir, release_dir, embedder, partition='validation')
    for name, data in [('train', train), ('validation', valid)]:
        if not data.group or sum(data.group) != len(data.y) or len(data.X) != len(data.y):
            raise ValueError(f'{name} 为空或分组行数不一致')
    if set(train.query_keys).intersection(valid.query_keys):
        raise ValueError('train 与 validation 查询组重叠')
    columns = [FEATURE_NAMES.index(name) for name in active]

    import lightgbm as lgb

    params = {
        'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [settings.evaluation.ranking_k],
        'device_type': 'cpu', 'num_threads': 4, 'seed': settings.project.seed,
        'data_random_seed': settings.project.seed, 'feature_fraction_seed': settings.project.seed,
        'bagging_seed': settings.project.seed, 'deterministic': True, 'force_col_wise': True,
        'label_gain': [0, 1, 3, 7], 'num_leaves': settings.training.num_leaves,
        'learning_rate': settings.training.learning_rate, 'verbosity': -1,
    }
    training = lgb.Dataset(train.X[:, columns], label=train.y, group=train.group,
                           feature_name=list(active))
    validation = lgb.Dataset(valid.X[:, columns], label=valid.y, group=valid.group,
                             feature_name=list(active), reference=training)
    booster = lgb.train(
        params, training, num_boost_round=settings.training.n_estimators,
        valid_sets=[validation], valid_names=['validation'],
        callbacks=[lgb.early_stopping(settings.training.early_stopping_rounds, verbose=False)],
    )
    best_iteration = booster.best_iteration or booster.current_iteration()
    # 所有输入与训练先验证成功，再独占创建目录；失败时不会留下有效 release 标记。
    output.mkdir(parents=True, exist_ok=False)
    for name in release['files']:
        if name in {'ranker.txt', 'model_manifest.json', 'release.json'}:
            continue
        with (release_dir / name).open('rb') as source, (output / name).open('xb') as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    with (output / 'ranker.txt').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(booster.model_to_string(num_iteration=best_iteration))
    write_json(output / 'model_manifest.json', {
        'feature_names': list(active), 'drop_features': list(dropped),
        'runtime_signature': release['runtime_signature'], 'dataset_id': release['dataset_id'],
        'best_iteration': int(best_iteration), 'model_sha256': sha256(output / 'ranker.txt'),
        'train_groups': len(train.group), 'train_rows': len(train.y),
        'valid_groups': len(valid.group), 'valid_rows': len(valid.y),
    })
    files = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    for name, expected in release['files'].items():
        if name not in {'ranker.txt', 'model_manifest.json', 'release.json'}:
            if files.get(name) != expected:
                raise ValueError(f'复制时源发布文件发生变化：{name}')
    new_release = {
        **release, 'kind': 'ranker', 'parent_release_id': release['release_id'],
        'release_id': fingerprint({'files': files, 'parent': release['release_id'],
                                   'nonce': uuid4().hex}),
        'files': files,
    }
    write_json(output / 'release.json', new_release)
    verify_release(output, settings)
    return output
