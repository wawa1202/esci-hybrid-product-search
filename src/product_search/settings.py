"""读取和校验项目配置；不加载模型，不创建目录。"""

import math
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import get_type_hints


class ConfigError(ValueError):
    """配置内容不符合本项目要求。"""


@dataclass(frozen=True)
class ProjectSettings:
    profile: str
    locale: str
    seed: int


@dataclass(frozen=True)
class DataSettings:
    raw_dir: str
    processed_dir: str
    reports_dir: str
    artifacts_dir: str
    validation_fraction: float
    sample_groups_per_partition: int
    catalog_scope: str


@dataclass(frozen=True)
class TextSettings:
    version: str
    max_tokens: int


@dataclass(frozen=True)
class BM25Settings:
    k1: float
    b: float
    top_k: int


@dataclass(frozen=True)
class EmbeddingSettings:
    model_id: str
    revision: str
    device: str
    batch_size: int
    max_seq_length: int
    normalize_embeddings: bool


@dataclass(frozen=True)
class RetrievalSettings:
    vector_block_size: int
    dense_top_k: int
    rrf_constant: int
    rerank_top_k: int


@dataclass(frozen=True)
class TrainingSettings:
    objective: str
    num_threads: int
    num_leaves: int
    learning_rate: float
    n_estimators: int
    early_stopping_rounds: int
    label_gain: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationSettings:
    ranking_k: int
    retrieval_k: int
    bootstrap_iterations: int


@dataclass(frozen=True)
class ServiceSettings:
    host: str
    port: int
    max_query_chars: int
    max_top_k: int
    max_concurrent_searches: int


@dataclass(frozen=True)
class Settings:
    project_root: Path
    config_path: Path

    project: ProjectSettings
    data: DataSettings
    text: TextSettings
    bm25: BM25Settings
    embedding: EmbeddingSettings
    retrieval: RetrievalSettings
    training: TrainingSettings
    evaluation: EvaluationSettings
    service: ServiceSettings

    def resolve_path(self, value: str) -> Path:
        """配置中的相对路径统一相对于项目根目录。"""
        path = Path(value)

        if not path.is_absolute():
            path = self.project_root / path

        return path.resolve()

    def require_embedding_ready(self) -> None:
        """向量编码前调用；要求固定为完整的模型提交 SHA。"""
        if re.fullmatch(
            r"[0-9a-fA-F]{40}",
            self.embedding.revision,
        ) is None:
            raise ConfigError(
                "embedding.revision 必须填写真实模型的完整提交 SHA"
            )


SECTION_TYPES = {
    "project": ProjectSettings,
    "data": DataSettings,
    "text": TextSettings,
    "bm25": BM25Settings,
    "embedding": EmbeddingSettings,
    "retrieval": RetrievalSettings,
    "training": TrainingSettings,
    "evaluation": EvaluationSettings,
    "service": ServiceSettings,
}


def _check_keys(
    actual: dict,
    expected: set[str],
    location: str,
) -> None:
    """拒绝缺失字段和多余字段，避免配置拼写错误被忽略。"""
    missing = expected - set(actual)
    extra = set(actual) - expected

    if missing or extra:
        raise ConfigError(
            f"{location}: "
            f"缺少字段 {sorted(missing)}；"
            f"未知字段 {sorted(extra)}"
        )


def _read_section(name: str, raw: object, cls: type):
    """先检查字段和类型，再构造对应的配置对象。"""
    if not isinstance(raw, dict):
        raise ConfigError(f"{name} 必须是 TOML 配置块")

    _check_keys(
        raw,
        {field.name for field in fields(cls)},
        name,
    )

    values = {}

    for key, expected_type in get_type_hints(cls).items():
        value = raw[key]
        location = f"{name}.{key}"

        if expected_type is float:
            if type(value) not in (int, float):
                raise ConfigError(f"{location} 必须是数值")

            try:
                value = float(value)
            except OverflowError as exc:
                raise ConfigError(
                    f"{location} 数值过大"
                ) from exc

            if not math.isfinite(value):
                raise ConfigError(
                    f"{location} 必须是有限数值"
                )

        elif expected_type == tuple[int, ...]:
            if type(value) is not list or any(
                type(x) is not int for x in value
            ):
                raise ConfigError(
                    f"{location} 必须是整数数组"
                )

            value = tuple(value)

        elif type(value) is not expected_type:
            raise ConfigError(
                f"{location} 必须是 {expected_type.__name__}"
            )

        if isinstance(value, str) and not value.strip():
            raise ConfigError(
                f"{location} 不能为空字符串"
            )

        values[key] = value

    return cls(**values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _validate(settings: Settings) -> None:
    """检查当前项目支持的取值范围和跨字段约束。"""
    p = settings.project
    d = settings.data
    t = settings.text
    b = settings.bm25
    e = settings.embedding
    r = settings.retrieval
    train = settings.training
    ev = settings.evaluation
    service = settings.service

    # 1. 项目与数据范围。
    _require(
        p.profile in {"dev", "full"},
        "project.profile 只能是 dev 或 full",
    )
    _require(
        p.locale == "us",
        "当前项目仅支持 project.locale=us",
    )
    _require(
        0 <= p.seed <= 2**31 - 1,
        "project.seed 必须在 0 到 2147483647 之间",
    )
    _require(
        0 < d.validation_fraction < 1,
        "data.validation_fraction 必须在 0 和 1 之间",
    )

    if p.profile == "dev":
        _require(
            d.sample_groups_per_partition > 0,
            "dev 模式的查询组抽样数必须大于 0",
        )
        _require(
            d.catalog_scope == "sampled_judged_union",
            "dev 必须使用 sampled_judged_union",
        )
    else:
        _require(
            d.sample_groups_per_partition == 0,
            "full 模式必须设抽样数为 0，表示不抽样",
        )
        _require(
            d.catalog_scope == "full_locale_catalog",
            "full 必须使用 full_locale_catalog",
        )

    # 2. 文本、关键词检索和向量编码。
    _require(
        t.version == "title-brand-bullet-v1",
        "当前仅支持 title-brand-bullet-v1 文本策略",
    )
    _require(
        b.k1 > 0,
        "bm25.k1 必须大于 0",
    )
    _require(
        0 <= b.b <= 1,
        "bm25.b 必须在 0 到 1 之间",
    )
    _require(
        e.model_id == "sentence-transformers/all-MiniLM-L6-v2",
        "当前仅支持本地 MiniLM 基线",
    )
    _require(
        e.device in {"cpu", "cuda"},
        "embedding.device 只能是 cpu 或 cuda",
    )
    _require(
        1 <= e.max_seq_length <= 256,
        "embedding.max_seq_length 必须在 1 到 256 之间",
    )
    _require(
        e.normalize_embeddings,
        "当前点积检索要求 normalize_embeddings=true",
    )

    # 3. 集中检查必须为正整数的参数。
    positive_fields = {
        "text.max_tokens": t.max_tokens,
        "bm25.top_k": b.top_k,
        "embedding.batch_size": e.batch_size,
        "retrieval.vector_block_size": r.vector_block_size,
        "retrieval.dense_top_k": r.dense_top_k,
        "retrieval.rrf_constant": r.rrf_constant,
        "retrieval.rerank_top_k": r.rerank_top_k,
        "training.num_threads": train.num_threads,
        "training.n_estimators": train.n_estimators,
        "training.early_stopping_rounds": train.early_stopping_rounds,
        "evaluation.ranking_k": ev.ranking_k,
        "evaluation.retrieval_k": ev.retrieval_k,
        "evaluation.bootstrap_iterations": ev.bootstrap_iterations,
        "service.max_query_chars": service.max_query_chars,
        "service.max_top_k": service.max_top_k,
        "service.max_concurrent_searches": service.max_concurrent_searches,
    }

    for name, value in positive_fields.items():
        _require(value > 0, f"{name} 必须大于 0")

    # 4. 排序模型约束。
    _require(
        train.objective == "lambdarank",
        "当前仅支持 lambdarank",
    )
    _require(
        train.num_leaves >= 2,
        "training.num_leaves 必须至少为 2",
    )
    _require(
        0 < train.learning_rate <= 1,
        "本项目要求 learning_rate 在 0 和 1 之间，包含 1",
    )
    _require(
        train.early_stopping_rounds < train.n_estimators,
        "早停轮数必须小于最大训练轮数",
    )
    _require(
        train.label_gain == (0, 1, 3, 7),
        "当前实验固定 label_gain=[0, 1, 3, 7]",
    )

    # 5. 评估与服务输出不能超过我们保留的候选深度。
    _require(
        ev.retrieval_k <= min(
            b.top_k,
            r.dense_top_k,
            r.rerank_top_k,
        ),
        "召回评估深度不能超过各路召回及融合保留深度",
    )
    _require(
        ev.ranking_k <= r.rerank_top_k,
        "排序评估深度不能超过重排候选数",
    )
    _require(
        service.max_top_k <= r.rerank_top_k,
        "接口返回上限不能超过重排候选数",
    )
    _require(
        1 <= service.port <= 65535,
        "service.port 必须在 1 到 65535 之间",
    )


def load_settings(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> Settings:
    """相对配置路径和配置内的数据路径，均以显式项目根目录为基准。"""
    root = Path(project_root).resolve()
    path = Path(config_path)

    if not path.is_absolute():
        path = root / path

    path = path.resolve()

    try:
        raw = tomllib.loads(
            path.read_text(encoding="utf-8-sig")
        )
    except (
        OSError,
        UnicodeError,
        tomllib.TOMLDecodeError,
    ) as exc:
        raise ConfigError(
            f"无法读取配置 {path}: {exc}"
        ) from exc

    _check_keys(
        raw,
        set(SECTION_TYPES),
        "配置顶层",
    )

    sections = {
        name: _read_section(name, raw[name], cls)
        for name, cls in SECTION_TYPES.items()
    }

    settings = Settings(
        project_root=root,
        config_path=path,
        **sections,
    )

    _validate(settings)

    return settings