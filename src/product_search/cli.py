"""项目命令行入口的业务编排，不实现数据处理算法。"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from product_search.data.audit_store import (
    audit_dataset,
    save_audit_report,
)
from product_search.data.download import ASSETS
from product_search.data.prepare import (
    _collect_queries,
    _plan_queries,
    prepare_dataset,
)
from product_search.data.store import DataConflictError
from product_search.settings import (
    ConfigError,
    Settings,
    load_settings,
)


def _add_config_argument(
    parser: argparse.ArgumentParser,
) -> None:
    """给子命令添加公共配置参数。"""
    parser.add_argument(
        "--config",
        default="configs/dev.toml",
        type=Path,
        help="配置文件路径，默认 configs/dev.toml",
    )

    parser.add_argument(
        "--project-root",
        default=Path.cwd(),
        type=Path,
        help="项目根目录，默认当前目录",
    )


def build_parser() -> argparse.ArgumentParser:
    """创建命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="product-search",
        description="公开 ESCI 商品搜索项目的数据准备与审计命令",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="准备 SQLite 数据集",
    )
    _add_config_argument(prepare_parser)
    prepare_parser.set_defaults(
        handler=_handle_prepare,
    )

    audit_parser = subparsers.add_parser(
        "audit",
        help="只读审计已发布数据集",
    )
    _add_config_argument(audit_parser)

    audit_parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "数据集目录；"
            "不提供时要求配置对应的唯一正式数据集"
        ),
    )

    audit_parser.add_argument(
        "--report",
        type=Path,
        help=(
            "报告输出路径；"
            "默认写入 reports/<run_id>/data-audit.json"
        ),
    )

    audit_parser.set_defaults(
        handler=_handle_audit,
    )

    preview_parser = subparsers.add_parser(
        "split-preview",
        help="预览查询分区和抽样结果",
    )
    _add_config_argument(preview_parser)
    preview_parser.set_defaults(
        handler=_handle_split_preview,
    )

    index_parser = subparsers.add_parser('build-indexes', help='构建 BM25 和本地向量索引')
    _add_config_argument(index_parser)
    index_parser.add_argument('--dataset', type=Path, required=True)
    index_parser.add_argument('--output', type=Path, required=True)
    index_parser.set_defaults(handler=_handle_build_indexes)

    train_parser = subparsers.add_parser('train-ranker', help='训练 LambdaRank 排序器')
    _add_config_argument(train_parser)
    train_parser.add_argument('--dataset', type=Path, required=True)
    train_parser.add_argument('--release', type=Path, required=True)
    train_parser.add_argument('--output', type=Path, required=True)
    train_parser.add_argument('--drop-feature', action='append', default=[])
    train_parser.set_defaults(handler=_handle_train_ranker)

    eval_parser = subparsers.add_parser('evaluate', help='运行 validation 或冻结后的 test 评估')
    _add_config_argument(eval_parser)
    eval_parser.add_argument('--dataset', type=Path, required=True)
    eval_parser.add_argument('--release', type=Path, required=True)
    eval_parser.add_argument('--output', type=Path, required=True)
    eval_parser.add_argument('--partition', choices=['validation', 'test'], default='validation')
    eval_parser.add_argument('--allow-test', action='store_true')
    eval_parser.set_defaults(handler=_handle_evaluate)

    serve_parser = subparsers.add_parser('serve', help='启动本地 FastAPI 搜索服务')
    _add_config_argument(serve_parser)
    serve_parser.add_argument('--release', type=Path, required=True)
    serve_parser.set_defaults(handler=_handle_serve)

    freeze_parser = subparsers.add_parser('freeze', help='冻结同一 release 的 validation 结果，允许之后一次 test')
    _add_config_argument(freeze_parser)
    freeze_parser.add_argument('--release', type=Path, required=True)
    freeze_parser.add_argument('--validation-report', type=Path, required=True)
    freeze_parser.set_defaults(handler=_handle_freeze)

    benchmark_parser = subparsers.add_parser('benchmark', help='预热后测量本地搜索流水线延迟')
    _add_config_argument(benchmark_parser)
    benchmark_parser.add_argument('--release', type=Path, required=True)
    benchmark_parser.add_argument('--output', type=Path, required=True)
    benchmark_parser.add_argument('--runs', type=int, default=30)
    benchmark_parser.set_defaults(handler=_handle_benchmark)

    return parser


def _load(
    args: argparse.Namespace,
) -> Settings:
    """根据命令行参数读取并校验配置。"""
    return load_settings(
        args.config,
        project_root=args.project_root,
    )


def _print_json(value: object) -> None:
    """统一用 JSON 输出命令结果。"""
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def _handle_prepare(
    args: argparse.Namespace,
) -> int:
    """执行数据准备命令。"""
    settings = _load(args)

    directory = prepare_dataset(settings)

    manifest = json.loads(
        (directory / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    _print_json(
        {
            "command": "prepare",
            "status": manifest["status"],
            "dataset_dir": str(directory),
            "basic_integrity": manifest[
                "basic_integrity"
            ],
        }
    )

    return 0


def _find_dataset(
    settings: Settings,
) -> Path:
    """
    自动查找唯一正式数据集。

    如果同一 profile 有多个数据集，
    就要求用户使用 --dataset 明确指定。
    """
    root = settings.resolve_path(
        settings.data.processed_dir
    )

    if not root.is_dir():
        raise FileNotFoundError(
            f"找不到已处理数据目录：{root}"
        )

    candidates: list[Path] = []

    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue

        if directory.name.startswith("."):
            continue

        manifest_path = directory / "manifest.json"

        if not manifest_path.is_file():
            continue

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        identity = manifest.get(
            "identity",
            {},
        )

        project = identity.get(
            "project",
            {},
        )

        if (
            manifest.get("status") == "prepared"
            and project.get("profile")
            == settings.project.profile
            and project.get("locale")
            == settings.project.locale
        ):
            candidates.append(directory)

    if not candidates:
        raise FileNotFoundError(
            "没有找到配置对应的正式数据集，"
            "请先运行 prepare"
        )

    if len(candidates) > 1:
        names = ", ".join(
            path.name
            for path in candidates
        )

        raise ValueError(
            "找到多个候选数据集，"
            f"请使用 --dataset 明确指定：{names}"
        )

    return candidates[0]


def _handle_audit(
    args: argparse.Namespace,
) -> int:
    """执行只读数据审计。"""
    settings = _load(args)

    directory = args.dataset

    if directory is None:
        directory = _find_dataset(settings)

    elif not directory.is_absolute():
        directory = (
            settings.project_root
            / directory
        )

    report = audit_dataset(directory)

    output = args.report

    if output is None:
        run_id = datetime.now(
            timezone.utc
        ).strftime(
            "audit-%Y%m%dT%H%M%S%fZ"
        )

        output = (
            settings.resolve_path(
                settings.data.reports_dir
            )
            / run_id
            / "data-audit.json"
        )

    elif not output.is_absolute():
        output = settings.project_root / output

    saved = save_audit_report(
        report,
        output,
    )

    _print_json(
        {
            "command": "audit",
            "status": report["status"],
            "dataset_dir": report[
                "dataset_dir"
            ],
            "report": str(saved),
            "products": report["products"],
            "partitions": {
                name: {
                    "queries": value["queries"],
                    "judgments": value[
                        "judgments"
                    ],
                }
                for name, value in report[
                    "partitions"
                ].items()
            },
        }
    )

    return 0


def _handle_split_preview(
    args: argparse.Namespace,
) -> int:
    """预览查询分区和抽样结果，不写入正式数据集。"""
    settings = _load(args)

    examples = (
        settings.resolve_path(
            settings.data.raw_dir
        )
        / ASSETS["examples"]["filename"]
    )

    # 复用 prepare.py 中已有的查询规划逻辑，
    # 避免 CLI 再复制一套划分算法。
    with TemporaryDirectory(
        prefix="product-search-cli-"
    ) as temporary:
        rejects = Path(temporary) / "rejects.jsonl"
        rejects.touch()

        collected = _collect_queries(
            examples,
            settings.project.locale,
            rejects,
        )

        selected = _plan_queries(
            collected,
            settings,
        )

    groups = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    queries = {
        name: 0
        for name in groups
    }

    for query, _source, partition in selected.values():
        queries[partition] += 1

        normalized_group = (
            " ".join(
                query.text.casefold().split()
            )
        )

        groups[partition].add(
            (
                query.locale,
                normalized_group,
            )
        )

    _print_json(
        {
            "command": "split-preview",
            "profile": settings.project.profile,
            "locale": settings.project.locale,
            "seed": settings.project.seed,
            "sample_groups_per_partition": (
                settings.data.sample_groups_per_partition
            ),
            "partitions": {
                name: {
                    "queries": queries[name],
                    "query_groups": len(
                        groups[name]
                    ),
                }
                for name in groups
            },
        }
    )

    return 0


def _handle_build_indexes(args: argparse.Namespace) -> int:
    from product_search.indexing import build_indexes
    from product_search.retrieval.embedder import Embedder
    settings = _load(args)
    dataset = args.dataset if args.dataset.is_absolute() else settings.project_root / args.dataset
    output = args.output if args.output.is_absolute() else settings.project_root / args.output
    result = build_indexes(settings, dataset, output, embedder=Embedder(settings))
    _print_json({'command': 'build-indexes', 'release_dir': str(result)})
    return 0


def _handle_train_ranker(args: argparse.Namespace) -> int:
    from product_search.ranking.train import train_ranker
    from product_search.retrieval.embedder import Embedder
    settings = _load(args)
    values = [Path(value) if Path(value).is_absolute() else settings.project_root / value
              for value in (args.dataset, args.release, args.output)]
    result = train_ranker(settings, *values[:2], values[2], embedder=Embedder(settings),
                          drop_features=tuple(args.drop_feature))
    _print_json({'command': 'train-ranker', 'release_dir': str(result)})
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    from product_search.evaluation.evaluate import evaluate
    settings = _load(args)
    values = [Path(value) if Path(value).is_absolute() else settings.project_root / value
              for value in (args.dataset, args.release, args.output)]
    result = evaluate(settings, *values, partition=args.partition, allow_test=args.allow_test)
    _print_json({'command': 'evaluate', 'report_dir': str(result)})
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from product_search.service.api import create_app
    settings = _load(args)
    release = args.release if args.release.is_absolute() else settings.project_root / args.release
    uvicorn.run(create_app(settings, release), host=settings.service.host, port=settings.service.port)
    return 0


def _handle_freeze(args):
    from product_search.evaluation.freeze import freeze_release
    settings = _load(args)
    result = freeze_release(settings, settings.resolve_path(str(args.release)),
                            settings.resolve_path(str(args.validation_report)))
    _print_json({'frozen': str(result)})
    return 0


def _handle_benchmark(args):
    from product_search.artifacts import write_json
    from product_search.evaluation.benchmark import summarize_timings
    from product_search.service.search import SearchEngine
    settings = _load(args)
    if not 1 <= args.runs <= 10000:
        raise ValueError('runs 必须在 1 到 10000 之间')
    output = settings.resolve_path(str(args.output))
    if output.exists():
        raise FileExistsError(output)
    queries = ['running shoes', 'wireless headphones', 'coffee maker', 'phone case', 'red dress']
    with SearchEngine(settings, settings.resolve_path(str(args.release))) as engine:
        for query in queries:
            engine.search(query, top_k=5)
        samples = [engine.search(queries[i % len(queries)], top_k=5)['timings_ms'] for i in range(args.runs)]
        report = {'release_id': engine.manifest['release_id'], 'warmup': len(queries),
                  'runs': args.runs, 'queries': queries, 'timings': summarize_timings(samples),
                  'scope': '单进程、预热后本地调用，不含 HTTP；不是并发吞吐测试'}
    write_json(output, report)
    _print_json(report)
    return 0


def run(
    argv: Sequence[str] | None = None,
) -> int:
    """执行命令并返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.handler(args)

    except (
        ConfigError,
        DataConflictError,
        FileNotFoundError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(
            f"错误：{exc}",
            file=sys.stderr,
        )
        return 2
