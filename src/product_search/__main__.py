"""允许使用 python -m product_search 启动命令行。"""

from product_search.cli import run


def main() -> None:
    """启动统一命令行入口。"""
    raise SystemExit(run())


if __name__ == "__main__":
    main()