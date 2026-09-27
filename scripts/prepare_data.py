"""复用统一 CLI，脚本与模块入口使用相同参数和路径规则。"""
import sys
from product_search.cli import run


if __name__ == '__main__':
    raise SystemExit(run(['prepare', *sys.argv[1:]]))
