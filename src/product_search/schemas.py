from dataclasses import dataclass
from typing import Literal

@dataclass
class Query:
    # 在这里声明 query_id、text、locale 三个字段及其类型
    query_id: str
    text: str
    locale: str

@dataclass
class Product:
    # 必填：product_id、locale、title
    # 可选：description、bullet_point、brand、color，默认值为空字符串
    product_id: str
    locale: str
    title: str
    description: str = ""
    bullet_point: str = ""
    brand: str = ""
    color: str = ""

@dataclass
class Judgment:
    """一个查询与一个商品之间的人工相关性标注。"""

    query_id: str
    product_id: str
    locale: str
    esci_label: Literal["E", "S", "C", "I"]


@dataclass
class SearchResult:
    """某个查询返回的一条商品排序结果。"""

    query_id: str
    product_id: str
    locale: str
    rank: int
    score: float

