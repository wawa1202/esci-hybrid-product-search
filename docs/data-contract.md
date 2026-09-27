# 数据契约

项目只支持 `us`。商品身份是 `(locale, product_id)`，查询身份是 `(locale, query_id)`，不能只用裸 ID。

`Judgment.esci_label` 属于 query-product 关系：E（Exact）是精确匹配，S（Substitute）是可替代商品，C（Complement）是互补/配套商品，I（Irrelevant）是不相关商品。S 和 C 有明确业务含义，不能简单称为“较相关”和“不理想”。例如搜索手机时，相应型号手机是 E，能满足类似需求的替代手机可能是 S，配套手机壳可能是 C。最终以数据中的人工标签为准。一个 Query 对不同 Product 可有不同标签，所以标签放在 `Judgment`。来源：[ESCI 官方 README](https://github.com/amazon-science/esci-data)。

训练映射为 I=0、C=1、S=2、E=3，LightGBM gain 固定 `[0,1,3,7]`。未标注商品是 unknown，不能自动当作 I。训练只读 train，早停只看 validation；test 需冻结 release 后显式允许。线上只加载 catalog、索引和模型，不加载 labels。
