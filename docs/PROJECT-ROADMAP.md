# 商品搜索与排序系统：完整实施总方案

> 本文一次性固定后续架构、目录、接口、实现顺序和验收标准。之后只按 T01—T32 执行，不再重新设计项目。

## 1. 项目目标

基于公开 ESCI 数据构建一个可解释、可评估、可本地运行的商品搜索系统：

~~~text
商品目录 + 查询标注
        ↓
数据准备与审计
        ↓
BM25 关键词召回 ─┐
向量召回       ─┼→ RRF 融合 → 特征 → LightGBM 排序
                  ↓
              FastAPI 搜索服务
~~~

最终交付：

- 给定候选集内的排序实验；
- 面向 US 商品目录的关键词、向量和融合检索；
- LightGBM LambdaRank 重排；
- NDCG、MRR、已知标注召回、延迟和错误分析；
- 可追溯的数据、索引、模型和实验版本；
- 本地 API 和最小搜索页面；
- 有真实报告支撑的简历项目。

项目使用 Amazon Science 的公开研究作为业务背景，使用公开 ESCI 数据，不使用 Amazon 内部数据、内部商品库、线上日志、内部模型或内部指标。公开标注不是完整商品库的全量相关性答案。

## 2. 当前事实

已经存在：

~~~text
src/product_search/schemas.py
src/product_search/validation.py
src/product_search/data/download.py
src/product_search/data/read_parquet.py
src/product_search/data/normalize.py
src/product_search/data/build_dataset.py
src/product_search/data/audit.py
src/product_search/data/split.py
data/raw/*parquet
data/raw/*manifest.json
~~~

已验证：

- examples 和 products 文件大小、SHA256、来源记录正确；
- PyArrow 25.0.1 已安装；
- 真实 Parquet 可以分批读取；
- 真实样本可以转换为 Query、Product、Judgment；
- US Task 1 官方 train 有 20,888 个查询和 419,653 条标注；
- US Task 1 官方 test 有 8,956 个查询和 181,701 条标注；
- US 商品目录约 1,215,854 行；
- 当前扫描未发现 train/test 查询 ID 或归一化文本重叠。

尚未完成：

- SQLite 数据准备；
- 全量审计报告；
- BM25、向量、RRF、LightGBM；
- 评估报告；
- API 和前端；
- 最终 release。

## 3. 技术决策

| 选择 | 决定 |
|---|---|
| 任务主线 | 商品搜索与 Learning-to-Rank |
| 第一市场 | US |
| 数据范围 | ESCI small version，即 Task 1 |
| 训练验证 | 只在官方 train 内按查询文本组划分 |
| 测试 | 保留官方 test，配置冻结后只评估一次 |
| 商品存储 | SQLite，避免把百万商品全部放进 Python 对象列表 |
| 关键词召回 | 自写 BM25 倒排索引 |
| 向量召回 | Sentence Transformers 编码，先做 NumPy 精确搜索 |
| 融合 | RRF，只按排名融合 |
| 排序 | LightGBM LambdaRank，CPU |
| 服务 | FastAPI 单进程 |
| GPU | 4060 可选用于 embedding；主线不依赖 GPU |
| Dify | 只做外围演示，不进入训练和评估主链路 |
| FAISS | 精确向量搜索通过后再适配 |
| Milvus、Kafka、Spark、Hive | 当前不加入，出现真实规模或部署需求再扩展 |

参考组件：

- Apache Arrow：Parquet 分批读取；
- SQLite：目录、查询、标注和版本数据持久化；
- Sentence Transformers：文本向量编码；
- LightGBM：Learning-to-Rank；
- FastAPI：HTTP 服务；
- FAISS：可选 ANN 向量后端。

## 4. 最终目录

~~~text
product-search/
├── README.md
├── pyproject.toml
├── requirements-lock.txt
├── .gitignore
├── main.py
├── configs/
│   ├── dev.toml
│   └── full.toml
├── data/
│   ├── raw/
│   └── processed/<dataset_id>/
│       ├── catalog.sqlite
│       ├── labels.sqlite
│       ├── manifest.json
│       └── rejects.jsonl
├── artifacts/<release_id>/
│   ├── release.json
│   ├── bm25.sqlite
│   ├── embeddings.npy
│   ├── embedding_ids.jsonl
│   ├── vector_manifest.json
│   ├── ranker.txt
│   └── model_manifest.json
├── reports/<run_id>/
│   ├── data-audit.json
│   ├── retrieval-results.jsonl
│   ├── ranking-results.json
│   ├── error-analysis.jsonl
│   └── benchmark.json
├── docs/
│   ├── PROJECT-ROADMAP.md
│   ├── data-contract.md
│   ├── evaluation.md
│   ├── troubleshooting.md
│   └── interview.md
├── scripts/
│   ├── prepare_data.py
│   ├── build_indexes.py
│   ├── train_ranker.py
│   └── run_evaluation.py
├── tests/
└── src/product_search/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py
    ├── contracts.py
    ├── settings.py
    ├── artifacts.py
    ├── schemas.py
    ├── validation.py
    ├── data/
    │   ├── download.py
    │   ├── read_parquet.py
    │   ├── normalize.py
    │   ├── build_dataset.py
    │   ├── audit.py
    │   ├── split.py
    │   ├── store.py
    │   ├── prepare.py
    │   └── audit_store.py
    ├── retrieval/
    │   ├── text.py
    │   ├── bm25.py
    │   ├── embedder.py
    │   ├── vector_index.py
    │   ├── fusion.py
    │   └── ann_index.py
    ├── ranking/
    │   ├── features.py
    │   ├── dataset.py
    │   ├── train.py
    │   └── predict.py
    ├── evaluation/
    │   ├── metrics.py
    │   ├── baselines.py
    │   ├── evaluate.py
    │   ├── compare.py
    │   ├── error_analysis.py
    │   └── benchmark.py
    └── service/
        ├── search.py
        ├── api.py
        ├── observability.py
        └── static/index.html
~~~

## 5. 数据层设计

### 5.1 两个数据库

catalog.sqlite 只保存商品目录：

~~~sql
CREATE TABLE products (
    locale TEXT NOT NULL,
    product_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    bullet_point TEXT NOT NULL,
    brand TEXT NOT NULL,
    color TEXT NOT NULL,
    PRIMARY KEY (locale, product_id)
);
~~~

labels.sqlite 保存查询、划分和公开标注：

~~~sql
CREATE TABLE queries (
    locale TEXT NOT NULL,
    query_id TEXT NOT NULL,
    text TEXT NOT NULL,
    normalized_group TEXT NOT NULL,
    source_split TEXT NOT NULL,
    partition TEXT NOT NULL,
    PRIMARY KEY (locale, query_id)
);

CREATE TABLE judgments (
    locale TEXT NOT NULL,
    query_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    esci_label TEXT NOT NULL,
    PRIMARY KEY (locale, query_id, product_id),
    FOREIGN KEY (locale, query_id)
        REFERENCES queries(locale, query_id)
);
~~~

在线服务不加载 labels.sqlite，从边界上防止人工标签进入在线排序。

### 5.2 数据准备顺序

1. 检查两个原始文件的 manifest 和 SHA256。
2. 读取 examples，筛选 locale=us、small_version=1。
3. 官方 train 按 locale 加 casefold 加合并空白后的 query text 分组。
4. 分配 train 与 validation；官方 test 固定为 test。
5. 写入 queries 和 judgments。
6. dev 模式只保存选中查询涉及的商品；full 模式保存完整有效 US 商品目录。
7. 分批扫描 products，写入 catalog。
8. 做外键、重复、标签、数量和版本审计。
9. 全部通过后，将临时目录改名为正式 dataset 目录。

失败行写入 rejects.jsonl。默认发现关联或字段错误就中止，不静默删除。

## 6. 检索设计

### 6.1 统一文本策略

retrieval/text.py 提供：

~~~python
def product_text(product, *, version): ...
def query_tokens(text, *, version): ...
~~~

v1 规则：

1. 使用 Unicode casefold；
2. 非字母数字字符转为空格；
3. 折叠连续空白；
4. 商品文本按 title、brand、bullet_point 固定顺序拼接；
5. 记录 text_version。

BM25、embedding 和特征工程都调用同一套策略，不能各写一套清洗。

### 6.2 BM25

retrieval/bm25.py 建立商品倒排索引：

~~~text
term → posting(doc_id, term_frequency)
~~~

保存商品键、文档长度、term document frequency、posting、k1、b、文本版本和目录 hash。

接口：

~~~python
class BM25Index:
    @classmethod
    def build(cls, catalog, config): ...
    def search(self, query, k): ...
    def score_pairs(self, query, keys): ...
~~~

零命中返回空列表，不重复填充商品。排序采用负 score 加 locale 加 product_id 的稳定规则。

### 6.3 向量召回

embedder.py 负责加载固定 revision 的 Sentence Transformers 模型；记录模型 ID、revision、维度、dtype、device、batch size 和归一化策略。

vector_index.py 先实现精确搜索：

1. 商品向量按固定商品键顺序保存；
2. 商品键单独保存；
3. 保存行数、维度、dtype 和键 hash；
4. 查询和商品向量归一化后点积；
5. 分块计算并维护 top-k；
6. 以精确搜索作为 ANN 的正确性真值。

向量错位、维度不一致、键 hash 不一致时拒绝加载。

### 6.4 RRF

~~~python
def reciprocal_rank_fusion(
    ranked_lists,
    *,
    k_constant=60,
): ...
~~~

每路贡献为：

~~~text
1 / (k_constant + rank)
~~~

RRF 只使用 rank，不直接把 BM25 和 cosine 分数相加。相同商品合并，保留每一路的 rank 和原始分数，最后按 fused score 和商品键稳定排序。

## 7. 排序设计

### 7.1 特征

features.py 同时供训练和在线服务使用：

- query token 数；
- 商品标题 token 数；
- 标题 token 重合数和比例；
- brand 是否命中；
- color 是否命中；
- BM25 分数；
- dense cosine；
- RRF 分数；
- title、brand、bullet 的长度。

禁止使用 query_id、product_id、esci_label、split、未来点击、订单、销量、由标注表计算出的候选数和 test 标签。

特征函数不接受 Judgment，从函数签名上减少标签泄漏可能。特征名称和顺序固定，保存 feature signature。

### 7.2 训练数据

dataset.py 将每个查询的候选组成一个 group：

~~~text
X：特征矩阵
y：E=3, S=2, C=1, I=0
groups：[第1个查询候选数, 第2个查询候选数, ...]
~~~

LightGBM 的 label 是排序等级，label_gain 单独设为 I=0、C=1、S=3、E=7，不能把 gain 再转换一次。

全目录里没有公开标注的商品保持 unknown，不自动标成 I 加入训练。

### 7.3 训练和预测

train.py 使用 LambdaRank：

- 只在 train 拟合；
- validation 用于 early stopping 和参数选择；
- CPU num_threads=4；
- 保存模型文件和模型 manifest；
- manifest 记录 dataset_id、feature signature、参数、训练时间和环境。

predict.py 加载前检查版本兼容性；不兼容则拒绝。输出 SearchResult，rank 从 1 开始，同分按商品键稳定排序。

## 8. 两套评估

### 8.1 给定候选集排序

候选是某个查询公开标注过的商品。BM25、dense、RRF、LTR 使用完全相同的候选，不在该实验中比较是否召回。

主指标：

- NDCG@10；
- MRR_E@10；
- 每查询胜、负、持平。

### 8.2 目录检索

候选来自固定版本的 US 商品目录。报告 known_E_recall@100、known_ES_recall@100、judged_fraction@100、空结果率以及各阶段延迟。

未被公开标注的商品是 unknown，不自动当成 I。分母为空的查询返回 undefined，并单独报告数量。ANN Recall 只表示近似索引相对于精确向量的重合率，不是业务相关性召回。

### 8.3 指标和统计

metrics.py 实现 ndcg_at_k、mrr_exact_at_k、recall_known_at_k、judged_fraction_at_k。

固定等级：I=0、C=1、S=2、E=3。

固定 gain：I=0、C=1、S=3、E=7。

标签顺序 [I,E]、k=2 时：

~~~text
NDCG = 0.6309297536
MRR_E = 0.5
~~~

重复商品、排名断档和非法结果必须拒绝。

对照固定为 BM25、dense、RRF、LTR。至少做三组消融：去掉 dense、去掉词项重合特征、title-only 对比 title+brand+bullet。对每查询差异做固定 seed 的配对 bootstrap，报告均值差、95% 区间、有效查询数和 undefined 数。

## 9. 服务设计

### 9.1 SearchEngine

search.py 执行：

1. 校验 query、locale、top_k；
2. 调用 BM25 和 dense 召回；
3. RRF 去重和融合；
4. 从 catalog 获取候选商品；
5. 生成与训练相同的特征；
6. 用 ranker 重排；
7. 生成 SearchResult；
8. 记录 pipeline 和各阶段耗时。

ranker 缺失时可以降级到 RRF，但响应必须明确写出 pipeline=rrf_fallback。

### 9.2 API

api.py 提供：

~~~text
GET  /health
GET  /version
POST /search
~~~

search 输入 query、locale、top_k；响应包含 query、results、pipeline、request_id、阶段耗时和 release_id。

使用 FastAPI lifespan 在服务启动时加载共享索引和模型，不能每次请求重复加载。空查询和超长查询返回 4xx；内部故障返回稳定 error_code；不回显本地路径、堆栈、标签或密钥。

### 9.3 前端

static/index.html 只负责输入查询、选择市场、选择 top_k、显示商品标题、分数、来源、耗时、错误和版本，不复制搜索算法。

## 10. 固定实施顺序

### 阶段 1：数据准备

| 编号 | 文件 | 目标 | 验收 |
|---|---|---|---|
| T01 | pyproject.toml、依赖和包入口 | 固定运行方式 | 包导入和环境检查 |
| T02 | settings.py、dev.toml、full.toml | 配置可读可校验 | 非法配置拒绝 |
| T03 | contracts.py | 固定跨模块对象 | key、Hit、Candidate 测试 |
| T04 | data/store.py | SQLite 表和事务 | 重复、冲突、回滚测试 |
| T05 | data/prepare.py | raw 到数据库 | 合成 Parquet 端到端 |
| T06 | data/audit_store.py | 全量审计 | 真实 US 统计报告 |
| T07 | cli.py、__main__.py | 统一命令 | prepare、audit、split-preview |

### 阶段 2：关键词基线

| 编号 | 文件 | 目标 | 验收 |
|---|---|---|---|
| T08 | retrieval/text.py | 统一文本策略 | 空白、标点、重音测试 |
| T09 | retrieval/bm25.py | 倒排和 BM25 | 手算分数、重开一致 |
| T10 | evaluation/baselines.py | 给定候选排序 | 不读取标签值 |
| T11 | evaluation/metrics.py | 指标公式 | 手算 NDCG/MRR |
| T12 | evaluation/evaluate.py | 基线报告 | judged 与 catalog scope 分开 |

### 阶段 3：向量和融合

| 编号 | 文件 | 目标 | 验收 |
|---|---|---|---|
| T13 | retrieval/embedder.py | 固定模型编码 | 空输入、维度、1000 条 CPU 测试 |
| T14 | retrieval/vector_index.py | 精确向量检索 | 与暴力结果一致 |
| T15 | retrieval/fusion.py | RRF | 手算两路融合 |
| T16 | evaluation/evaluate.py | dense/RRF 对照 | 固定 dev 查询 |
| T17 | retrieval/ann_index.py | 可选 FAISS | ANN Recall 单列报告 |

### 阶段 4：Learning-to-Rank

| 编号 | 文件 | 目标 | 验收 |
|---|---|---|---|
| T18 | ranking/features.py | 特征定义 | 无标签、顺序稳定 |
| T19 | ranking/dataset.py | group 和标签 | 行数、groups、label 正确 |
| T20 | ranking/train.py | LambdaRank | 模型可保存加载 |
| T21 | ranking/predict.py | 重排输出 | 版本不匹配拒绝 |
| T22 | evaluation/compare.py | 对照、消融、bootstrap | test 只在冻结后评估 |

### 阶段 5：分析、服务和交付

| 编号 | 文件 | 目标 | 验收 |
|---|---|---|---|
| T23 | evaluation/error_analysis.py | 错误分类 | 至少 30 个 validation 查询 |
| T24 | evaluation/benchmark.py | 性能测量 | p50/p95/p99 和分阶段耗时 |
| T25 | service/search.py | 在线编排 | fallback、空查询、版本校验 |
| T26 | service/api.py | HTTP 接口 | health、version、search |
| T27 | service/observability.py | 结构化日志 | 不记录原始查询和标签 |
| T28 | service/static/index.html | 本地演示 | 搜索、错误和版本可见 |
| T29 | cli.py、scripts/ | 一键运行 | 重复运行不污染旧产物 |
| T30 | README 与 docs | 项目交接 | 新人可按 README 启动 |
| T31 | release compatibility tests | 兼容性边界 | 错配产物拒绝加载 |
| T32 | reports/final | 最终交付 | 测试、指标、错误分析齐全 |

每个 T 任务仍由你一次实现一个完整文件或一个紧密文件组；路线、接口、验收和依赖关系已经固定，不会再临时改变项目架构。

## 11. 统一运行流程

从项目根目录运行：

~~~powershell
$env:PYTHONPATH = "$(Get-Location)\src"
& 'D:/python3127/python.exe' -m product_search prepare --config configs/dev.toml
& 'D:/python3127/python.exe' -m product_search audit --config configs/dev.toml
& 'D:/python3127/python.exe' -m product_search build-indexes --config configs/dev.toml
& 'D:/python3127/python.exe' -m product_search evaluate --config configs/dev.toml --scope judged
& 'D:/python3127/python.exe' -m product_search train --config configs/dev.toml
& 'D:/python3127/python.exe' -m product_search evaluate --config configs/dev.toml --scope catalog
& 'D:/python3127/python.exe' -m product_search serve --config configs/dev.toml
~~~

真实全量实验将 dev.toml 换成 full.toml，并使用新的 run_id。不能把 dev 的指标写成最终结果。

## 12. 资源与硬件

主线无需 GPU：Parquet、SQLite、BM25、LightGBM 都走 CPU；embedding 先用 CPU 小批量验证；4060 可将同一模型、同一 revision、同一 batch 配置迁移到 CUDA；无 GPU 云服务器可以运行 API 或 Dify，但不是完成项目的必要条件。

全目录向量本体约为 1,215,854 × 384 × 4 字节，约 1.74 GiB，还不包括文本、模型、临时矩阵和 Python 开销。因此必须使用分块计算、memmap 或磁盘索引，不能把所有向量和商品对象无计划地装入内存。

## 13. 最终简历验收

- 原始数据版本和 SHA256；
- train、validation、test 查询隔离证据；
- SQLite 全量审计；
- BM25、dense、RRF、LTR 的同口径对照；
- NDCG@10、MRR、known recall、延迟；
- 至少三组消融；
- 至少 30 个错误查询分类；
- 特征版本、防泄漏测试；
- API health、version、search；
- 数据、索引、模型、报告的兼容性检查；
- 一键复现命令。

推荐简历表述：

> 基于公开 ESCI 商品搜索数据构建可复现的多路召回与 Learning-to-Rank 系统，设计 SQLite 数据准备、BM25/向量/RRF 召回和 LightGBM LambdaRank 重排链路；在固定查询组划分上完成多模型对照、特征消融与错误分析，并实现带版本校验和分阶段延迟观测的 FastAPI 搜索服务。

方括号中的指标只能替换成最终报告中的真实数字。

## 14. 参考资料

- Amazon Science 商品检索案例：https://www.amazon.science/blog/interpretable-ensemble-models-improve-product-retrieval
- ESCI 数据集：https://github.com/amazon-science/esci-data
- Apache Arrow ParquetFile：https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html
- Sentence Transformers：https://github.com/huggingface/sentence-transformers
- LightGBM LambdaRank：https://lightgbm.readthedocs.io/en/stable/Parameters.html
- FAISS Getting Started：https://github.com/facebookresearch/faiss/wiki/Getting-started
- FastAPI lifespan：https://fastapi.tiangolo.com/advanced/events/
