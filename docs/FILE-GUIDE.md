# 按构建顺序理解文件职责与关联

这里按“从空目录亲手搭建这个项目”的建议顺序排列，不按文件夹名称排序，也不表示之前实际编写文件的时间顺序。

项目根目录：当前仓库根目录。下表路径均相对于这个根目录；业务源码统一位于 `src/product_search/`。

阅读每一行时回答三个问题：这个文件解决什么问题？需要前面哪些文件？做完后交给谁？每阶段先看验收目标，再写对应测试和实现。已有文件不需要删除重建，可以按此顺序重新阅读和运行。

**总顺序：工程骨架 → 数据契约 → 配置 → 原始数据 → 数据集准备与审计 → 产物校验 → 关键词/向量召回 → 索引发布 → 评估规则 → 排序模型 → 搜索流水线 → 离线评估 → HTTP 页面 → 统一入口 → 可选优化。**

## 1. 先让 Python 认识这个项目

| 顺序 | 文件 | 写什么 | 与其他文件的关系 |
|---|---|---|---|
| 1 | `pyproject.toml` | 包名、Python 版本、src 目录、依赖、测试设置 | 先写构建与包发现；console 入口和静态页面打包配置到对应阶段再补 |
| 2 | `.gitignore` | 排除环境、缓存、模型、原始大数据及临时文件 | 避免以后把运行产物混进业务源码 |
| 3 | `src/product_search/__init__.py` | 声明顶层 Python 包 | 后面才能统一使用 `product_search.xxx` 导入 |

子包创建到哪一步，就补该目录的 `__init__.py`。初期可以只有说明文字，不能提前导入尚不存在的模块。

**验收：**安装为可编辑包后，能够导入 `product_search`。这时不需要模型、数据库或 HTTP 服务。

## 2. 先定义“项目处理的是什么”

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 4 | `src/product_search/schemas.py` | Query、Product、Judgment、SearchResult 四种业务对象 | 无项目内依赖 → 数据、检索、评估各层 |
| 5 | `src/product_search/validation.py` | 字段类型、空值、市场、标签、rank、score 校验 | schemas → 数据转换与各层入口 |
| 6 | `src/product_search/contracts.py` | QueryKey、ProductKey、单路 Hit、融合 Candidate | schemas + validation → BM25、向量、融合、排序 |

同时写 `docs/data-contract.md`，明确 E=Exact、S=Substitute、C=Complement、I=Irrelevant，标签属于 query-product 关系；未标注不是 I。

**验收：**可以手工创建一个查询、两个商品及不同的标签；能说明 Product 是商品事实，Judgment 是查询与商品的关系，Hit 是某路召回的结果。

## 3. 把可调参数集中管理

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 7 | `configs/dev.toml` | 小规模数据、文本策略、模型、召回、训练、评估、服务参数 | 先明确配置结构 → settings |
| 8 | `src/product_search/settings.py` | 配置对象、TOML 读取、类型和跨字段校验、路径解析 | dev.toml → 后续所有编排模块 |
| 9 | `configs/full.toml` | 完整 US 数据的配置 | 沿用相同结构 → 全量实验，先不运行 |

按最终文件阅读时可以看到所有参数；从零实现时，随模块落地逐步补齐参数和校验，不需要提前写其内部算法。

**验收：**dev 配置读成 Settings；拼错字段、非法值会报清晰错误；相对路径统一以项目根目录为基准。

## 4. 把原始数据变成可使用的对象

先创建 `src/product_search/data/__init__.py`。

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 10 | `src/product_search/data/download.py` | 固定来源版本，下载 parquet，校验大小和 SHA256 | 标准库 → 原始文件及下载清单 |
| 11 | `src/product_search/data/read_parquet.py` | 分批读行，过滤市场和官方 split | PyArrow + 原始 parquet → prepare；适合串联 normalize 验证 |
| 12 | `src/product_search/data/normalize.py` | 清洗 ID、空值与空白；一行转 Query/Product/Judgment | schemas + validation → build_dataset、prepare |
| 13 | `src/product_search/data/build_dataset.py` | 定义 Dataset，小规模组装和重复冲突检查 | normalize + schemas → split、audit |
| 14 | `src/product_search/data/audit.py` | 检查内存 Dataset 的关系与划分情况 | build_dataset → 小样本验证 |
| 15 | `src/product_search/data/split.py` | 按归一化查询文本组划分 train/validation | build_dataset → prepare 的分区规划 |

`build_dataset.py`、`audit.py` 适合先用小样本理解数据；正式 prepare 会流式处理，不是把完整 parquet 一次装进内存。`split.py` 防止大小写/空白归一化后的同文本跨分区，不负责识别语义同义词。

**验收：**能从少量原始行得到合法对象；相同查询文本组不会同时进入训练和验证分区。已有下载文件先校验，不必重复下载。

## 5. 把数据可靠地落到数据库

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 16 | `src/product_search/data/store.py` | 表结构、事务、只读连接、插入冲突处理、按键读取 | contracts + schemas + validation → prepare、训练、搜索 |
| 17 | `src/product_search/data/prepare.py` | 校验来源，规划分区，抽样，流式写库，成功后发布 | settings + 前面数据组件 → 正式数据集目录 |
| 18 | `src/product_search/data/audit_store.py` | 核查 manifest、哈希、外键、孤立引用、分区与标签统计 | store + 已发布数据 → 建索引和评估的前置检查 |

**这一阶段生成：**`catalog.sqlite` 保存商品；`labels.sqlite` 保存查询与人工标签；`manifest.json` 保存来源、数据身份与哈希；`rejects.jsonl` 保存拒绝记录。

**验收：**prepare 成功且正式审计通过。后续主要读取这两个数据库，不反复解析 parquet；线上只能使用 catalog。

## 6. 定义产物如何辨认和验收

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 19 | `src/product_search/artifacts.py` | JSON 读写、SHA256、配置语义签名、release 校验、数据验收 | settings；数据验收函数调用 audit_store → 索引、模型、服务 |

配套 `tests/test_artifacts.py`：检查损坏产物被拒绝、历史文件不能被静默覆盖。

**验收：**改动产物内容会导致哈希检查失败。此文件放在数据审计之后，是因为它的完整数据验收功能确实依赖 audit_store。

## 7. 先实现关键词召回

先创建 `src/product_search/retrieval/__init__.py`。

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 20 | `src/product_search/retrieval/text.py` | 统一清洗、分词、截断和商品字段拼接 | schemas → BM25、编码器、排序特征 |
| 21 | `src/product_search/retrieval/bm25.py` | 构建倒排、保存/加载、全库 top-k、指定候选补分 | text + contracts + validation → 索引编排、搜索、训练 |

配套 `tests/test_text.py`、`tests/test_bm25.py`：验证 Unicode、公式、同分排序、大候选、损坏索引和生命周期。

**验收：**只用几件商品便能检索；无词命中与“商品不存在”能明确区分。此时不需要下载 MiniLM。

## 8. 再实现向量召回与融合

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 22 | `src/product_search/retrieval/embedder.py` | 固定 revision 加载 MiniLM，分别编码查询和商品 | settings + text + schemas → indexing、训练、搜索 |
| 23 | `src/product_search/retrieval/vector_index.py` | 保存矩阵与商品键，校验向量，分块精确搜索和补分 | artifacts + contracts + NumPy → indexing、训练、搜索 |
| 24 | `src/product_search/retrieval/fusion.py` | 按排名做 RRF，保留各路 Hit，得到 Candidate | contracts → 搜索和评估 |
| 25 | `src/product_search/indexing.py` | 串联 BM25 构建、分批商品编码、向量保存和发布校验 | 前面数据/召回模块 + artifacts → 完整索引发布包 |

配套 `tests/test_retrieval.py`：先用手工向量验证搜索和 RRF，再让真实编码器接入；下载模型发生在这一阶段。

**这一阶段生成：**`bm25.sqlite`、`embeddings.npy`、`embedding_ids.jsonl`、`vector_manifest.json`、`release.json`，并复制商品库。当前构建代码还保存 `dataset_manifest.json`；早期已生成发布包可能没有这个快照。

**验收：**同一查询可以得到 BM25、dense 和 RRF 结果；保存并重载后可用；查询编码器与商品向量版本一致。

## 9. 在训练之前写清楚“怎么算好”

先创建 `src/product_search/evaluation/__init__.py`。

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 26 | `src/product_search/evaluation/metrics.py` | NDCG、MRR_E、known recall、judged fraction，空分母规则 | schemas + validation → evaluate |
| 27 | `src/product_search/evaluation/baselines.py` | 对给定候选补算 BM25 并稳定排序，保留零分 | BM25 + contracts → 单独的基线实验；当前 evaluate 自行编排多路候选打分，未直接调用它 |
| 28 | `src/product_search/evaluation/compare.py` | 同一查询配对 bootstrap、差值、置信区间 | 标准库 → evaluate |
| 29 | `src/product_search/evaluation/error_analysis.py` | 从标签和排名提取失败类别及证据 | 标准库 → evaluate |
| 30 | `src/product_search/evaluation/benchmark.py` | 汇总阶段耗时的 p50/p95/p99 | 标准库 → evaluate、CLI benchmark |

配套 `tests/test_metrics.py`、`test_compare.py`、`test_error_analysis.py`、`test_benchmark.py`，同步写 `docs/evaluation.md`。

**验收：**手算小例子的指标与程序一致；明确“人工候选内排序”和“从商品库召回”是两种口径。这时先完成指标组件，完整评估编排等搜索流水线完成后再写。

## 10. 构造特征，训练和加载排序模型

创建 `src/product_search/ranking/__init__.py`；等 features 存在后，再补其共享特征导出。

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 31 | `src/product_search/ranking/features.py` | 固定特征顺序，构造 BM25/dense、文本长度、词项重合、品牌/颜色匹配 | text + contracts + settings → dataset、predict |
| 32 | `src/product_search/ranking/dataset.py` | 读取 train/validation 人工候选，补齐双路分数，组织 X/y/group | store + BM25 + vector_index + features + artifacts；接收编码器 → train |
| 33 | `src/product_search/ranking/train.py` | 按查询分组训练 LambdaRank，validation 早停，保存新发布包 | dataset + features + Embedder + artifacts → ranker 模型 |
| 34 | `src/product_search/ranking/predict.py` | 核对模型/特征身份，构造相同特征并稳定重排 | features + artifacts + 模型 → SearchEngine |

配套 `tests/test_ranking.py`：验证 group 对齐、测试集隔离、特征消融一致性、训练/保存/预测往返。

**这一阶段生成：**`ranker.txt`、`model_manifest.json` 和新的 `release.json`。训练产生新目录，不覆盖原索引发布包。

**验收：**新模型可以重新加载并排序；训练和预测调用同一个 features；ID 和人工标签不进入在线特征。

## 11. 把算法接成一次真正的搜索

先创建 `src/product_search/service/__init__.py`。

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 35 | `src/product_search/service/observability.py` | 耗时计时器、请求日志，日志不记录查询原文 | 标准库 → search、api |
| 36 | `src/product_search/service/search.py` | 加载发布包；查询编码 → BM25+dense → RRF → 取商品 → 重排 → 返回 | store + 召回模块 + Ranker + observability → evaluate、HTTP、性能测试 |

**验收：**直接调用 SearchEngine 即可搜索，无需 HTTP。线上只读 catalog；未加载排序器时明确返回 `rrf_fallback`。

这里先写 search，是因为后面的 evaluate 会复用它加载和校验检索资源；先有可调用流水线，再包 HTTP 接口。

## 12. 完成离线评估和测试集冻结

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 37 | `src/product_search/evaluation/freeze.py` | 核对同一 release 的 validation 报告，保存冻结凭据，限制一次 test 消费 | artifacts + settings → evaluate、CLI freeze |
| 38 | `src/product_search/evaluation/evaluate.py` | 遍历查询，分别评估人工候选排序与商品库召回，汇总比较和错误案例 | SearchEngine + store + 指标/比较/错误分析/延迟组件 + freeze → 报告 |

**这一阶段生成：**`ranking-results.json`、`ranking-per-query.jsonl`、`retrieval-results.jsonl`、`error-analysis.jsonl`、`retrieval-timings.json`。freeze 代码可以先实现，但实际冻结一定在 validation 完成、模型选定之后。

**验收：**各方法对同一批查询比较，标签未知与 I 分开处理；报告同时有平均指标和逐查询证据。先运行 validation，不因代码已写完就自动消费 test。

## 13. 最后增加 HTTP 与页面

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 39 | `src/product_search/service/api.py` | 请求校验、资源生命周期、并发限制、health/version/search | SearchEngine + observability → HTTP 客户端 |
| 40 | `src/product_search/service/static/index.html` | 查询框、结果、召回来源、耗时和错误提示 | 调 `/search`、`/version` → 用户操作界面 |

配套 `tests/test_api.py`：先验证请求、异常隐藏、资源关闭，再用真实模型验证响应。此时补齐 pyproject 的静态 HTML 打包配置。

**验收：**输入商品需求能在页面看到结果；业务算法仍在 SearchEngine 中，页面和 API 不复制检索逻辑。

## 14. 把各步骤接到统一运行入口

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 41 | `src/product_search/cli.py` | 参数解析，编排 prepare/audit/build-indexes/train-ranker/evaluate/freeze/benchmark/serve | 前面已完成的组件 → 用户命令 |
| 42 | `src/product_search/__main__.py` | 支持 `python -m product_search`，传递退出码 | cli → Python 模块运行 |
| 43 | `main.py` | PyCharm 可选启动壳 | __main__ → IDE 运行 |
| 44 | `scripts/prepare_data.py` | 把脚本参数转给 CLI prepare | cli → 数据准备快捷入口 |
| 45 | `scripts/build_indexes.py` | 转给 CLI build-indexes | cli → 索引快捷入口 |
| 46 | `scripts/train_ranker.py` | 转给 CLI train-ranker | cli → 训练快捷入口 |
| 47 | `scripts/run_evaluation.py` | 转给 CLI evaluate | cli → 评估快捷入口 |

配套 `tests/test_workflow.py`：命令入口、冻结凭据和没有 LTR 时的评估边界。此时补齐 pyproject 的 console 入口。

**验收：**脚本与模块命令有相同的参数、路径和行为；按 README 可以完成准备 → 索引 → 训练 → validation → serve。

从零做时，CLI 可以在第 5 阶段先实现 prepare/audit，再随阶段增加子命令。这里把它的**最终完整文件**放在最后，是为了避免先写一个导入大量尚不存在文件的入口。

## 15. 主流程跑通后再做可选性能优化

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 48 | `src/product_search/retrieval/ann_index.py` | FAISS HNSW 构建、搜索、保存/加载、哈希校验 | artifacts + vector_index 的检查函数 + FAISS → 独立近似检索实验 |

配套 `tests/test_ann.py`。默认 SearchEngine 仍是分块精确搜索，**并未调用 ann_index**。

**验收：**先验证合成向量保存和重载；再在相同查询与向量上比较近似检索的邻居召回、延迟、内存。完成对比后才决定是否替换默认实现，不能把“安装了 FAISS”当作优化完成。

## 16. 收尾与学习资料：随着阶段维护，不集中补作业

| 文件 | 何时维护 | 用途与关系 |
|---|---|---|
| `README.md` | 第 1 阶段写目标；入口完成后补命令 | 用户从这里启动和复现 |
| `docs/USAGE.md` | 主流程可运行后 | 集中说明启动、搜索、复现实验、读报告和排障 |
| `requirements-lock.txt` | 环境验证通过后生成 | 固定实测依赖版本，不包含模型权重 |
| `docs/data-contract.md` | 第 2 阶段开始 | 记录身份、标签和数据泄漏边界 |
| `docs/evaluation.md` | 第 9 阶段开始 | 记录指标、冻结、性能口径 |
| `docs/troubleshooting.md` | 每次复现并解决故障后 | 保存原因与排查方法 |
| `docs/interview.md` | 每完成一项设计后 | 记录设计取舍与答法 |
| `docs/VERIFICATION-20260926.md` | 验证完成后 | 记录真实结果和未验证范围 |
| `docs/FILE-GUIDE.md` | 随结构调整更新 | 本文件，给出构建次序和依赖 |
| `docs/PROJECT-ROADMAP.md` | 设计阶段 | 原路线记录，实际运行以 README 为准 |
| `docs/plans/2026-09-26-complete-project.md` | 本次补齐实现前 | 此次执行计划与约束 |
| `PROJECT-GUIDE.html` | 历史学习材料 | 不是 FastAPI 搜索页面，也不是最新运行入口 |

测试文件应在对应阶段写并运行，而不是等最后才写；最终再跑全量测试、真实 validation、接口与性能验证。`.venv` 是依赖环境，`.cache/models` 是权重缓存，`.backups` 是原文件副本，均不属于业务实现。

## 构建顺序与运行顺序的区别

**构建时先写底层，再接编排：**schemas/validation → 数据与索引 → 模型与 SearchEngine → evaluate/api → CLI。

**使用时从入口进入：**CLI → 选定的流程。例如 serve → api → SearchEngine → BM25/dense → RRF → Ranker → 结果；离线 evaluate 还会打开 labels 计算指标。

因此不要从 cli.py 的第一行一路往下读所有源码。先按本文把每个组件的输入输出弄清，再回到入口看它们如何被串起来。

## 17. 把学习流程接成点击式面板

这些文件在核心算法完成后加入，负责把已经存在的 CLI 和产物变成可按步骤操作的学习入口；它们不重新实现召回、排序或评估算法。

| 顺序 | 文件 | 写什么 | 依赖 → 交给谁 |
|---|---|---|---|
| 49 | `src/product_search/workbench.py` | 扫描有效数据集、发布包和报告；为每轮实验生成唯一目录；启动后台进程并把日志交给界面 | `pathlib`、`subprocess`、CLI 产物 → `desktop.py` |
| 50 | `src/product_search/desktop.py` | Tkinter 四页面板：体验搜索、结果解读、重做实验、学习路线；后台加载服务和实验，主线程更新控件 | `workbench.py`、`settings.py`、CLI → 用户操作 |
| 51 | `打开项目.cmd` | 从项目根目录双击启动 `product_search.desktop` | `.venv/Scripts/pythonw.exe` → `desktop.py` |
| 52 | `tests/test_workbench.py` | 验证资源扫描、实验前置条件、进程停止、页面切换和重复输出保护 | `workbench.py`、`desktop.py` → 回归检查 |
| 53 | `docs/COMMANDS.md` | 保存 HTTP、环境恢复、test、benchmark 和终端复现命令 | CLI、配置、产物目录 → 进阶学习 |

面板的构建顺序是：先让 `workbench.py` 能安全规划实验并管理自己创建的进程，再让 `desktop.py` 调用它，最后提供双击入口。使用顺序则是：体验搜索 → 阅读报告 → 重做 validation → 回到源码。两种顺序不要混淆。

面板不会自动运行完整目录、test 或 GPU 实验；这些操作仍需阅读 `docs/COMMANDS.md`，并在模型选择完成后单独确认。
