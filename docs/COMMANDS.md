# 命令参考：进阶与复现附录

本指南面向当前 Windows 笔记本。项目已有可用数据、MiniLM 缓存、索引和排序模型，第一次使用直接启动即可，不必先下载数据或训练。默认 CPU，不需要调用 Dify、云服务器或聊天模型 API。

项目做的是“输入商品需求，返回商品排序”，不是聊天问答；当前支持 US 市场，优先使用英文查询。

## 1. 打开终端，进入项目

打开 PowerShell 或 PyCharm 的 Terminal，执行：

```powershell
$projectRoot = (Get-Location).Path
Set-Location $projectRoot
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
```

后续命令在同一个终端执行；新开终端需要再次执行这三行。这里直接指定项目 Python，不需要激活虚拟环境。

**不要把这些命令粘贴进 Python 的 `>>>` 窗口。** 如果当前看到 `>>>`，先输入 `exit()` 返回终端。

当前已有资源：

| 资源 | 相对项目根目录的位置 | 用途 |
|---|---|---|
| 配置 | `configs/dev.toml` | 所有命令的默认配置 |
| 数据集 | `data/processed/dev-c882caf5632ad442` | 11,934 件商品及分区标签 |
| 召回发布包 | `artifacts/dev-indexes-20260926` | BM25、向量、商品库；没有排序模型 |
| 完整搜索发布包 | `artifacts/dev-ltr-20260926` | 召回组件加 LambdaRank |
| 验证集报告 | `reports/validation-20260926` | 已有 200 个 validation 查询的评估 |
| 性能报告 | `reports/benchmark-20260926.json` | 预热后的本地调用耗时 |

## 2. 直接启动现有项目

```powershell
& $pythonExe -m product_search serve --release artifacts/dev-ltr-20260926
```

服务先校验发布文件，再加载本地模型。看到 `Application startup complete` 和监听地址后，用浏览器打开：

- 搜索页面：<http://127.0.0.1:8000/>
- 接口文档：<http://127.0.0.1:8000/docs>
- 存活检查：<http://127.0.0.1:8000/health>
- 发布版本：<http://127.0.0.1:8000/version>

终端一直占用是正常的，说明服务在运行。使用期间保留窗口；结束时在该终端按 **Ctrl+C**。再次使用只需重新启动，不必重建索引。

此页面属于商品搜索项目，端口为 **8000**；此前课程学习站使用的 **8770** 是另一个服务。

## 3. 实际搜索并看懂结果

先尝试 `running shoes`、`wireless headphones`、`coffee maker`、`phone case`。选择结果数量后点击“搜索”。小规模商品库不保证覆盖所有需求。

| 页面信息 | 怎么理解 |
|---|---|
| 排名和商品标题 | 系统认为更符合查询的商品排在前面，不代表绝对正确 |
| 品牌、商品 ID | 对应商品事实库中的记录 |
| 分数 | 当前排序方法的分数；不是购买概率，不同查询/方法的分数不能直接比较 |
| `bm25 #…` | 商品在关键词召回中的名次 |
| `dense #…` | 商品在语义召回中的名次；只显示一路时，表示另一条路没有把它召回进 top-k |
| 混合召回 + 学习排序 | 当前响应的 pipeline 为 `hybrid_ltr` |
| RRF 融合 | pipeline 为 `rrf_fallback`，没有加载排序模型 |
| 分阶段耗时 | 编码、关键词检索、向量检索、融合、排序所用时间；服务内部耗时不等于浏览器总等待时间 |
| 发布版本和数据版本 | 确认实际使用的模型、索引和数据集 |

想体验没有 LTR 的版本：先 Ctrl+C 停止当前服务，再执行：

```powershell
& $pythonExe -m product_search serve --release artifacts/dev-indexes-20260926
```

这是 RRF 与 LTR 的体验对比，不是纯 dense 与 LTR 的对比。正式效果比较以同一验证集报告为准。

## 4. 直接调用 HTTP 接口

保持服务终端运行，**另外打开一个 PowerShell 窗口**执行：

```powershell
$body = @{ query = 'running shoes'; locale = 'us'; top_k = 5 } | ConvertTo-Json
$response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/search' -Method Post -ContentType 'application/json' -Body $body
$response | ConvertTo-Json -Depth 8
```

请求参数：

| 参数 | 要求 |
|---|---|
| `query` | 必填，非空字符串，最多 256 个字符 |
| `locale` | 当前只能是 `us`，省略时默认 `us` |
| `top_k` | 1–50 的整数，省略时默认 10；页面只提供 5/10/20 的选项 |

主要响应字段：`results` 是排名结果；`pipeline` 是搜索策略；`release_id` 是发布身份；`timings_ms` 是阶段耗时；`request_id` 用于关联服务日志。

## 5. 先查看已有报告，不必重新训练

回到第 1 节设置过变量的终端。如果它正在运行服务，先 Ctrl+C，或新开终端并重新执行第 1 节。

```powershell
$reportFile = Join-Path $projectRoot 'reports/validation-20260926/ranking-results.json'
$report = Get-Content -LiteralPath $reportFile -Raw -Encoding UTF8 | ConvertFrom-Json
$report.ranking | ConvertTo-Json -Depth 6
$report.retrieval | ConvertTo-Json -Depth 6
$report.comparisons | ConvertTo-Json -Depth 6
```

这个报告读取块使用完整路径，可以在新 PowerShell 窗口独立运行。若其他命令提示 `$projectRoot` 为空，说明当前窗口还没有执行第 1 节的变量初始化；激活 `.venv` 或进入项目目录不会自动创建这个变量。

| 报告内容 | 回答的问题 |
|---|---|
| `ranking` 中的 NDCG/MRR | 同一批人工候选商品，谁排得更好？ |
| `retrieval` 中的 known recall | 从整个 dev 商品库里，多少已知相关商品被找回？ |
| `judged_fraction` | 返回商品中有多少存在这个查询的人工标签？没有标签不代表不相关 |
| `comparisons` | 相对 BM25 的平均差值、置信区间、胜负查询数是多少？ |
| `ranking-per-query.jsonl` | 每个查询具体排了什么、指标多少？ |
| `retrieval-results.jsonl` | 每个查询从商品库召回了什么？ |
| `error-analysis.jsonl` | 哪些查询出现指标退化、I 排在 E 前面等有证据的问题？ |

已有报告的人工候选 NDCG@10：BM25 0.7682、dense 0.7829、RRF 0.7777、LTR 0.7735。当前 LTR 没有超过 dense，相对 BM25 也没有显著提升证据。validation 同时用于早停，不能把这些结果当作独立测试集或线上效果。

## 6. 亲手重跑一轮实验

仅想使用搜索时无需执行本节。重跑时先停止本地服务，便于观察资源和耗时。

**以下步骤按顺序逐条执行，每条成功后再进入下一条。** 在已有数据上建新索引即可，不必重新 prepare。

### 6.1 测试环境和审计数据

```powershell
& $pythonExe -m pytest -q
& $pythonExe -m product_search audit --dataset data/processed/dev-c882caf5632ad442
```

验收：测试无失败；审计返回 `passed`。审计会在 reports 下创建新的报告目录。

### 6.2 给本轮实验创建唯一名称

```powershell
$runId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$datasetDir = Join-Path $projectRoot 'data/processed/dev-c882caf5632ad442'
$indexDir = Join-Path $projectRoot ('artifacts/indexes-' + $runId)
$modelDir = Join-Path $projectRoot ('artifacts/ltr-' + $runId)
$evaluationDir = Join-Path $projectRoot ('reports/validation-' + $runId)
$benchmarkFile = Join-Path $projectRoot ('reports/benchmark-' + $runId + '.json')
```

不要预先创建这些输出目录。构建程序会创建它们，并拒绝覆盖历史实验。中途失败时先看错误；重跑可换新 `$runId`，保留失败目录用于检查。

### 6.3 建立召回索引

```powershell
& $pythonExe -m product_search build-indexes --dataset $datasetDir --output $indexDir
```

执行：商品库 → BM25 倒排；商品文本 → 本地 MiniLM → 向量文件；最后发布清单。

验收：命令成功返回 `release_dir`；目录内有 `bm25.sqlite`、`embeddings.npy` 和 `release.json`。这是索引包，还没有 LTR 模型。

### 6.4 训练排序模型

```powershell
& $pythonExe -m product_search train-ranker --dataset $datasetDir --release $indexDir --output $modelDir
```

执行：读取 train/validation 人工候选 → 补算双路分数 → 构造特征和查询 group → LambdaRank 训练及早停 → 新模型包。

验收：新目录有 `ranker.txt`、`model_manifest.json`、`release.json`。MiniLM 在这里用于编码查询，不做微调；训练的是 LightGBM 排序器。

### 6.5 评估本轮模型

```powershell
& $pythonExe -m product_search evaluate --dataset $datasetDir --release $modelDir --output $evaluationDir --partition validation
```

验收：评估查询数符合数据集；生成排序、召回、逐查询和错误报告。按第 5 节方式读取新目录中的 `ranking-results.json`。此步骤没有运行 test。

### 6.6 测量速度，启动本轮模型

```powershell
& $pythonExe -m product_search benchmark --release $modelDir --output $benchmarkFile --runs 30
& $pythonExe -m product_search serve --release $modelDir
```

benchmark 先预热，再报告 p50/p95/p99。它是本地串行 SearchEngine 耗时，不含 HTTP，也不是 QPS 压测。启动后刷新页面，用 `/version` 确认 release 已切换。

## 7. 修改项目时，需要重做哪一步

| 修改内容 | 需要做什么 |
|---|---|
| 只换搜索关键词或返回数量 | 直接搜索 |
| 只改页面 | 刷新页面；必要时强制刷新缓存 |
| 改服务端口 | 重启服务，浏览器使用新端口 |
| 改 LambdaRank 的树数、叶子数或学习率 | 用已有索引重新训练到新目录，再评估；已有模型不会自动改变 |
| 做特征删除消融 | 训练时使用 `--drop-feature`，新模型、新报告 |
| 改模型 revision、文本策略、BM25 语义参数 | 新建索引、重新训练、重新评估 |
| 改数据分区、抽样或市场数据 | 重新 prepare，再完整执行索引、训练和评估 |

当前兼容性校验偏严格：召回 top-k、RRF 常数等也在 runtime signature 中；直接改配置并加载旧 release 会被拒绝。请让一轮实验始终使用同一份匹配配置，不要手改 manifest 绕过检查。`settings.py` 只接受已实现的文本策略，不是任意写一个版本名都能生效。

例如在第 6 节生成的索引上删除 dense 特征，保持其他条件相同：

```powershell
$ablationDir = Join-Path $projectRoot ('artifacts/ltr-no-dense-' + $runId)
$ablationReportDir = Join-Path $projectRoot ('reports/validation-no-dense-' + $runId)
& $pythonExe -m product_search train-ranker --dataset $datasetDir --release $indexDir --output $ablationDir --drop-feature dense_score
& $pythonExe -m product_search evaluate --dataset $datasetDir --release $ablationDir --output $ablationReportDir --partition validation
```

这是**排序特征消融**：仅从 LTR 特征中删除 dense_score；向量召回本身仍然存在，不应解释为“关闭整个 dense 链路”。

## 8. test：只在模型选择结束后运行

日常学习、调参使用 validation。以下命令是最终验收步骤，本次已有报告尚未消费真实 test。前提是第 6 节实验成功完成、配置和模型已选定。

```powershell
$validationReport = Join-Path $evaluationDir 'ranking-results.json'
$testReportDir = Join-Path $projectRoot ('reports/test-' + $runId)
& $pythonExe -m product_search freeze --release $modelDir --validation-report $validationReport
& $pythonExe -m product_search evaluate --dataset $datasetDir --release $modelDir --output $testReportDir --partition test --allow-test
```

freeze 核对模型、数据、验证报告和评估截断；test 会写一次性使用凭据。同一版本再次消费 test 会被拒绝，失败时也保留凭据用于排障，不自动清除。

## 9. 环境恢复与 PyCharm 设置

**当前环境已经安装，正常使用跳过恢复。** 若在新目录/机器建立环境，先复制源码、配置、必要的数据和发布包，再在目标根目录执行：

```powershell
py -3.12 -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements-lock.txt
& .venv/Scripts/python.exe -m pip install -e . --no-deps
```

没有 `py` 启动器时，用该机器实际的 Python 3.12 可执行文件替代；不要覆盖一个已有且正常使用的环境。

服务采用 `local_files_only=True`，不会在启动时自动下载权重。新环境如果缺少 `.cache/models`，先在项目根目录执行一次：

```powershell
& .venv/Scripts/python.exe -c "from product_search.settings import load_settings; from product_search.retrieval.embedder import Embedder; Embedder(load_settings('configs/dev.toml', project_root='.'))"
```

首次下载需要联网；缓存就绪后再启动服务。数据/模型产物不是 pip 安装包的一部分；`.gitignore` 排除了大文件，所以只复制源码不等于完整迁移。

若迁移时连原始数据也缺失，可在原始目标文件不存在的情况下使用现有下载函数，再 prepare：

```powershell
& .venv/Scripts/python.exe -c "from pathlib import Path; from product_search.data.download import download_file; download_file('examples', Path('data/raw')); download_file('products', Path('data/raw'))"
& .venv/Scripts/python.exe -m product_search prepare
```

下载和 prepare 均拒绝覆盖已有目标。prepare 成功后，将实际输出的数据目录传给第 6 节命令，不依赖写死的旧数据目录名。

PyCharm 配置：

| 设置 | 当前机器填写值 |
|---|---|
| Interpreter | `.venv/Scripts/python.exe` |
| Working directory | `.` |
| Module name | `product_search` |
| Parameters | `serve --release artifacts/dev-ltr-20260926` |

也可运行根目录的 `main.py`，但仍须填写同样的 Parameters。直接运行 main.py 且不带子命令，会显示缺少 command。

## 10. 常见现象与处理

| 现象 | 处理 |
|---|---|
| `No module named product_search` | 确认使用项目 `.venv`；在项目根目录运行 `pip install -e . --no-deps` |
| PowerShell 命令在 `>>>` 报错 | `exit()` 回到终端，不在 Python 交互解释器里执行 |
| 找不到配置、数据库或 release | 确认工作目录；当前目录不便切换时，子命令加 `--project-root` 指向项目根目录 |
| 启动时报模型文件不存在 | 按第 9 节初始化编码器，下载当前固定版本到本项目缓存 |
| 8000 端口被占用 | 优先找到自己之前启动的服务并正常停止；或更改本项目 service.port 后重启，不要随意结束未知进程 |
| `SHA256` / signature 不匹配 | 使用匹配配置或新建发布包，不手工篡改哈希 |
| `FileExistsError` | 换新实验名；这是保护历史结果，不是让你删除已成功产物 |
| HTTP 422 | 检查 query 是否为空/超长、locale 是否 us、top_k 是否整数且在 1–50 |
| HTTP 503 | 单工作线程执行搜索，执行与排队请求合计上限为 2；稍后重试，不等于两路搜索并行 |
| HTTP 500 | 保存 request_id，在服务终端查对应异常；接口不会向外暴露内部堆栈 |
| 英文查询结果也不理想 | 先检查商品库是否覆盖需求，再区分没召回与排错；使用已有错误报告定位 |
| 仅出现上游弃用提示 | 当前 TestClient/httpx 的提示不等于测试失败，以退出码及测试结果为准 |

建议先完成：启动 → 搜索 3 个不同需求 → 查看两路命中 → 查看 validation → 重跑一轮 → 做一个特征消融。读源码时再按 `docs/FILE-GUIDE.md` 的构建顺序推进。
