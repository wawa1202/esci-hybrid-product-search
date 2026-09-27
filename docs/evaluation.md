# 评估口径

**候选集排序**：BM25、dense、RRF、LTR 在相同人工候选上排序，计算 NDCG@10 和 MRR_E@10，回答“候选找到后排得好不好”。

**商品库召回**：从全 catalog 搜索，计算 known E recall、known E/S recall、judged fraction 和空结果，回答“相关商品有没有进候选”。unknown 不是 I。

RRF 是 `sum(1 / (60 + rank))`，不能把 BM25 和余弦原始分数直接相加。FAISS 的 ANN recall 是工程近似指标，应与相关性指标分开。

`reports/validation-20260926/` 是 dev 的 200 个 validation 查询和 11,934 件商品并集，适合学习和回归，不代表线上效果。

冻结与 test 命令如下；本次只验证机制，没有消费真实 test。请在设计和超参最终选定后执行：

```powershell
& .venv/Scripts/python.exe -m product_search freeze --release artifacts/my-ltr-v1 --validation-report reports/my-validation-v1/ranking-results.json
& .venv/Scripts/python.exe -m product_search evaluate --dataset data/processed/dev-c882caf5632ad442 --release artifacts/my-ltr-v1 --output reports/my-test-v1 --partition test --allow-test
```

冻结记录保存 release 的 SHA256、验证报告 SHA256 和指标截断；`.used.json` 是一次消费凭据。若 test 运行失败，凭据仍保留以供排障，不自动删除重试。此机制是本地实验纪律，不是不可绕过的访问控制；不能换个文件名重新挑测试结果。

`benchmark` 先预热 5 次再采样，输出编码、BM25、dense、融合、排序和总耗时的 p50/p95/p99。不包含 HTTP 排队、网络延迟，也不是并发 QPS 测试。

消融不应修改已有模型文件。例：训练时增加 `--drop-feature dense_score`，或删除 `overlap_count`、`overlap_ratio`、`bullet_overlap_count`、`bullet_overlap_ratio`，写入新 release 后用同一 validation 评估。模型清单会记录实际特征列。文本字段消融尚未提供配置开关，需要修改文本策略版本并重建索引；不能仅改报告名称。
