# 本次实现与验证记录

实现范围：补齐本地商品搜索的 BM25+dense 召回、RRF、LambdaRank、评估、FastAPI、演示页面和 CLI；保留既有数据准备代码。原文件副本位于 `.backups/20260926-151349`，运行依赖仅安装到项目 `.venv`。此目录不是 Git 仓库，没有创建分支或提交。

## 已验证

- 133 项自动化测试全部通过；修正 SQLite 暂存清理测试的连接生命周期后，60 项 BM25 回归再通过。
- 源码和脚本通过 compileall；pip check 没有依赖冲突；可编辑包重新构建安装成功。
- 真实 MiniLM 固定 revision 在 CPU 加载，查询输出为 384 维。
- 对 11,934 件真实 dev 商品构建 BM25、向量索引；产物清单的 SHA256 验证通过。
- LambdaRank 使用 200 个 train 查询和 200 个 validation 查询，早停后成功保存、重新加载、预测。
- 对 200 个 validation 查询分别完成候选集排序与商品库召回评估，生成 30 条错误案例。
- 真实模型通过 FastAPI TestClient：health/version/页面可访问，`running shoes` 搜索返回 HTTP 200、`hybrid_ltr` 和 3 条结果。
- 5 次预热 + 30 次本地 SearchEngine 采样：总耗时 p50=29.7255ms，p95=40.7282ms，p99=41.7037ms。不含 HTTP 和并发。
- HNSW 可选适配通过合成向量保存、重载、排序、键错配和损坏检查。冻结 test 的机制只用合成数据验证。

## 实际效果

| 方法 | 人工候选 NDCG@10 | 人工候选 MRR_E@10 | 商品库 known E recall@100 |
|---|---:|---:|---:|
| BM25 | 0.7682 | 0.7389 | 0.8047 |
| dense | 0.7829 | 0.7798 | 0.8636 |
| RRF | 0.7777 | 0.7541 | 0.8750 |
| LTR | 0.7735 | 0.7406 | 不适用：LTR 不增加召回候选 |

LTR 相比 BM25 的平均 NDCG 差值为 +0.00531，95% 配对 bootstrap 区间为 [-0.01412, 0.02369]。没有显著提升证据，也没有超过 dense。validation 被用于早停，结果属于开发诊断。不要在简历中写成“上线提升”。

## 未验证或保留的范围

- 没有消费真实 test，没有跑完整 US 目录、GPU 路线、并发压测或线上 A/B。
- FAISS 尚未在真实 dev/full 数据上与精确搜索作召回/延迟对比，也没有成为默认线上检索器。
- 特征删除消融已有训练/预测接口和往返测试；尚未批量运行真实消融。title-only 文本消融尚未提供配置入口。
- 当前 FastAPI TestClient 使用 httpx 会打印上游弃用提示；功能验证通过，没有为消除提示而替换已验证依赖。
- 索引的分块矩阵节省了向量内存，但 Python 商品键映射和 BM25 命中集合仍会随全量数据增长；完整规模需另测资源。

## 查阅

`reports/validation-20260926/ranking-results.json` 是主要指标；同目录两个 per-query 文件保存排名证据，`error-analysis.jsonl` 保存错误案例；`reports/benchmark-20260926.json` 保存延迟原始汇总。逐文件职责见 `docs/FILE-GUIDE.md`。
