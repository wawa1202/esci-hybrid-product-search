# 排障顺序

- `No module named product_search`：回到项目根目录，用 `.venv/Scripts/python.exe`。
- 模型下载失败：先单独初始化 Embedder，检查网络和 `.cache/models`；revision 必须是完整 SHA。
- 发布哈希失败：不要手改 release 或索引，换新输出目录重建；历史发布不覆盖。
- dataset mismatch：dataset、catalog、BM25、向量必须来自同一次 prepare；换数据后重新 build-indexes。
- 接口 503：并发上限为 2；`/health` 是存活检查，`/version` 才能确认 release。
- 指标变差：先看 recall；相关商品没进候选时，改 LTR 特征不能补救召回。再查文本版本、模型 revision 和 unknown 口径。
