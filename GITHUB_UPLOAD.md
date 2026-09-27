# GitHub 上传检查清单

1. 在 GitHub 新建空仓库，建议名称为 `esci-hybrid-product-search`。
2. 使用仓库根目录的 `README.md` 作为首页说明。
3. 将 `GITHUB_DESCRIPTION.md` 中的 Repository About、Topics 和中文简介复制到 GitHub 仓库设置。
4. 首次提交前运行：

   ```powershell
   & .venv/Scripts/python.exe -m pytest -q
   ```

5. 确认没有 `.venv`、`.cache`、`.backups`、原始 Parquet、日志、API key 或个人路径被加入提交。
6. `data/processed`、`artifacts` 和 `reports` 中保留的是公开可复现实验快照；不要把新的个人数据或线上日志放入仓库。
7. 上传后从 GitHub 页面重新下载 ZIP，在干净目录按 README 安装依赖并验证 `pytest`，再把仓库链接写进简历。
8. 如果不希望把二进制快照提交到 Git，删除快照并保留 manifest，然后在 README 中改成“下载数据后重建”；当前复制版保留快照是为了让面试官可以直接查看结果。
