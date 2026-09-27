"""下载固定版本的 ESCI 文件，并记录来源与完整性校验结果。"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

REVISION = "7916cdf6ab75a462e77f20ab40428a10923998d5"
REPOSITORY = "https://github.com/amazon-science/esci-data"

ASSETS = {
    "examples": {
        "filename": "shopping_queries_dataset_examples.parquet",
        "size": 51286808,
        "sha256": "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    },
    "products": {
        "filename": "shopping_queries_dataset_products.parquet",
        "size": 1108857465,
        "sha256": "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265",
    },
}

CHUNK_SIZE = 1024 * 1024


def download_file(name: str, output_dir: Path) -> dict[str, object]:
    """下载一个文件；已有同名结果时拒绝覆盖。"""

    if name not in ASSETS:
        raise ValueError("name 必须是 examples 或 products")

    asset = ASSETS[name]
    filename = asset["filename"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    destination = output_dir / filename
    manifest_path = output_dir / f"{filename}.manifest.json"

    if destination.exists() or manifest_path.exists():
        raise FileExistsError(f"目标文件或来源记录已存在：{destination}")

    url = (
        "https://media.githubusercontent.com/media/"
        f"amazon-science/esci-data/{REVISION}/"
        f"shopping_queries_dataset/{filename}"
    )
    request = Request(url, headers={"User-Agent": "product-search-learning"})
    digest = hashlib.sha256()
    size_bytes = 0

    with TemporaryDirectory(prefix="esci-", dir=output_dir) as temp_dir:
        temporary_path = Path(temp_dir) / filename

        with urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise ValueError(f"下载响应不是完整文件：HTTP {response.status}")

            resolved_url = response.geturl()

            with temporary_path.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    output.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)

                    if size_bytes > asset["size"]:
                        raise ValueError("下载内容超过预期大小")

        actual_sha256 = digest.hexdigest()

        if size_bytes != asset["size"]:
            raise ValueError("文件大小不匹配，下载可能不完整")

        if actual_sha256 != asset["sha256"]:
            raise ValueError("SHA256 不匹配，下载内容与固定版本不一致")

        manifest = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "source_url": url,
            "resolved_url": resolved_url,
            "filename": filename,
            "size_bytes": size_bytes,
            "sha256": actual_sha256,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        temporary_manifest = Path(temp_dir) / manifest_path.name
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        temporary_path.rename(destination)
        temporary_manifest.rename(manifest_path)

    return manifest