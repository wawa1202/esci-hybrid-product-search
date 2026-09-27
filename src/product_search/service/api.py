"""FastAPI 入口：校验输入、管理生命周期、限制排队，不放检索算法。"""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from product_search.service.observability import log_search


def create_app(settings, release_dir, *, engine_factory=None):
    if engine_factory is None:
        from product_search.service.search import SearchEngine
        engine_factory = partial(SearchEngine, settings, release_dir)

    @asynccontextmanager
    async def lifespan(app):
        # 单工作线程：SQLite 连接在同一线程创建、使用、关闭。
        # 信号量限制在执行与排队中的请求总数，避免无限队列。
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='search')
        loop = asyncio.get_running_loop()
        try:
            app.state.engine = await loop.run_in_executor(pool, engine_factory)
            app.state.pool = pool
            app.state.slots = asyncio.Semaphore(settings.service.max_concurrent_searches)
            yield
        finally:
            if hasattr(app.state, 'engine'):
                await loop.run_in_executor(pool, app.state.engine.close)
            pool.shutdown(wait=True)

    app = FastAPI(title='ESCI 商品搜索', version='0.1.0', lifespan=lifespan)

    class SearchRequest(BaseModel):
        model_config = ConfigDict(extra='forbid')
        query: str = Field(min_length=1, max_length=settings.service.max_query_chars, strict=True)
        locale: Literal['us'] = 'us'
        top_k: int = Field(default=10, ge=1, le=settings.service.max_top_k, strict=True)

        @field_validator('query')
        @classmethod
        def strip_query(cls, value):
            if not value.strip():
                raise ValueError('查询不能为空')
            return value.strip()

    @app.get('/')
    async def home():
        return FileResponse(Path(__file__).parent / 'static' / 'index.html')

    @app.get('/health')
    async def health():
        return {'status': 'ok'}

    @app.get('/version')
    async def version():
        manifest = app.state.engine.manifest
        result = {key: manifest.get(key) for key in ('release_id', 'dataset_id', 'kind', 'catalog_scope')}
        # 兼容早期发布清单的 kind；以实际已加载排序器判定，不误报为 indexes。
        if getattr(app.state.engine, 'ranker', None) is not None:
            result['kind'] = 'ranker'
        return result

    @app.post('/search')
    async def search(payload: SearchRequest):
        request_id = uuid.uuid4().hex
        slots = app.state.slots
        if slots.locked():
            raise HTTPException(503, {'message': '搜索繁忙，请稍后重试', 'request_id': request_id})
        await slots.acquire()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(app.state.pool, partial(app.state.engine.search,
                                      payload.query, top_k=payload.top_k, locale=payload.locale))
        # 客户端断开也不提前释放容量：后台任务完成后才允许下一次排队。
        future.add_done_callback(lambda _: slots.release())
        try:
            result = await asyncio.shield(future)
        except Exception:
            logging.getLogger('product_search').exception('search failed request_id=%s', request_id)
            raise HTTPException(500, {'message': '搜索暂时不可用', 'request_id': request_id}) from None
        result['request_id'] = request_id
        log_search(request_id, result)
        return result

    return app
