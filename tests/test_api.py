import pytest
from fastapi.testclient import TestClient
from product_search.settings import load_settings
from product_search.service.api import create_app


class FakeEngine:
    def __init__(self):
        self.closed = False
        self.manifest = {'release_id': 'test-release', 'dataset_id': 'd', 'kind': 'indexes'}

    def search(self, text, *, top_k=10, locale='us'):
        if text == 'crash':
            raise RuntimeError('SECRET_PATH')
        return {'query': text, 'pipeline': 'rrf_fallback', 'release_id': 'test-release', 'results': [], 'timings_ms': {}}

    def close(self):
        self.closed = True


def test_api_validation_lifespan_and_sanitized_errors():
    from pathlib import Path
    settings = load_settings('configs/dev.toml', project_root=Path(__file__).resolve().parents[1])
    engine = FakeEngine()
    app = create_app(settings, 'unused', engine_factory=lambda: engine)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get('/health').json()['status'] == 'ok'
        assert client.get('/').status_code == 200
        assert client.get('/version').json()['release_id'] == 'test-release'
        for payload in ({'query': ' '}, {'query': 'a', 'locale': 'jp'}, {'query': 'a', 'top_k': 51},
                        {'query': 'a', 'top_k': True}, {'query': 'a', 'unknown': 1}):
            assert client.post('/search', json=payload).status_code == 422
        response = client.post('/search', json={'query': 'running shoes'})
        assert response.status_code == 200
        assert response.json()['request_id']
        assert response.json()['pipeline'] == 'rrf_fallback'
        error = client.post('/search', json={'query': 'crash'})
        assert error.status_code == 500
        assert 'SECRET_PATH' not in error.text
    assert engine.closed
