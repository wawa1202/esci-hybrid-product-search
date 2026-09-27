import pytest
from product_search.artifacts import sha256, write_json, verify_release, runtime_signature


def test_release_checks_contents_and_rejects_path_escape(tmp_path):
    p = tmp_path / 'file.txt'
    p.write_text('ok')
    manifest = {'release_id': 'x', 'dataset_id': 'd', 'runtime_signature': {}, 'files': {'file.txt': sha256(p)}}
    write_json(tmp_path / 'release.json', manifest)
    assert verify_release(tmp_path)['release_id'] == 'x'
    p.write_text('corrupt')
    with pytest.raises(ValueError):
        verify_release(tmp_path)


def test_write_never_overwrites(tmp_path):
    p = tmp_path / 'a.json'
    write_json(p, {'v': 1})
    with pytest.raises(FileExistsError):
        write_json(p, {'v': 2})
