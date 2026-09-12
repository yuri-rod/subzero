import hashlib
import json
import os

import pytest

from subzero.caption_scan_cache import CaptionScanCache


def cache_at(root, video='video-one', recognition='vision-one'):
    return CaptionScanCache(root, video, recognition)


def test_completed_scan_preserves_blank_frames_text_and_input_timestamps(tmp_path):
    cache = cache_at(tmp_path)
    windows = [(1, 2), (3, 4)]
    detections = [(1.001, ''), (1.534, 'Keep this\nsecret.'), (3.002, 'Olá.')]
    assert cache.read(windows, 2) is None
    cache.write(windows, detections, 2)
    assert cache_at(tmp_path).read(windows, 2) == detections
    assert len(list(tmp_path.rglob('*.json'))) == 1


def test_empty_completed_scan_is_distinct_from_a_cache_miss(tmp_path):
    cache = cache_at(tmp_path)
    cache.write([(0, 1)], [], 10, retry_all=True)
    assert cache.read([(0, 1)], 10, retry_all=True) == []


def test_scan_identity_changes_cannot_reuse_other_observations(tmp_path):
    cache = cache_at(tmp_path)
    cache.write([(1, 2)], [(1.5, 'Yes.')], 2)
    assert cache.read([(1, 2.001)], 2) is None
    assert cache.read([(1, 2)], 10) is None
    assert cache.read([(1, 2)], 2, retry_all=True) is None
    assert cache_at(tmp_path, video='video-two').read([(1, 2)], 2) is None
    assert cache_at(tmp_path, recognition='vision-two').read([(1, 2)], 2) is None


@pytest.mark.parametrize('contents', [b'{broken', b'null', b'[]', b'\xff'])
def test_corrupt_cache_is_a_miss(tmp_path, contents):
    cache = cache_at(tmp_path)
    cache.write([(0, 1)], [(0.5, 'Yes.')], 2)
    path = next(tmp_path.rglob('*.json'))
    path.write_bytes(contents)
    assert cache.read([(0, 1)], 2) is None


def test_tampered_caption_text_fails_digest_validation(tmp_path):
    cache = cache_at(tmp_path)
    cache.write([(0, 1)], [(0.5, 'Yes.')], 2)
    path = next(tmp_path.rglob('*.json'))
    contents = json.loads(path.read_text())
    contents['detections'][0][1] = 'No.'
    path.write_text(json.dumps(contents))
    assert cache.read([(0, 1)], 2) is None


@pytest.mark.parametrize('detections', [
    [(True, 'Yes.')], [(float('nan'), '')], [(float('inf'), '')], [(10 ** 500, '')],
    [(-0.1, '')], [(1, '')], [(2.5, '')],
    [(0.5, ''), (0.4, '')], [(0.5, ''), (0.5, '')],
    [(0.5, None)], [(0.5, ['Yes.'])], [(0.5, 'a' * 20_000)],
    [(0.5, '\ud800')], [(0.5, '', 'unexpected')],
])
def test_invalid_observations_are_not_written(tmp_path, detections):
    cache = cache_at(tmp_path)
    with pytest.raises(ValueError):
        cache.write([(0, 1), (3, 4)], detections, 2)
    assert not list(tmp_path.rglob('*'))


@pytest.mark.parametrize('windows,fps,retry', [
    ([(1, 1)], 2, False), ([(2, 1)], 2, False),
    ([(-1, 1)], 2, False), ([(0, float('inf'))], 2, False),
    ([(0, True)], 2, False), ([(2, 3), (0, 1)], 2, False),
    ([(0, 2), (1, 3)], 2, False), ([(0, 1)], 0, False),
    ([(0, 1)], float('nan'), False), ([(0, 1)], True, False),
    ([(0, 1)], 2, 'false'),
])
def test_invalid_scan_parameters_fail_before_filesystem_access(tmp_path, windows, fps, retry):
    cache = cache_at(tmp_path)
    with pytest.raises(ValueError):
        cache.read(windows, fps, retry_all=retry)
    with pytest.raises(ValueError):
        cache.write(windows, [], fps, retry_all=retry)
    assert not list(tmp_path.rglob('*'))


def test_oversized_cache_is_a_miss(tmp_path):
    cache = cache_at(tmp_path)
    cache.write([(0, 1)], [], 2)
    path = next(tmp_path.rglob('*.json'))
    with path.open('wb') as handle:
        handle.truncate(6 * 1024 * 1024)
    assert cache.read([(0, 1)], 2) is None


def test_serialized_size_limit_rejects_large_scan_before_writing(tmp_path):
    cache = cache_at(tmp_path)
    detections = [(n / 1000, 'caption ' * 1500) for n in range(500)]
    with pytest.raises(ValueError, match='size|large|MB'):
        cache.write([(0, 1)], detections, 1000)
    assert not list(tmp_path.rglob('*'))


def test_symlink_cache_never_reads_or_changes_its_target(tmp_path):
    root = tmp_path / 'cache'
    cache = cache_at(root)
    cache.write([(0, 1)], [(0.5, 'Yes.')], 2)
    path = next(root.rglob('*.json'))
    outside = tmp_path / 'outside.json'
    path.rename(outside)
    original = outside.read_bytes()
    path.symlink_to(outside)
    assert cache.read([(0, 1)], 2) is None
    with pytest.raises(RuntimeError, match='symlink|regular'):
        cache.write([(0, 1)], [(0.5, 'No.')], 2)
    assert path.is_symlink()
    assert outside.read_bytes() == original


def test_directory_in_place_of_cache_is_not_opened_as_a_file(tmp_path):
    cache = cache_at(tmp_path)
    cache.write([(0, 1)], [], 2)
    path = next(tmp_path.rglob('*.json'))
    path.unlink()
    path.mkdir()
    assert cache.read([(0, 1)], 2) is None


def test_failed_atomic_replace_preserves_previous_scan_and_removes_temporary_file(tmp_path, monkeypatch):
    cache = cache_at(tmp_path)
    old = [(0.5, 'Yes.')]
    cache.write([(0, 1)], old, 2)
    before = set(tmp_path.rglob('*'))

    def fail_replace(*args, **kwargs):
        raise OSError('disk unavailable')

    monkeypatch.setattr(os, 'replace', fail_replace)
    with pytest.raises(RuntimeError, match='write.*caption scan|caption scan.*write'):
        cache.write([(0, 1)], [(0.5, 'No.')], 2)
    assert set(tmp_path.rglob('*')) == before
    assert cache.read([(0, 1)], 2) == old


def test_failed_fsync_leaves_no_partial_cache_or_temporary_file(tmp_path, monkeypatch):
    cache = cache_at(tmp_path)

    def fail_fsync(fd):
        raise OSError('disk full')

    monkeypatch.setattr(os, 'fsync', fail_fsync)
    with pytest.raises(RuntimeError, match='caption scan'):
        cache.write([(0, 1)], [(0.5, 'Yes.')], 2)
    assert not [path for path in tmp_path.rglob('*') if path.is_file()]


def test_read_permission_error_is_reported_instead_of_becoming_a_cache_miss(tmp_path, monkeypatch):
    cache = cache_at(tmp_path)
    cache.write([(0, 1)], [(0.5, 'Yes.')], 2)
    original_open = os.open

    def deny_json(path, flags, *args, **kwargs):
        if str(path).endswith('.json'):
            raise PermissionError('permission denied')
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, 'open', deny_json)
    with pytest.raises(RuntimeError, match='read.*caption scan|caption scan.*read'):
        cache.read([(0, 1)], 2)


@pytest.mark.parametrize('detections', [[[0.5, 'Yes.'], [0.4, 'No.']], [[1, 'Yes.']],
                                       [[True, 'Yes.']], [[0.5, None]], [[10 ** 500, 'Yes.']]])
def test_semantically_invalid_cache_is_a_miss_even_with_matching_digest(tmp_path, detections):
    cache = cache_at(tmp_path)
    cache.write([(0, 1)], [(0.5, 'Yes.')], 2)
    path = next(tmp_path.rglob('*.json'))
    content = json.loads(path.read_text())
    content.pop('digest')
    content['detections'] = detections
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    content['digest'] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(content))
    assert cache.read([(0, 1)], 2) is None


def test_symlink_cache_directory_cannot_redirect_storage(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    root = tmp_path / 'cache'
    root.symlink_to(outside, target_is_directory=True)
    cache = cache_at(root)
    assert cache.read([(0, 1)], 2) is None
    with pytest.raises(RuntimeError, match='directory|symlink'):
        cache.write([(0, 1)], [(0.5, 'Yes.')], 2)
    assert not list(outside.iterdir())


def test_identity_strings_cannot_escape_cache_root(tmp_path):
    root = tmp_path / 'cache'
    cache = cache_at(root, '../outside', '../../elsewhere')
    cache.write([(0, 1)], [(0.5, 'Yes.')], 2)
    assert cache.read([(0, 1)], 2) == [(0.5, 'Yes.')]
    assert list(tmp_path.iterdir()) == [root]
