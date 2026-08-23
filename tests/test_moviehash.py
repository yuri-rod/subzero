import pytest
from subzero.moviehash import moviehash, CHUNK

def test_moviehash_computes_64bit_hash(tmp_path):
    p = tmp_path / "sample.mkv"
    # Write at least 2 * CHUNK bytes
    data = b"A" * (CHUNK * 2 + 100)
    p.write_bytes(data)
    h = moviehash(p)
    assert len(h) == 16
    assert isinstance(h, str)

def test_moviehash_rejects_small_files(tmp_path):
    p = tmp_path / "small.mkv"
    p.write_bytes(b"too small")
    with pytest.raises(ValueError, match="too small"):
        moviehash(p)
