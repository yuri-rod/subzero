import pytest

from subzero.moviehash import moviehash

CHUNK = 65536


def write(tmp_path, size, edits=(), name="video.mkv"):
    data = bytearray(size)
    for offset, value in edits:
        data[offset] = value
    p = tmp_path / name
    p.write_bytes(bytes(data))
    return str(p)


def test_zero_file_hashes_to_its_size(tmp_path):
    path = write(tmp_path, 2 * CHUNK)
    assert moviehash(path) == format(2 * CHUNK, "016x")


def test_bytes_from_both_ends_join_the_sum(tmp_path):
    # um 1 no primeiro bloco e outro no ultimo, cada um vale 1 na soma de 64 bits
    path = write(tmp_path, 2 * CHUNK, edits=[(0, 1), (2 * CHUNK - 8, 1)])
    assert moviehash(path) == format(2 * CHUNK + 2, "016x")


def test_middle_bytes_are_ignored(tmp_path):
    quiet = write(tmp_path, 4 * CHUNK, name="quiet.mkv")
    noisy = write(tmp_path, 4 * CHUNK, edits=[(2 * CHUNK, 255)], name="noisy.mkv")
    assert moviehash(quiet) == moviehash(noisy)


def test_small_file_is_rejected(tmp_path):
    path = write(tmp_path, 1024)
    with pytest.raises(ValueError):
        moviehash(path)
