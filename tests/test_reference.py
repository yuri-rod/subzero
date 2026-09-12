from subzero import reference
import json
import subprocess


def test_fingerprint_changes_when_video_is_replaced(tmp_path):
    video = tmp_path/'movie.mkv'
    video.write_bytes(b'one')
    first = reference.fingerprint(video)
    video.write_bytes(b'two')
    assert reference.fingerprint(video) != first


def test_forced_and_commentary_tracks_are_excluded():
    tracks = [
        {'index':1,'codec_type':'subtitle','codec_name':'subrip','disposition':{'forced':1}},
        {'index':2,'codec_type':'subtitle','codec_name':'subrip','tags':{'title':'Commentary'}},
        {'index':3,'codec_type':'subtitle','codec_name':'subrip','tags':{'language':'eng'}},
    ]
    assert [s['index'] for s in reference.dialogue_tracks(tracks)] == [3]


def test_absent_audio_evidence_cannot_certify_a_subtitle():
    report = reference.verify_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n',
                                   {'speech':[], 'text':'', 'spans':[]})
    assert report.status == 'inconclusive'


def test_cached_reference_recovers_duration_without_reextracting_audio(tmp_path, monkeypatch):
    video = tmp_path/'movie.mkv'
    video.write_bytes(b'video')
    cache = tmp_path/f'{reference.fingerprint(video)}.json'
    cache.write_text(json.dumps({'speech': [[1, 2]], 'text': '', 'spans': []}))

    def probe(argv, **kwargs):
        if argv[0] != 'ffprobe':
            raise AssertionError('Cached audio should not be extracted again')
        return subprocess.CompletedProcess(argv, 0, b'{"format":{"duration":"5150.976"}}')

    monkeypatch.setattr(reference, '_run', probe)
    cached = reference.build_reference(video, tmp_path)
    assert cached['duration'] == 5150.976
    assert cached['speech'] == [[1, 2]]
