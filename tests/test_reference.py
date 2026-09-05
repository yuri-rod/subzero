from subzero import reference


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
