import json
import subprocess
import sys

import pytest

from subzero.ocr import get_vision_ocr_bin


@pytest.mark.skipif(sys.platform != 'darwin', reason='Apple Vision requires macOS')
def test_vision_reports_an_unreadable_frame(tmp_path):
    binary = get_vision_ocr_bin()
    if binary is None:
        pytest.skip('Apple Vision compiler is unavailable')
    frame = tmp_path/'missing.jpg'
    proc = subprocess.run([binary, '--json', str(frame)], capture_output=True, text=True)
    assert proc.returncode != 0
    frames = json.loads(proc.stdout)
    assert frames[0]['file'] == str(frame)
    assert frames[0]['error']
