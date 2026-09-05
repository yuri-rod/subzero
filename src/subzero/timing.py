from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .shift import TIME, _to_seconds

VERSION = '1'


@dataclass
class Window:
    center: float
    offset: float
    score: float
    margin: float
    confident: bool


@dataclass
class Report:
    status: str
    reason: str
    windows: list[Window] = field(default_factory=list)
    validator: str = VERSION

    def json(self):
        return asdict(self)


def spans(text):
    found = []
    for block in text.replace('\r', '').split('\n\n'):
        match = TIME.search(block)
        if match is None or any(mark in block for mark in ('♪', '♫')):
            continue
        groups = match.groups()
        found.append((_to_seconds(*groups[:4]), _to_seconds(*groups[4:])))
    return found


def evaluate(candidate, reference, tolerance=.5, window_seconds=180, stride=90,
             search_seconds=300, phase=0):
    import numpy as np

    if not candidate or not reference:
        return Report('inconclusive', 'No usable dialogue reference or subtitle')
    if any(not math.isfinite(a+b) or a < 0 or b <= a for a,b in candidate):
        return Report('reject', 'Invalid subtitle timestamps')
    if any(a > b[0] for (a,_),b in zip(candidate,candidate[1:])):
        return Report('reject', 'Subtitle timestamps are not ordered')
    end = max(b for _,b in reference)
    if end > 43200 or end < 120 or len(reference) < 20 or len(candidate) < 20:
        return Report('inconclusive', 'Insufficient dialogue or unsupported duration')
    if candidate[-1][1] < .85 * end or candidate[0][0] > reference[0][0] + 120:
        return Report('reject', 'Subtitle does not cover the dialogue span')
    rate = 10
    count = int((end + search_seconds + window_seconds) * rate) + 1

    def signal(intervals):
        track = np.zeros(count)
        for a,b in intervals:
            track[max(0,int(a*rate)):min(count,int(b*rate))] = 1
        return track

    ref, sub = signal(reference), signal(candidate)
    search = int(search_seconds * rate)
    sub = np.pad(sub, (search,search))
    seconds = min(window_seconds, end)
    width = int(seconds * rate)
    starts = list(range(int(phase*rate), max(1,int(end*rate)-width+1), int(stride*rate)))
    starts.append(max(0, int(end*rate)-width))
    windows = []
    for lo in sorted(set(starts)):
        r = ref[lo:lo+width]
        variance = float(np.var(r))
        if variance < .04:
            windows.append(Window((lo+width/2)/rate, 0, 0, 0, False))
            continue
        c = sub[lo:lo+width+2*search]
        sums = np.concatenate(([0.0],np.cumsum(c)))
        totals = sums[width:] - sums[:-width]
        variances = np.maximum(0, totals-totals*totals/width)
        denom = np.sqrt(variances * variance * width)
        scores = np.divide(np.correlate(c, r-r.mean(), mode='valid'), denom,
                           out=np.zeros_like(denom), where=denom > 1e-8)
        best = int(np.argmax(scores))
        score = float(scores[best])
        others = scores.copy()
        others[max(0,best-10):best+11] = -1
        margin = score-float(np.max(others))
        windows.append(Window((lo+width/2)/rate, (search-best)/rate,
                              round(score,4), round(margin,4), score >= .25 and margin >= .035))
    valid = [w for w in windows if w.confident]
    bad = [w for w in valid if abs(w.offset) > tolerance]
    if bad:
        consistent = any(b.center-a.center <= stride*1.5 and
                         abs(a.offset-b.offset) <= 6 for a,b in zip(bad,bad[1:]))
        if consistent:
            return Report('reject', 'Repeated local timing mismatch', windows)
        return Report('inconclusive', 'Isolated timing mismatch needs review', windows)
    covered = len(valid)/max(1,len(windows))
    thirds = {min(2,int(w.center/end*3)) for w in valid}
    if len(valid) < 3 or covered < .7 or len(thirds) < 3:
        return Report('inconclusive', 'Timing evidence is weak or incomplete', windows)
    return Report('pass', 'Timing agrees across the dialogue span', windows)


def correction(report):
    import numpy as np

    windows = [w for w in report.windows if w.confident]
    if len(windows) < 6:
        return None
    train, held = windows[::2], windows[1::2]
    x = np.array([w.center-w.offset for w in train])
    y = np.array([w.offset for w in train])
    slope, offset = np.polyfit(x,y,1)
    scale = 1+float(slope)
    if not .95 <= scale <= 1.05 or abs(offset) > 120:
        return None
    residual = [abs((w.center-w.offset)*slope+offset-w.offset) for w in windows]
    if max(residual) > .4 or not held:
        return None
    return scale, float(offset)
