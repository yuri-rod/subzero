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
class Segment:
    start: float
    end: float
    offset: float
    scale: float = 1.0
    confidence: float = 1.0
    windows_count: int = 0
    ref_start: float = 0.0
    ref_end: float = 0.0

    def json(self):
        return asdict(self)


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
        if any(abs(w.offset) > 5.0 for w in bad):
            return Report('reject', 'Severe timing offset detected', windows)
        if len(bad) >= 3:
            return Report('reject', 'Repeated timing mismatch across multiple windows', windows)
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
    if len(windows) >= 6:
        train, held = windows[::2], windows[1::2]
        x = np.array([w.center-w.offset for w in train])
        y = np.array([w.offset for w in train])
        slope, offset = np.polyfit(x,y,1)
        scale = 1+float(slope)
        if .95 <= scale <= 1.05 and abs(offset) <= 120:
            residual = [abs((w.center-w.offset)*slope+offset-w.offset) for w in windows]
            if max(residual) <= .4 and held:
                return scale, float(offset)

    if len(windows) >= 3:
        offsets = np.array([w.offset for w in windows])
        median_offset = float(np.median(offsets))
        residuals = np.abs(offsets - median_offset)
        if max(residuals) <= 0.35 and abs(median_offset) <= 120 and abs(median_offset) >= 0.05:
            return 1.0, median_offset

    return None


def piecewise_correction(
    report: Report,
    min_segments: int = 2,
    min_windows_per_segment: int = 2,
    max_residual: float = 0.4,
    min_jump: float = 1.0,
) -> list[Segment] | None:
    """Detect segmented constant offsets across act breaks or recut broadcast masters.

    Requires per-segment speech-anchor confidence thresholds so sparse intervals
    or noise do not skew dialogue timing.
    """
    import numpy as np

    valid = [w for w in report.windows if w.confident]
    if len(valid) < min_segments * min_windows_per_segment:
        return None

    # Detect candidate break indices between consecutive confident windows
    breaks = []
    for i in range(len(valid) - 1):
        delta = abs(valid[i + 1].offset - valid[i].offset)
        if delta >= min_jump:
            breaks.append(i + 1)

    if len(breaks) < min_segments - 1:
        return None

    # Partition windows into contiguous slices based on breaks
    slices = []
    prev_idx = 0
    for b_idx in breaks:
        slices.append(valid[prev_idx:b_idx])
        prev_idx = b_idx
    slices.append(valid[prev_idx:])

    # Validate each segment against confidence thresholds
    segment_stats = []
    for s in slices:
        if len(s) < min_windows_per_segment:
            return None
        offsets = np.array([w.offset for w in s])
        median_offset = float(np.median(offsets))
        residuals = np.abs(offsets - median_offset)
        if float(np.max(residuals)) > max_residual:
            return None
        avg_score = float(np.mean([w.score for w in s]))
        segment_stats.append((median_offset, avg_score, len(s)))

    # Verify that adjacent segments actually have distinct offsets
    for i in range(len(segment_stats) - 1):
        if abs(segment_stats[i + 1][0] - segment_stats[i][0]) < min_jump:
            return None

    # Construct time boundaries covering the full timeline
    segments = []
    num_segs = len(slices)
    ref_boundaries = [0.0]
    for i in range(num_segs - 1):
        midpoint = (slices[i][-1].center + slices[i + 1][0].center) / 2.0
        ref_boundaries.append(midpoint)
    ref_boundaries.append(float('inf'))

    sub_boundaries = [0.0]
    for i in range(1, num_segs):
        o_prev = segment_stats[i - 1][0]
        o_curr = segment_stats[i][0]
        sub_mid = max(sub_boundaries[i - 1], ref_boundaries[i] - (o_prev + o_curr) / 2.0)
        sub_boundaries.append(sub_mid)
    sub_boundaries.append(float('inf'))

    for i in range(num_segs):
        med_off, score, count = segment_stats[i]
        segments.append(Segment(
            start=sub_boundaries[i],
            end=sub_boundaries[i + 1],
            offset=med_off,
            scale=1.0,
            confidence=round(score, 4),
            windows_count=count,
            ref_start=ref_boundaries[i],
            ref_end=ref_boundaries[i + 1],
        ))

    return segments
