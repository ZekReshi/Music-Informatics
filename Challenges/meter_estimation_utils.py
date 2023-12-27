import numpy as np


def compute_autocorrelation(x: np.ndarray, mode: str = "full") -> np.ndarray:
    result = np.correlate(x, x, mode=mode)
    # Consider only positive lags
    result = result[result.shape[0] // 2 :]

    return result


def get_frames_chordify(
    note_array: np.ndarray,
    framerate: int = 50,
    chord_spread_time: float = 1 / 12,
    threshold: float = 0.0,
) -> np.ndarray:
    onsets = note_array["onset_sec"]
    sort_idx = np.argsort(onsets)

    onsets = onsets[sort_idx]

    # (onset, agg_val, longest_duration, sum_vel, lowest_pitch)
    aggregated_notes = [(0, 0)]

    for note_on in onsets:
        prev_note_on = aggregated_notes[-1][0]
        if abs(note_on - prev_note_on) >= chord_spread_time:
            aggregated_notes.append((note_on, 1))

    frames = np.zeros(int(onsets.max() * framerate) + 1)
    for note in aggregated_notes:
        frames[int((note[0]) * framerate)] += note[1]

    if threshold > 0:
        frames[frames < threshold]

    return frames

