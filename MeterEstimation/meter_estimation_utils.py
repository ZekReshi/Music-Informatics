#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for meter estimation notebook
"""
import os

import numpy as np
import partitura as pt

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_autocorrelation(x: np.ndarray, mode: str = "full") -> np.ndarray:
    """
    Compute non-normalized autocorrelation (consider only positive lags)

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    result : np.ndarray
        Autocorrelation
    """
    result = np.correlate(x, x, mode=mode)
    # Consider only positive lags
    result = result[result.shape[0] // 2 :]

    return result


def get_frames_quantized(
    note_array: np.ndarray,
    framerate: int = 50,
    aggregation="num_notes",
    threshold: float = 0.0,
) -> np.ndarray:
    if aggregation not in ("num_notes", "sum_vel", "max_vel"):
        raise ValueError(
            "`aggregation` must be 'num_notes', 'sum_vel', 'max_vel' "
            f"but is {aggregation}"
        )

    if aggregation == "num_notes":
        # Count number of simultaneous notes
        binary = True
        agg_fun = np.sum
    elif aggregation == "sum_vel":
        binary = False
        agg_fun = np.sum
    elif aggregation == "max_vel":
        binary = False
        agg_fun = np.max

    onset_pr = pt.utils.music.compute_pianoroll(
        note_info=note_array,
        time_unit="sec",
        time_div=framerate,
        onset_only=True,
        binary=binary,
    ).toarray()

    frames = agg_fun(onset_pr, axis=0)

    if threshold > 0:
        frames[frames < threshold] = 0

    return frames


def get_frames_chordify(
    note_array: np.ndarray,
    framerate: int = 50,
    chord_spread_time: float = 1 / 12,
    aggregation="num_notes",
    threshold: float = 0.0,
) -> np.ndarray:

    if aggregation not in ("num_notes", "sum_vel", "max_vel", "salience"):
        raise ValueError(
            "`aggregation` must be 'num_notes', 'sum_vel', 'max_vel', 'salience' "
            f"but is {aggregation}"
        )

    if aggregation == "num_notes":
        # Count number of simultaneous notes
        binary = True
        agg_fun = np.sum
    elif aggregation == "sum_vel":
        binary = False
        agg_fun = np.sum
    elif aggregation == "max_vel":
        agg_fun = np.max


    onsets = note_array["onset_sec"]
    sort_idx = np.argsort(onsets)

    onsets = onsets[sort_idx]
    velocity = note_array["velocity"][sort_idx]
    duration = note_array["duration_sec"][sort_idx]
    pitch = note_array["pitch"][sort_idx]

    avg_duration = np.average(duration)
    avg_velocity = np.average(velocity)

    # (onset, agg_val, longest_duration, sum_vel, lowest_pitch)
    aggregated_notes = [(0, 0, 0, 0, 0)]

    for (note_on, note_vel, note_duration, note_pitch) in zip(onsets, velocity, duration, pitch):
        prev_note_on = aggregated_notes[-1][0]
        prev_note_vel = aggregated_notes[-1][1]
        prev_longest_duration = aggregated_notes[-1][2]
        prev_sum_vel = aggregated_notes[-1][3]
        prev_lowest_pitch = aggregated_notes[-1][4]
        if abs(note_on - prev_note_on) < chord_spread_time:

            if aggregation == "num_notes":
                agg_val = 1
            elif aggregation == "sum_vel":
                agg_val = prev_note_vel + note_vel
            elif aggregation == "max_vel":
                agg_val = note_vel if note_vel > prev_note_vel else prev_note_vel
            elif aggregation == "salience":
                agg_val = aggregated_notes[-1][1]
                if note_duration > 1.5 * avg_duration:
                    agg_val += 1
                if note_vel > 1.5 * avg_velocity:
                    agg_val += 1
                agg_val = min(3, agg_val)
            elif aggregation == "salience_add":
                agg_val = 300 * max(prev_longest_duration, note_duration) - 4 * max(48, min(72, prev_lowest_pitch, note_pitch)) + 1 * (prev_note_vel + note_vel)
            elif aggregation == "salience_mul":
                agg_val = max(prev_longest_duration, note_duration) * (84 - max(48, min(72, prev_lowest_pitch, note_pitch))) * math.log(prev_note_vel + note_vel)

            aggregated_notes[-1] = (note_on, agg_val, max(prev_longest_duration, note_duration), prev_sum_vel + note_vel, min(prev_lowest_pitch, note_pitch))
        else:

            if aggregation == "num_notes":
                agg_val = 1
            elif aggregation in ("sum_vel", "max_vel"):
                agg_val = note_vel
            elif aggregation == "salience":
                agg_val = 1
                if note_duration > 1.5 * avg_duration:
                    agg_val += 1
                if note_vel > 1.5 * avg_velocity:
                    agg_val += 1
            elif aggregation == "salience_add":
                agg_val = 300 * note_duration - 4 * max(48, min(72, note_pitch)) + 1 * note_vel
            elif aggregation == "salience_mul":
                agg_val = note_duration * (84 - max(48, min(72, note_pitch))) * math.log(note_vel)

            aggregated_notes.append((note_on, agg_val, note_duration, note_vel, note_pitch))

    frames = np.zeros(int(onsets.max() * framerate) + 1)
    for note in aggregated_notes:
        frames[int((note[0]) * framerate)] += note[1]

    if threshold > 0:
        frames[frames  < threshold]

    return frames

