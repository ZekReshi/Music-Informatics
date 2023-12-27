#!/usr/bin/env python
# -*- coding: utf-8 -*-

# single script challenge submission template
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from typing import Tuple, Iterable, Dict, List

import numpy as np
import partitura as pt

from partitura.utils.misc import PathLike

from scipy.signal import find_peaks

from hiddenmarkov import HMM, ConstantTransitionModel, ObservationModel

from meter_estimation_utils import get_frames_chordify, compute_autocorrelation

from meter_estimation_challenge import load_submission

import warnings

warnings.filterwarnings("ignore")


FRAMERATE = 16


class MeterObservationModel(ObservationModel):
    def __init__(
        self,
        states: int = 20,
        downbeat_idx: Iterable = [0],
        beat_idx: Iterable = [50],
        subbeat_idx: Iterable = [25],
    ):
        super().__init__()
        self.states = states
        # observation 1 = note onset present, 0 = nothing present
        self.probabilities = np.ones((2, states)) / 100
        self.probabilities[0, :] = 0.99
        for idx in subbeat_idx:
            self.probabilities[:, idx] = [0.5, 0.5]
        for idx in beat_idx:
            self.probabilities[:, idx] = [0.3, 0.7]
        for idx in downbeat_idx:
            self.probabilities[:, idx] = [0.1, 0.9]
        self.db = downbeat_idx
        self.b = beat_idx
        self.sb = subbeat_idx

    def get_beat_states(self, state_sequence: np.ndarray) -> np.ndarray:
        state_encoder = np.zeros_like(state_sequence)
        for i, state in enumerate(state_sequence):
            if state in self.sb:
                state_encoder[i] = 1
            if state in self.b:
                state_encoder[i] = 2
            if state in self.db:
                state_encoder[i] = 3
        return state_encoder

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if not self.use_log_probabilities:
            return self.probabilities[observation, :]
        else:
            return np.log(self.probabilities[observation, :])


def getTransitionMatrix(
    states: int,
    distribution: Iterable = [0.1, 0.8, 0.1],
) -> np.ndarray:
    """
    Compute transition Matrix
    """
    if states == 1:
        raise ValueError("The number of states should be > 1")
    transition_matrix = (
        np.eye(states, k=0) * distribution[0]
        + np.eye(states, k=1) * distribution[1]
        + np.eye(states, k=2) * distribution[2]
        + np.ones((states, states)) / 1e7
    )

    transition_matrix[-2, 0] = distribution[2]
    transition_matrix[-1, 0] = distribution[2] + distribution[1]

    return transition_matrix


def createHMM(
    tempo: float = 50,
    frame_rate: int = FRAMERATE,  # frames_per_beat
    beats_per_measure: int = 4,
    subbeats_per_beat: int = 2,
) -> Tuple[MeterObservationModel, ConstantTransitionModel]:
    frames_per_beat = 60 / tempo * frame_rate
    frames_per_measure = frames_per_beat * beats_per_measure
    states = int(frames_per_measure)
    downbeat_idx = [0]
    beat_idx = [int(states / beats_per_measure * k) for k in range(beats_per_measure)]
    subbeat_idx = [
        int(states / (beats_per_measure * subbeats_per_beat) * k)
        for k in range(beats_per_measure * subbeats_per_beat)
    ]

    observation_model = MeterObservationModel(
        states=states,
        downbeat_idx=downbeat_idx,
        beat_idx=beat_idx,
        subbeat_idx=subbeat_idx,
    )

    transition_matrix = getTransitionMatrix(states)
    transition_model = ConstantTransitionModel(transition_matrix)

    return observation_model, transition_model


def subbeats_from_durations(note_array: np.ndarray):
    durations = note_array["duration_sec"]
    hist, bins = np.histogram(durations, bins=100)

    peaks, _ = find_peaks(hist, prominence=20)

    if len(peaks) > 1:
        candidate = bins[peaks[1]] / bins[peaks[0]]
        if 1.9 < candidate < 2.1:
            return [2]
        if 2.9 < candidate < 3.1:
            return [3]
    return [2, 3]


def tempi_from_iois(note_array: np.ndarray, min_tempo: float, max_tempo: float, subbeats_per_beat: List[int]):
    IOIs = np.diff(np.sort(note_array["onset_sec"]))
    hist, bins = np.histogram(IOIs, bins=100)

    valid_from = 0
    for i in range(len(bins)):
        if bins[i] >= 1 / 16:
            valid_from = i
            break

    new_hist = []
    new_labels = []
    for i in range(valid_from - 1, len(hist) - 1):
        new_hist.append((hist[i-1] + hist[i] + hist[i+1]) / 3)
        new_labels.append((bins[i+1] + bins[i]) / 2)

    peaks, _ = find_peaks(new_hist, prominence=5)

    if len(peaks) == 0:
        return []

    subbeat_duration = new_labels[peaks[0]]
    subbeats_per_minute = 60 / subbeat_duration
    tempi = []
    for sbpb in subbeats_per_beat:
        beats_per_minute = subbeats_per_minute / sbpb
        for i in range(3):
            tempo = beats_per_minute / (2 ** i)
            if min_tempo < tempo < max_tempo:
                tempi.append(tempo)
    return tempi


def estimate_tempo(
    filename: PathLike,
    beats_per_measure: int,
    value_aggregation: str = "num_notes",
    framerate: int = FRAMERATE,
    frame_threshold: float = 0.0,
    chord_spread_time: float = 1 / 12,
    max_tempo: float = 250,
    min_tempo: float = 30,
) -> float:
    """
    Estimate tempo, meter (currently only time signature numerator)

    Parameters
    ----------
    note_array : structured array

    Returns
    -------
    meter_numerator: int
        The numerator of the time signature
    tempo: float
        The tempo in beats per minute
    """
    # get note array
    performance = pt.load_performance_midi(filename)
    note_array = performance.note_array()
    subbeats_per_beat = subbeats_from_durations(note_array)
    tempi = tempi_from_iois(note_array, min_tempo, max_tempo, subbeats_per_beat)
    if len(tempi) == 0:
        tempi = "auto"

    frames = get_frames_chordify(
        note_array=note_array,
        framerate=framerate,
        chord_spread_time=chord_spread_time,
        aggregation=value_aggregation,
        threshold=frame_threshold,
    )

    if tempi == "auto":
        autocorr = compute_autocorrelation(frames)
        beat_period, _ = find_peaks(autocorr[1:], prominence=None)
        tempi = 60 * framerate / (beat_period + 1)
        tempi = tempi[np.logical_and(tempi <= max_tempo, tempi >= min_tempo)]

        if len(tempi) == 0:
            tempi = np.linspace(min_tempo, max_tempo, 10)

    likelihoods = []

    for sbpb in subbeats_per_beat:
        for tempo in tempi:
            observation_model, transition_model = createHMM(
                tempo=tempo,
                frame_rate=framerate,
                beats_per_measure=beats_per_measure,
                subbeats_per_beat=sbpb,
            )

            hmm = HMM(
                observation_model=observation_model,
                transition_model=transition_model,
            )

            frames[frames < 1.0] = 0
            frames[frames >= 1.0] = 1

            observations = np.array(frames, dtype=int)
            _, log_lik = hmm.find_best_sequence(observations)

            likelihoods.append((sbpb, tempo, log_lik))

    likelihoods = np.array(likelihoods)

    best_result = likelihoods[likelihoods[:, 2].argmax()]

    best_tempo = best_result[1]

    return best_tempo


def process_file(
    mfn: PathLike, file_to_fix: Dict[str, Tuple[float, float]],
) -> Tuple[str, int, float]:
    """
    Compute meter and get evaluation for
    """
    piece: str = os.path.basename(mfn)
    meter = int(file_to_fix[piece][0])
    predicted_tempo: float = estimate_tempo(filename=mfn, beats_per_measure=meter)

    return (
        piece,
        meter,
        predicted_tempo,
    )


if __name__ == "__main__":
    import argparse
    import os
    import glob

    # DO NOT CHANGE THIS!
    parser = argparse.ArgumentParser(description="Meter Estimation")
    parser.add_argument(
        "--datadir",
        "-i",
        help="path to the input files",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--challenge",
        "-c",
        help="Export results for challenge",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--outfile",
        "-o",
        help="Output file",
        type=str,
        default="meter_estimation.txt",
    )

    args = parser.parse_args()

    # Adapt this part as needed!
    midi_files = glob.glob(os.path.join(args.datadir, "*.mid"))
    midi_files.sort()

    file_to_fix = load_submission(args.outfile)

    # Parallel processing with concurrent.futures
    with ProcessPoolExecutor() as executor:
        # Using executor.map for parallel processing
        results_ = list(
            tqdm(
                executor.map(
                    process_file,
                    midi_files,
                    len(midi_files) * [file_to_fix],
                ),
                total=len(midi_files),
            )
        )

    results = [res[:3] for res in results_]

    if args.challenge:
        # Export predictions for the challenge
        np.savetxt(
            args.outfile + "_fixed.txt",
            np.array(results),
            fmt="%s",
            delimiter=",",
            comments="//",
            header="filename,ts_num,tempo",
        )
