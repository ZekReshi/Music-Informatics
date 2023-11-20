from typing import Union, Tuple, Iterable

import partitura as pt
from partitura.performance import Performance
import os
import matplotlib.pyplot as plt
import numpy as np
from meter_estimation_utils import compute_autocorrelation
from scipy.signal import find_peaks
import time

from hiddenmarkov import HMM, ConstantTransitionModel, ObservationModel

import warnings
warnings.filterwarnings("ignore")


def pianoroll(performance: Performance):
    note_array = performance.note_array()

    time_div = 16

    piano_roll = pt.utils.music.compute_pianoroll(
        note_info=note_array,
        time_unit="sec",
        time_div=time_div,
        note_separation=True,
    ).toarray()

    frames_nq = (
        pt.utils.music.compute_pianoroll(
            note_info=note_array,
            time_unit="sec",
            time_div=time_div,
            onset_only=True,
            binary=True,  # Count number of notes, instead of velocity
        )
        .toarray()
        .sum(0)
    )

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 8),
        sharex=True,
    )
    ax[0].imshow(
        piano_roll,
        origin="lower",
        aspect="auto",
        cmap="binary",
        interpolation="nearest",
    )
    ax[1].set_xlabel("Time (s)")
    ax[0].set_ylabel("MIDI pitch")
    ax[1].set_ylabel("Simultaneous played notes")
    ax[1].set_xticks(
        np.arange(0, len(frames_nq), 100),
        np.arange(0, len(frames_nq), 100) / time_div,
    )
    ax[1].plot(frames_nq, color="firebrick")
    plt.tight_layout()
    plt.show()


def iois(performance: Performance):
    note_array = performance.note_array()

    IOIs = np.diff(np.sort(note_array["onset_sec"]))
    hist, bins = np.histogram(IOIs, bins=100)

    print(60 / (4 * bins[np.argmax(hist)]))
    fig, ax = plt.subplots()
    ax.hist(IOIs, bins=bins)
    ax.set_xlabel("IOI (s)")
    ax.set_ylabel("Count")
    plt.show()


def quantize(performance: Performance):
    note_array = performance.note_array()

    FRAMERATE = 50

    onset_min = note_array["onset_sec"].min()
    onset_max = note_array["onset_sec"].max()
    onset_duration = onset_max - onset_min
    frames = np.zeros(int(onset_duration * FRAMERATE) + 1)
    for note in note_array:
        frames[int((note["onset_sec"] - onset_min) * FRAMERATE)] += 1

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(frames)) / FRAMERATE, frames)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Note count")
    plt.show()


def chordify(performance: Performance):
    note_array = performance.note_array()
    ppart = performance[0]

    FRAMERATE = 50

    onset_min = note_array["onset_sec"].min()
    onset_max = note_array["onset_sec"].max()
    onset_duration = onset_max - onset_min

    aggregated_notes = [(0, 0)]
    CHORD_SPREAD_TIME = 1 / 12  # a little faster than 16th notes at 180 bpm

    for note in ppart.notes:
        prev_note_on = aggregated_notes[-1][0]
        note_on = note["note_on"]
        prev_note_vel = aggregated_notes[-1][1]
        if abs(note_on - prev_note_on) < CHORD_SPREAD_TIME:
            aggregated_notes[-1] = (note_on, prev_note_vel + 1)
            # aggregated_notes[-1] = (note_on, prev_note_vel + note["velocity"])  # 1
        else:
            aggregated_notes.append((note_on, 1))  # 1
            # aggregated_notes.append((note_on, note["velocity"]))  # 1

    frames_a = np.zeros(int(onset_duration * FRAMERATE) + 1)
    for note in aggregated_notes:
        frames_a[int((note[0] - onset_min) * FRAMERATE)] += note[1]

    frames_a[frames_a < 0.1] = 0.0

    fig, ax = plt.subplots()
    ax.plot(
        np.arange(len(frames_a)) / FRAMERATE,
        frames_a,
        label="cordified",
        color="firebrick",
        alpha=0.5,
    )

    plt.legend(loc="best")

    plt.show()


def autocorr(performance: Performance, time_div: int):
    note_array = performance.note_array()

    frames_nq = (
        pt.utils.music.compute_pianoroll(
            note_info=note_array,
            time_unit="sec",
            time_div=time_div,
            onset_only=True,
            binary=True,  # Count number of notes, instead of velocity
        )
        .toarray()
        .sum(0)
    )

    autocorr = compute_autocorrelation(frames_nq)

    #fig, ax = plt.subplots()
    #ax.plot(autocorr, color="firebrick")
    #ax.set_xlabel("Lag")
    #plt.show()

    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    a, _ = find_peaks(autocorr[1:], prominence=30)
    m = np.argmax(autocorr[1:])
    prediction = 60 / ((m + 1) / time_div)
    #print(60 / ((a[0:16] + 1) / time_div), prediction)

    return prediction


class MeterObservationModel(ObservationModel):
    def __init__(
        self,
        states: int = 100,
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


def getTransitionMatrix(states: int, distribution: Iterable = [0.1, 0.8, 0.1]):
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
    frame_rate: int = 50,  # frames_per_beat
    beats_per_measure: int = 4,
    subbeats_per_beat: int = 2,
):
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
        agg_fun = np.max

    onsets = note_array["onset_sec"]
    sort_idx = np.argsort(onsets)

    onsets = onsets[sort_idx]
    velocity = note_array["velocity"][sort_idx]

    # (onset, agg_val)
    aggregated_notes = [(0, 0)]

    for (note_on, note_vel) in zip(onsets, velocity):
        prev_note_on = aggregated_notes[-1][0]
        prev_note_vel = aggregated_notes[-1][1]
        if abs(note_on - prev_note_on) < chord_spread_time:

            if aggregation == "num_notes":
                agg_val = 1
            elif aggregation == "sum_vel":
                agg_val = prev_note_vel + note_vel
            elif aggregation == "max_vel":
                agg_val = note_vel if note_vel > prev_note_vel else prev_note_vel

            aggregated_notes[-1] = (note_on, agg_val)
        else:

            if aggregation == "num_notes":
                agg_val = 1
            elif aggregation in ("sum_vel", "max_vel"):
                agg_val = note_vel

            aggregated_notes.append((note_on, agg_val))

    frames = np.zeros(int(onsets.max() * framerate) + 1)
    for note in aggregated_notes:
        frames[int((note[0]) * framerate)] += note[1]

    if threshold > 0:
        frames[frames < threshold]

    return frames


def meter_identification(
    note_info: Performance,
    beats_per_measure: Iterable[int] = [2, 3, 4, 6, 12],
    subbeats_per_beat: Iterable[int] = [2, 3],
    tempi: Union[Iterable[int], str] = "auto",
    frame_aggregation: str = "chordify",
    value_aggregation: str = "num_notes",
    framerate: int = 50,
    frame_threshold: float = 0.0,
    chord_spread_time: float = 1 / 12,
    max_tempo: float = 250,
    min_tempo: float = 30,
) -> Tuple[int, float]:
    """
    Meter Identification using HMMs
    """
    # get note array
    note_array = pt.utils.ensure_notearray(note_info)

    if frame_aggregation == "chordify":
        frames = get_frames_chordify(
            note_array=note_array,
            framerate=framerate,
            chord_spread_time=chord_spread_time,
            aggregation=value_aggregation,
            threshold=frame_threshold,
        )
    elif frame_aggregation == "quantize":
        frames = get_frames_quantized(
            note_array=note_array,
            framerate=framerate,
            aggregation=value_aggregation,
            threshold=frame_threshold,
        )

    if tempi == "auto":
        autocorr = compute_autocorrelation(frames)
        beat_period, _ = find_peaks(autocorr[1:], prominence=20)
        tempi = 60 * framerate / (beat_period + 1)
        tempi = tempi[np.logical_and(tempi <= max_tempo, tempi >= min_tempo)]

    likelihoods = []

    for ts_num in beats_per_measure:
        for sbpb in subbeats_per_beat:
            print(str(ts_num) + '/' + str(sbpb), end='')
            for tempo in tempi:
                observation_model, transition_model = createHMM(
                    tempo=tempo,
                    frame_rate=framerate,
                    beats_per_measure=ts_num,
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

                likelihoods.append((ts_num, sbpb, tempo, log_lik))
            print(" done")

    likelihoods = np.array(likelihoods)

    best_result = likelihoods[likelihoods[:, 3].argmax()]

    best_ts = int(best_result[0])
    best_tempo = best_result[2]

    return best_ts, best_tempo


if __name__ == '__main__':
    path_to_dataset = os.path.join(
        os.path.dirname(__file__),
        "Dataset"
    )
    path_to_train = os.path.join(
        path_to_dataset,
        "train"
    )
    path_to_result = os.path.join(
        path_to_dataset,
        "result.txt"
    )
    with open(path_to_result, "w") as f:
        i = 0
        for piece in os.listdir(path_to_train):
            start = time.time()
            if not piece.endswith('.mid'):
                continue
            print(piece)
            path_to_piece = os.path.join(
                path_to_train,
                piece
            )
            p: Performance = pt.load_performance(path_to_piece)
            ts, tempo = meter_identification(p)
            f.write(f"{piece},,{str(ts)},,{str(tempo)}\n")
            f.flush()
            print(f"{time.time() - start} seconds")
            i += 1
            if i == 3:
                break
