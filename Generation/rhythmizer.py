import argparse

import numpy as np
import partitura as pt

from Generation.grammars import get_rhythm, ONSET, HOLD, MUTE

import warnings
warnings.filterwarnings("ignore")


BPM = 100
SIXTEENTH = 15 / BPM


def rhythmize(note_array: np.ndarray):
    phases = [
        (20, 'slow'),
        (20, 'normal'),
        (20, 'slow'),
        (20, 'fast'),
        (20, 'slow')
    ]

    cur_phase = 0

    rhythm = ' '

    note_array_rhythmized = []
    cur_note = None
    state = 0
    phase_offset = 0
    cur_array_note = 0

    while cur_array_note < len(note_array):
        note = note_array[cur_array_note]
        if state >= len(rhythm):
            if cur_note is not None:
                note_array_rhythmized.append(cur_note)

            if cur_phase >= len(phases):
                break

            bars, speed = phases[cur_phase]
            print(speed)
            rhythm = get_rhythm(bars, speed, True)
            print(rhythm)
            phase_offset += 0 if cur_phase == 0 else 60 / BPM * 4 * phases[cur_phase-1][0]
            cur_phase += 1
            state = 0
            cur_note = None if rhythm[0] == MUTE else (phase_offset, SIXTEENTH, note_array[0][4], note_array[0][5])

        if note[0] > (state * SIXTEENTH + phase_offset):
            if rhythm[state] == ONSET:
                if cur_note is not None:
                    note_array_rhythmized.append(cur_note)
                cur_note = (state * SIXTEENTH + phase_offset, SIXTEENTH, note[4], note[5])
            elif rhythm[state] == HOLD:
                if cur_note is not None:
                    cur_note = (cur_note[0], cur_note[1] + SIXTEENTH, cur_note[2], cur_note[3])
            elif rhythm[state] == MUTE:
                if cur_note is not None:
                    note_array_rhythmized.append(cur_note)
                    cur_note = None
            state += 1
        else:
            cur_array_note += 1

    fields = [
        ("onset_sec", "f4"),
        ("duration_sec", "f4"),
        ("pitch", "i4"),
        ("velocity", "i4"),
    ]

    return np.array(note_array_rhythmized, dtype=fields)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Give a .mid file a rhythm.')

    parser.add_argument('--infile', type=str, help='MIDI input file name.')
    parser.add_argument('--outfile', type=str, help='MIDI input file name.')

    args = parser.parse_args()

    p = pt.load_performance(args.infile)
    pp = pt.performance.PerformedPart.from_note_array(rhythmize(p.note_array()))
    pt.save_wav(pp, args.outfile)
