import numpy as np
import torch

from meter_estimation_challenge import load_submission
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import partitura as pt
import warnings

warnings.filterwarnings("ignore")


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
                agg_val = 2
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

    frames = np.zeros(int(onsets.max() * framerate) + 1, dtype=np.float32)
    for note in aggregated_notes:
        frames[int((note[0]) * framerate)] += note[1]

    if threshold > 0:
        frames[frames < threshold]

    return frames


class Net(nn.Module):
    def __init__(self, framerate, seconds, features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(framerate * seconds * features, 1024, dtype=torch.float32)
        self.fc2 = nn.Linear(1024, 128, dtype=torch.float32)
        self.fc3 = nn.Linear(128, 16, dtype=torch.float32)
        self.fc4 = nn.Linear(16, 1, dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_epoch(epoch_index, model, optimizer, loss_fn, data, labels):
    print(f"Epoch {epoch_index+1}")
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (input, label) in enumerate(zip(data, labels)):
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output = model(input)

        # Compute the loss and its gradients
        loss = loss_fn(output, label)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            #tb_x = epoch_index * len(training_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train(midi, gt):
    framerate = 24
    seconds = 16
    frames_per_window = framerate * seconds
    stride_seconds = 4
    stride_frames = stride_seconds * framerate

    data = []
    labels = []
    for file in midi:
        piece = os.path.basename(file)
        print(piece)
        performance = pt.load_performance_midi(file)
        note_array = performance.note_array()
        frames = get_frames_chordify(
            note_array=note_array,
            framerate=framerate
        )
        frame = 0
        while frame+frames_per_window < len(frames):
            data.append(torch.from_numpy(frames[frame:frame+frames_per_window]))
            frame += stride_frames
            labels.append(torch.tensor(gt[piece][1], dtype=torch.float32))

    model = Net(framerate, seconds, 1)

    def loss_fn(input, target):
        return (input - target) ** 2

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 10

    for i in range(epochs):
        train_epoch(i, model, optimizer, loss_fn, data, labels)


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
        "--ground-truth",
        "-t",
        help="File with the ground truth labels",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # Adapt this part as needed!
    midi_files = glob.glob(os.path.join(args.datadir, "*.mid"))
    midi_files.sort()

    ground_truth = {}
    if args.ground_truth:
        ground_truth = load_submission(args.ground_truth)

    train(midi_files, ground_truth)
