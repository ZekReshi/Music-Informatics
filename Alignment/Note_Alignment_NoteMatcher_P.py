#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline submission for the music alignment challenge 
for Musical Informatics WS23
"""
import warnings

import parangonar as pa
import os
from concurrent.futures import ProcessPoolExecutor

# Uncomment this line if the kernel keeps crashing
# See https://stackoverflow.com/a/53014308
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from typing import Tuple, Dict, Any

import numpy as np

from challenge_utils import (
    load_dataset,
    compare_alignments,
    export_to_challenge,
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="partitura.*",
)

#### For Parallel Processing


def process_piece(
    piece_name: str,
    pdata: Tuple,
) -> Tuple[str, Dict[str, Any], Tuple[float, float, float]]:
    """
    Compute the alignment of a piece (for parallel processing)

    This method is the inner computation in the for loop in 
    Baseline_Alignment.py
    """
    # Extract data from pdata
    performance_note_array, score_note_array, gt_alignment = pdata

    # Use automatic note matcher for note matching
    sdm = pa.AutomaticNoteMatcher(s_time_div=32,
                                  p_time_div=32,
                                  window_size=2)

    predicted_alignment = sdm(score_note_array,
                              performance_note_array)

    # Compute evaluation (Do not change this)
    piece_eval = compare_alignments(
        prediction=predicted_alignment,
        ground_truth=gt_alignment,
    )

    return piece_name, predicted_alignment, piece_eval


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score-to-performance Alignment",
    )

    parser.add_argument(
        "--datadir",
        "-i",
        help="path to the input files",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--outdir",
        "-o",
        help="Output text file directory",
        type=str,
        default=".",
    )

    parser.add_argument(
        "--challenge",
        "-c",
        help="Export results for challenge",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--max-workers",
        "-m",
        help=(
            "Maximum number of workers for parallel processing "
            "of the pieces (by default will use all available processors)"
        ),
        default=None,
        type=int,
    )
    args = parser.parse_args()

    if args.datadir is None:
        raise ValueError("No data directory given")

    # Create output directory if it does not exist
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # 1. Load the data
    dataset = load_dataset(args.datadir)

    # Lists to store results
    alignments = []
    evaluation = []
    piece_names = []

    # Using ProcessPoolExecutor to parallelize the loop
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(executor.map(process_piece, dataset.keys(), dataset.values()))

    # Aggregate results
    for res_piece_name, res_predicted_alignment, res_piece_eval in results:
        piece_names.append(res_piece_name)
        alignments.append(res_predicted_alignment)
        evaluation.append(res_piece_eval)
        print(
            f"{res_piece_name}: "
            f"F-score:{res_piece_eval[2]:.2f} "
            f"Precision:{res_piece_eval[0]:.2f} "
            f"Recall:{res_piece_eval[1]:.2f}"
        )

    # compute mean evaluation
    mean_eval = np.mean(evaluation, 0)

    print(
        "\n\nAverage Performance over the dataset\n"
        f"F-score:{mean_eval[2]:.2f}\t"
        f"Precision:{mean_eval[0]:.2f}\t",
        f"Recall:{mean_eval[1]:.2f}",
    )

    if args.challenge:
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        outfile = os.path.join(args.outdir, f"{script_name}_challenge.npz")

        export_to_challenge(
            alignments=alignments,
            piece_names=piece_names,
            out=outfile,
        )
