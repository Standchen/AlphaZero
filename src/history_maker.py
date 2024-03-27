"""This script runs self-play for a given board environment and model using MCTS, and stores the generated game history.

The script accepts command-line arguments to configure the board, model, MCTS parameters, and parallelization settings.

Board games and models must be defined in the environment and network modules, respectively, and imported into this script.

Note
----
Refer to training.py for a comprehensive explanation of the command-line arguments.
"""


import argparse
import torch

from environment.othello import OthelloBoard
from environment.tictactoe import TictactoeBoard
from mcts.mcts import *
from mcts.network.network import *
from mcts.network.process import *


TMP_PATH = "../data/tmp/"
HISTORY_PATH = "../data/history/"
MODEL_PATH = "../models/"


if __name__ == "__main__":
    # Argument parsing.
    parser = argparse.ArgumentParser()

    parser.add_argument("--generation", type=int, required=True, dest="generation")
    parser.add_argument("--subproc-no", type=int, required=True, dest="subproc_no")
    parser.add_argument("--num-play", type=int, required=True, dest="num_play")
    parser.add_argument("--pickle-path", type=str, required=True, dest="pickle_path")

    args = parser.parse_args()

    # Load pickled data for subprocessing.
    with open(args.pickle_path, "rb") as f:
        data = pickle.load(f)

    # Translate parsed arguments to actual objects.
    board_class = data["board_class"]
    model_class = data["model_class"]
    num_evaluation = data["num_evaluation"]
    temp_threshold = data["temp_threshold"]
    dirichlet_alpha = data["dirichlet_alpha"]
    dirichlet_weight = data["dirichlet_weight"]
    num_thread = data["num_thread"]

    # if isinstance(board_class, str):
    #     board_class = globals().get(board_class)
    # if isinstance(model_class, str):
    #     model_class = globals().get(model_class)

    # Load model.
    model = model_class()
    model.load_state_dict(load_model(generation=args.generation, attr="tmp", path=MODEL_PATH))

    # Instantiate MCTS.
    mcts = MCTS(
        board_class,
        model,
        num_evaluation,
        temp_threshold,
        dirichlet_alpha,
        dirichlet_weight,
    )

    # Generate game history using MCTS self-play.
    with torch.no_grad():
        if num_thread == 1:
            history = mcts.self_play(num_play=args.num_play)
        else:
            history = mcts.make_history(num_play=args.num_play, num_thread=num_thread)

    # Save the generated history.
    write_history(
        history,
        generation=args.generation,
        subproc_no=args.subproc_no,
        path=HISTORY_PATH,
    )
