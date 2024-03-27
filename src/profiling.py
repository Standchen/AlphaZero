from environment.tictactoe import TictactoeBoard
from environment.othello import OthelloBoard
from fight import fight
from mcts.network.augment import *
from mcts.network.network import TictactoeDualNetwork, OthelloDualNetwork
from mcts.network.process import *
from mcts.mcts import *
from strategy.strategy import *
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

import cProfile
import pstats
import io

HISTORY_PATH = "../data/history/"
MODEL_PATH = "../models/"


def train(
    num_generation: int,
    num_epoch: int,
    lr: float,
    num_play: int,
    num_thread: int,
    update_per: int,
    num_fight: int,
    board_class,
    model_class,
    load_generation: int | None = None,
    augment_fn: callable = None,
    num_evaluation: int = 50,
    temperature: float = 1.0,
    dirichlet_alpha: float | None = 0.3,
    dirichlet_weight: float = 0.25,
):
    if num_play % num_thread:
        print(f"[*] Warning: Multiprocessing might be unbalanced ({num_play = }, {num_thread = }).")

    # Create model.
    model = model_class()

    # Load model weight is specified.
    if load_generation is None:
        print("[*] No model specified to load. Starting from scratch.")
        start_generation = 0
    elif isinstance(load_generation, int):
        print(f"[*] generation {load_generation} is specified to load.")
        start_generation = load_generation + 1
        model.load_state_dict(load_model(generation=load_generation, path=MODEL_PATH))
    else:
        raise ValueError

    # model = model.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mcts = MCTS(
        board_class,
        model,
        num_evaluation,
        temperature,
        dirichlet_alpha,
        dirichlet_weight,
    )

    losses, policy_losses, value_losses = [], [], []
    for generation in (pbar := tqdm(range(start_generation, start_generation + num_generation))):
        pbar.set_description(f"generation {generation}")

        # Make history.
        with torch.no_grad():
            if num_thread == 1:
                history = mcts.self_play(num_play=num_play)
            else:
                history = mcts.make_history(num_play=num_play, num_thread=num_thread)

        # write_history(history, generation=generation, path=HISTORY_PATH)

        """
        Optimize the model with the history.
        Make sure the model resides in CPU after the optimization.
        """
        model.to("cuda")
        loss, policy_loss, value_loss = optimize_model(model, optimizer, num_epoch, history, augment_fn)
        model.to("cpu")

        # Save loss information.
        losses.extend(loss)
        policy_losses.extend(policy_loss)
        value_losses.extend(value_loss)

        # Update progress bar with loss info.
        loss_str = f"loss = {loss[-1]:.3f} (policy_loss = {policy_loss[-1]:.3f}, value_loss = {value_loss[-1]:.3f})"
        pbar.set_postfix_str(loss_str)

        # To leave progress on the display.
        print("")

        # Save model
        # save_model(model, generation=generation, path=MODEL_PATH)

        # Update the best model
        if generation == 0:
            # Initial model be the best model at first.
            # save_model(model, generation=generation, path=MODEL_PATH, is_best=True)
            pass
        elif generation != 0 and generation % update_per == 0:
            # Evaluate network
            print("[*] Evaluate network")
            curr_model = model_class()
            curr_model.load_state_dict(model.state_dict())
            curr_mcts = MCTS(
                board_class,
                curr_model,
                num_evaluation,
                temperature,
                dirichlet_alpha=None,
                dirichlet_weight=0,
            )

            best_model = model_class()
            best_model.load_state_dict(load_model(generation=None, path=MODEL_PATH, is_best=True))
            best_mcts = MCTS(
                board_class,
                best_model,
                num_evaluation,
                temperature,
                dirichlet_alpha=None,
                dirichlet_weight=0,
            )

            curr_win, best_win = fight(
                board_class,
                num_fight=num_fight,
                playfunc_0=curr_mcts.play,
                playfunc_1=best_mcts.play,
            )
            # If current model wins, update the best model.
            print(f"[*] Last: {100 * curr_win / num_fight:.2f}%")
            print(f"[*] Best: {100 * best_win / num_fight:.2f}%")
            if curr_win > best_win:
                print(f"[*] Change the model at generation {generation}")
                # save_model(model, generation=generation, path=MODEL_PATH, is_best=True)
            else:
                print(f"[*] NOT changing the model at generation {generation}")

            # # Fight with alpha-beta pruning strategy.
            # print('[*] Fight with alpha-beta pruning strategy.')
            # curr_win, ab_win = fight(board_class, num_fight=num_fight,
            #                          playfunc_0=curr_mcts.play, playfunc_1=alpha_beta_pruning_play)
            # print(f'[*] Last: {100 * curr_win / num_fight:.2f}%')
            # print(f'[*] Alpha-beta: {100 * ab_win / num_fight:.2f}%')

    return losses, policy_losses, value_losses


if __name__ == "__main__":
    load_generation = None

    pr = cProfile.Profile()
    pr.enable()
    loss_info = train(
        num_generation=1,
        num_epoch=1,
        lr=1e-4,
        num_play=30,
        num_thread=1,
        update_per=4,
        num_fight=64,
        board_class=OthelloBoard,
        model_class=OthelloDualNetwork,
        load_generation=load_generation,
        augment_fn=None,
        num_evaluation=90,
        temperature=1.0,
    )
    pr.disable()

    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    exit()
