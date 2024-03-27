from datetime import datetime
import glob
import os
import pickle

import numpy as np
import torch
import torch.multiprocessing as mp

from .node import Node


class MCTS:
    """A class to perform Monte Carlo Tree Search (MCTS) using a neural network to evaluate
    board states and predict move probabilities.

    Parameters
    ----------
    board_class : Class
        The class of the board game to be played.
    model : torch.nn.Module
        The neural network model used for predictions.
    num_evaluation : int, optional, default=50
        The number of times to evaluate a node during the search.
    temp_threshold : int | float
        The threshold of temperature scheduler used in Boltzmann scaling for controlling the level of exploration during the search.
        It should be either integer or `float("inf")`.
    dirichlet_alpha : float | None, optional, default=0.3
        The alpha parameter for the Dirichlet distribution used to add noise to the move probabilities.
    dirichlet_weight : float, optional, default=0.25
        The weight of the Dirichlet noise in the final move probabilities.
    """

    def __init__(
        self,
        board_class,
        model,
        num_evaluation: int = 50,
        temp_threshold: int | float = float("inf"),
        dirichlet_alpha: float | None = 0.3,
        dirichlet_weight: float = 0.25,
    ):
        self.board_class = board_class
        self.model = model

        self.num_evaluation = num_evaluation
        self.temp_threshold = temp_threshold

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

        # Set model to eval mode.
        self.model.eval()

        # The exploration-exploitation balance coefficient.
        self.c_puct = 1.0

        # Whether the MCTS is running in parallel mode.
        self.is_parallel = False

    def play(self, board):
        """Given the current board state, return the next move based on MCTS."""
        node = Node(board=board, mcts=self)
        move, _, _ = self._play_internal(node)
        return move
        # try:
        #     if board.nth_move < self.last_node.board.nth_move:
        #         raise TypeError
        #     found = False
        #     for child_node in self.last_node.child_nodes:
        #         for grandchild_node in child_node.child_nodes:
        #             if torch.equal(grandchild_node.board.to_tensor(), board.to_tensor()):
        #                 node = grandchild_node
        #                 found = True
        #                 break
        #         if found:
        #             break
        #     else:
        #         assert (
        #             False
        #         ), f"{self.last_node.board.nth_move}, {self.last_node.board.pieces}, {child_node.board.pieces}, {board.pieces}"
        # except (AttributeError, TypeError):
        #     # Root
        #     node = Node(board=board, mcts=self)
        # move, _, _ = self._play_internal(node)
        # self.last_node = node
        # return move

    def self_play(self, num_play: int):
        """Generate a history of game states and policies during self-play.

        It runs in the main process, sequentially.

        Parameters
        ----------
        num_play : int
            The number of games to play in self-play.

        Returns
        -------
        history : list
            A list of game states, policies, and values from self-play.
        """
        history = []
        for _ in range(num_play):
            h = self._self_play_once()
            history.extend(h)
        return history

    def make_history(self, num_play: int, num_thread: int):
        """Generate a self-play history parallelly.

        This method spawns multiple child processes and runs self-play simultaneously.
        The main process remains in this method and manages its children
        while they left to execute `_self_play_with_queues` method.

        It collects tensors from child procs,
        processes them 'at once' with the neural network (to exploit GPU parallelism)
        and then distributes the results.

        It also manages the termination of child processes, collects their histories,
        and finally returns the aggregated history.

        Parameters
        ----------
        num_play: int
            The number of games to play in parallel self-play.
        num_thread: int
            The number of threads to use for parallel self-play.

        Returns
        -------
        history : list
            A list of game states, policies, and values from parallel self-play.
        """
        # MCTS is now switched to parallel mode.
        self.is_parallel = True

        play_per_thread = [num_play // num_thread] * num_thread
        play_per_thread[-1] += num_play % num_thread

        # Multiprocessing queues.
        # Scatter queues: Main -> Child
        # Gather queues: Child -> Main
        mp.set_sharing_strategy("file_system")
        ctx = mp.get_context("spawn")
        self.gather_queues = [ctx.Queue() for _ in range(num_thread)]
        self.scatter_queues = [ctx.Queue() for _ in range(num_thread)]
        self.result_queues = [ctx.Queue() for _ in range(num_thread)]

        # Spawn child processes.
        # Child processes are created to execute the `_self_play_with_queues` method,
        # while the main process continues executing the rest of the code in this method.
        processes = []
        for rank in range(num_thread):
            proc = ctx.Process(target=self._self_play_with_queues, args=(play_per_thread[rank], rank))
            proc.start()
            processes.append(proc)

        # Move the main process's model to CUDA.
        # It is important to do this only after spawning the child processes,
        # as doing it earlier can lead to unexpected CUDA behavior.
        assert not next(self.model.parameters()).is_cuda
        self.model.to("cuda")

        # Collect & process tensors.
        active_status = [True for _ in range(num_thread)]
        while sum(active_status) > 0:
            xs = []
            sizes = []
            wait_ranks = []
            for rank in range(num_thread):
                # If terminated child proc, continue.
                if not active_status[rank]:
                    continue
                # Gather the tensors to process from child processes.
                if (x := self.gather_queues[rank].get()) is None:
                    # If child proc says that its job is done, acknowledge and continue.
                    active_status[rank] = False
                    continue
                xs.append(x)
                sizes.append(x.shape[0])
                wait_ranks.append(rank)

            if not xs:
                break
            # Merge the collected tensors to process them at once.
            xs = torch.cat(xs, dim=0).to("cuda")

            # Process the tensor and split them.
            raw_policies, raw_values = self._process_network(xs, override=True)
            raw_policies = raw_policies.split(sizes, dim=0)
            raw_values = raw_values.split(sizes, dim=0)

            # Distribute the results to each child procs.
            for rank, policies, values in zip(wait_ranks, raw_policies, raw_values):
                policies, values = policies.clone(), values.clone()
                self.scatter_queues[rank].put((policies, values))

        # Aggregate the results.
        history = []
        for rank in range(num_thread):
            h = self.result_queues[rank].get()
            history.extend(h)
            # Now allow children process to terminate.
            self.scatter_queues[rank].put(None)

        # Clean-up.
        for rank, proc in enumerate(processes):
            proc.join()

        # Move the main process's model back to cpu
        # to mitigate the CUDA bug associated with multiprocessing.
        self.model.to("cpu")

        # Switch MCTS back to non-parallel mode.
        self.is_parallel = False

        return history

    def _self_play_with_queues(self, num_play, rank):
        """Execute self-play with multiprocessing queues for parallelism.

        This method is run by child process.

        Parameters
        ----------
        num_play : int
            The number of self-play to be performed by this child process.
        rank : int
            The rank (index) of the current child process.
        """
        self.rank = rank
        history = self.self_play(num_play)
        self.result_queues[rank].put(history)

        # Signal to the main process that this child process has completed its job.
        self.gather_queues[self.rank].put(None)

        # Wait for the main process to send `None`
        # as an indication that the child process can terminate.
        assert self.scatter_queues[self.rank].get() is None

    def _play_internal(self, node):
        """Choose the next move based on MCTS evaluation and Dirichlet noise.

        Evaluates the MCTS node for `num_evaluation` times and
        chooses the next move probabilistically, where the probability is
        boltzmann-scaled MCTS scores.
        Dirichlet noise is added to the to the prior probabilities in the root node
        to encourage exploration.

        Parameters
        ----------
        node : Node
            The current MCTS node.

        Returns
        -------
        move : int
            The chosen move.
        scores : list[float]
            The move probabilities based on MCTS evaluation (boltzmann-scaled).
        next_node : Node
            The next MCTS node after applying the chosen move, corresponding to
            the child node associated with the chosen move.
        """
        # Ensure the node has not been evaluated for `num_evaluation` times yet.
        assert node.n < self.num_evaluation
        # Dirichlet noise is added to root node to encourage exploration.
        node.add_dirichlet_noise()
        # Perform MCTS evaluations for the remaining times until `num_evaluation` is reached.
        for _ in range(self.num_evaluation - node.n):
            node.evaluate()
        assert node.n == self.num_evaluation

        # Calculate the MCTS scores.
        scores = node.get_scores(temperature=1.0 if node.board.nth_move < self.temp_threshold else 0.0)

        # Choose the move probabilistically based on the scores.
        idx = np.random.choice([i for i in range(len(node.leg_moves))], p=scores)
        move = node.leg_moves[idx]
        next_node = node.child_nodes[idx]
        return move, scores, next_node

    def _self_play_once(self):
        """Execute a single game of self-play.

        Returns
        -------
        history : list
            A list of game states, policies, and values from self-play.
        """
        # history = []
        # turn = 0
        # node = Node(board=self.board_class(), mcts=self)
        # while True:
        #     if node.board.has_lost():
        #         value = -1
        #         break
        #     elif node.board.has_drawn():
        #         value = 0
        #         break

        #     _, scores, next_node = self._play_internal(node)

        #     policy = torch.zeros(node.board.move_size)
        #     policy[node.board.legal_moves()] = torch.tensor(scores, dtype=torch.float)

        #     # (Board, Policy, Value)
        #     # Value will be determined after the game is over.
        #     history.append([node.board.to_tensor(), policy, None])

        #     # Step.
        #     node = next_node
        #     turn ^= 1

        # if turn & 1:
        #     value = -value

        # for i in range(len(history)):
        #     history[i][2] = value
        #     value = -value
        # return history


        history = []
        node = Node(board=self.board_class(), mcts=self)
        while True:
            if node.board.has_lost():
                policy = torch.zeros(node.board.move_size)
                policy[-1] = 1.0
                history.append([node.board.to_tensor(), policy, -1])
                break
            elif node.board.has_drawn():
                policy = torch.zeros(node.board.move_size)
                policy[-1] = 1.0
                history.append([node.board.to_tensor(), policy, 0])
                break

            _, scores, next_node = self._play_internal(node)

            policy = torch.zeros(node.board.move_size)
            policy[node.board.legal_moves()] = torch.tensor(scores, dtype=torch.float)

            # (Board, Policy, Value)
            # Value will be determined after the game is over.
            history.append([node.board.to_tensor(), policy, None])

            # Step.
            node = next_node

        for i in range(len(history) - 2, -1, -1):
            history[i][2] = -history[i+1][2]
        return history


    def _process_network(self, x, override=False):
        """Process input tensor using the neural network model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to process using the neural network model.
        override : boolean
            If `True`, process the tensor eagerly without utilizing parallelism.

        Returns
        -------
        policies : torch.Tensor
            A tensor of move probability distributions for each input tensor.
        values : torch.Tensor
            A tensor of board state values for each input tensor.
        """
        if self.is_parallel and not override:
            # In parallel mode, pass the input tensor to the main process
            # and wait to receive the processed results.
            self.gather_queues[self.rank].put(x)
            return self.scatter_queues[self.rank].get()
        else:
            # Process the tensor (this branch is executed within the main process).
            x = x.to("cuda")
            policies, values = self.model(x)
            policies, values = policies.to("cpu"), values.to("cpu")
            return policies, values


def write_history(history, generation: int, subproc_no: int, path: str):
    """Write the history of self-play to a file.

    Parameters
    ----------
    history : list
        The list of game states, policies and values.
    generation : int
        Current generation number.
    subproc_no : int
        The id number of subprocess that generated the history.
    path : str
        The directory where the history file will be saved.
    """
    os.makedirs(path, exist_ok=True)
    now = datetime.now()
    timestamp = f"{now.year:04}{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}{now.second:02}"
    nonce = os.urandom(4).hex()
    filename = f"history_generation-{generation:06}_{subproc_no:02}_{timestamp}_{nonce}.pkl"
    with open(os.path.join(path, filename), "wb") as f:
        pickle.dump(history, f)


def read_history(generation: int, subproc_no: int, path: str):
    """Read the history of self-play from a file.

    Parameters
    ----------
    generation : int
        The generation number of the history to read.
    subproc_no : int
        The id number of subprocess that generated the history.
    path : str
        The directory where the history file is stored.

    Returns
    -------
    list
        The list of game states, policies and values.
    """
    filenames = glob.glob(os.path.join(path, f"history_generation-{generation:06}_{subproc_no:02}*.pkl"))
    assert len(filenames) == 1, "There should be only one history per generation."
    with open(filenames[-1], "rb") as f:
        return pickle.load(f)
