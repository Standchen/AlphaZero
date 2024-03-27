import math
import random

import numpy as np
import torch


def my_argmax(collection):
    """Find the index of the maximum value in the given collection.

    Iterate through the collection to find the maximum value
    and its corresponding index.
    If all the elements in the collection are equal to negative infinity,
    a random index is returned.

    Parameters
    ----------
    collection : Iterable
        An iterable collection of numerical values (e.g. list, tuple, or numpy array).

    Returns
    -------
    int
        The index of the maximum value in the collection. If all values are equal to
        negative infinity, a random index is returned.
    """
    max_val, max_i = -float("inf"), None
    for i, val in enumerate(collection):
        if val > max_val:
            max_val = val
            max_i = i
    if max_i is None:
        return random.randint(0, len(collection) - 1)
    return max_i


class Node:
    """A node in the Monte Carlo Tree Search (MCTS) representing a game state.

    Each node represents a game state and stores the associated board, legal moves,
    visit counts, and other relevant information.
    Nodes are expanded and evaluated during MCTS.

    Attributes
    ----------
    board : Board
        The game board associated with this node.
    leg_moves : list
        A list of legal moves available from the game state.
    w : float
        The total action value of the node.
    n : int
        The visit count of the node.
    policy : torch.Tensor
        The policy vector output by the neural network is first
        filtered to include only legal moves, and then re-normalized
        to ensure that the probabilities sum up to 1.
    value : float
        The value estimate output by the neural network.
    child_nodes : list[Node] or None
        A list of child nodes or `None` if not expanded yet.
    mcts : MCTS
        A reference to the MCTS instance it is belonged to.
    """

    def __init__(self, board, mcts, policy=None, value=None):
        self.board = board
        self.leg_moves = board.legal_moves()

        self.w = 0
        self.n = 0

        self.mcts = mcts

        self.policy = policy
        self.value = value

        self.child_nodes = None

        # For root node, initialize `value` and `policy` manually.
        if self.policy is None or self.value is None:
            assert self.policy is None and self.value is None
            policies, values = self.mcts._process_network(self.board.to_tensor())
            self.policy, self.value = policies[0], values.squeeze(1).tolist()[0]

        # Re-normalize the policy to include only legal moves.
        self.policy = self.policy[self.leg_moves]
        self.policy /= self.policy.sum()

    def evaluate(self):
        """Evaluate the node by applying the MCTS process recursively.

        There are three branches of evaluation:
        1. If the game has reached a terminal state, determine the exact value.
        2. If child nodes exist, evaluate the value using one of them.
        3. If child nodes have not been expanded yet, expand them.

        After evaluating the value, update the current node's total action value
        and increase the visit count by 1.

        Returns
        -------
        float
            The value of the node after evaluation.
        """
        if self.board.has_lost():
            value = -1.0
        elif self.board.has_drawn():
            value = 0.0
        elif self.child_nodes is not None:
            value = -self._get_next_child().evaluate()
        else:
            self._expand()
            value = self.value

        self.w += value
        self.n += 1
        return value

    def get_scores(self, temperature: float):
        """Compute Boltzmann-scaled action probabilities based on visit counts.

        Parameters
        ----------
        temperature : float
            The temperature parameter used for Boltzmann scaling.

        Returns
        -------
        numpy.ndarray
            The Boltzmann-scaled action probabilities.
        """
        assert self.child_nodes is not None
        ns = [child.n for child in self.child_nodes]
        if temperature == 0:
            # For zero-temperature,
            # assign all probability to the action with the highest visit count.
            res = np.zeros(len(ns))
            res[my_argmax(ns)] = 1
        else:
            res = np.array([n ** (1 / temperature) for n in ns])
            res /= res.sum()
        return res

    def _expand(self):
        """Expand the node by generating child nodes for each legal move."""
        assert self.child_nodes is None

        # Generate child boards for each legal move,
        # and compute policy / value pairs for each of them.
        child_boards = [self.board.play(move) for move in self.leg_moves]
        child_tensors = torch.cat([board.to_tensor() for board in child_boards], dim=0)
        child_policies, child_values = self.mcts._process_network(child_tensors)

        # Convert to list.
        child_values = child_values.squeeze(1).tolist()

        # Instantiate child nodes.
        assert len(child_boards) == len(child_policies) == len(child_values)
        self.child_nodes = [
            Node(
                board=child_board,
                mcts=self.mcts,
                policy=child_policy,
                value=child_value,
            )
            for child_board, child_policy, child_value in zip(child_boards, child_policies, child_values)
        ]

    def _get_next_child(self):
        """Select a child node to explore using the PUCT (Polynomial Upper Confidence Trees) algorithm."""
        t = sum(child.n for child in self.child_nodes)
        if t == 0:
            return random.choice(self.child_nodes)

        pucb_scores = []
        for i, child in enumerate(self.child_nodes):
            action_value = (-child.w / child.n) if child.n != 0 else 0.0
            bias = self.mcts.c_puct * self.policy[i] * (math.sqrt(t) / (1 + child.n))
            pucb_scores.append(action_value + bias)
        return self.child_nodes[my_argmax(pucb_scores)]

    def add_dirichlet_noise(self):
        if self.mcts.dirichlet_weight:
            dirichlet_noise = torch.distributions.dirichlet.Dirichlet(
                torch.full_like(self.policy, fill_value=self.mcts.dirichlet_alpha)
            ).sample()
            self.policy = (1 - self.mcts.dirichlet_weight) * self.policy + self.mcts.dirichlet_weight * dirichlet_noise
