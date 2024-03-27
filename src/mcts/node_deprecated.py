from .network.process import process_network
import math
import numpy as np
import random
import torch


def my_argmax(collection):
    max_val, max_i = -float("inf"), None
    for i, val in enumerate(collection):
        if val > max_val:
            max_val = val
            max_i = i
    if max_i is None:
        return random.randint(0, len(collection) - 1)
    return max_i


"""
The code snippet below is too slow.
"""
# def my_argmax(collection):
#     indices = np.where(collection == max(collection))[0]
#     if len(indices) > 1:
#         max_i = random.choice(indices)
#     else:
#         max_i = indices[0]
#     return max_i


class Node:
    def __init__(self, board, p, model=None, c_puct=1.0):
        self.board = board
        self.leg_moves = board.legal_moves()

        self.w = 0
        self.n = 0
        self.p = p

        self.children_nodes = None

        self.model = model
        self.c_puct = c_puct

    def evaluate(self):
        if self.board.has_lost():
            value = -1
        elif self.board.has_drawn():
            value = 0
        elif self.children_nodes is not None:
            value = -self._get_next_child().evaluate()
        else:
            policy, value = process_network(model=self.model, x=self.board.to_tensor())
            self._expand(policy)

        self.w += value
        self.n += 1
        return value

    def get_scores(self, temperature: float):
        return self._boltzmann(temperature)

    def _expand(self, policy):
        assert self.children_nodes is None
        policy = np.array(policy)
        policy = policy[self.leg_moves]
        policy /= policy.sum()
        self.children_nodes = [
            Node(board=self.board.play(move), p=p, c_puct=self.c_puct, model=self.model)
            for move, p in zip(self.leg_moves, policy)
        ]

    def _get_next_child(self):
        t = sum(child.n for child in self.children_nodes)
        if t == 0:
            return random.choice(self.children_nodes)

        pucb_scores = []
        for child in self.children_nodes:
            action_value = (-child.w / child.n) if child.n else 0.0
            bias = self.c_puct * child.p * (math.sqrt(t) / (1 + child.n))
            pucb_scores.append(action_value + bias)
        return self.children_nodes[my_argmax(pucb_scores)]

    """
    The code snippet below is scalabe,
    but it's slower than the naive implementation when the search space is small.
    It outperforms the naive one if child nodes are about 300+.
    """
    # def _get_next_child(self):
    #     ns = np.array([child.n for child in self.children_nodes])
    #     t = ns.sum()
    #     if t == 0:
    #         return random.choice(self.children_nodes)

    #     sqrt_t = math.sqrt(t)
    #     ws = np.array([child.w for child in self.children_nodes])
    #     ps = np.array([child.p for child in self.children_nodes])

    #     action_values = np.divide(-ws, ns, out=np.zeros_like(ws, dtype=float), where=(ns != 0))
    #     biases = self.c_puct * ps * (sqrt_t / (1 + ns))

    #     pucb_scores = action_values + biases
    #     return self.children_nodes[my_argmax(pucb_scores)]

    def _boltzmann(self, temperature):
        assert self.children_nodes is not None
        ns = [child.n for child in self.children_nodes]

        if temperature == 0:
            res = np.zeros(len(ns))
            res[my_argmax(ns)] = 1
        else:
            res = np.array([n ** (1 / temperature) for n in ns])
            res /= sum(res)
        return res
