from copy import deepcopy

import torch

from .board import Board


class TictactoeBoard(Board):
    """A subclass of `Board` that implements the tic-tac-toe game environment.

    Attributes
    ----------
    pieces : list[list[int]]
        A nested list representing the pieces on the board.
    nth_move : int
        The index of current board (starting from 0).
    turn : int
        A value of either 0 or 1 indicating the current player's turn.
        (0 - First player, 1 - Second player)
    move_size : int
        A fixed constant 9.
    """

    def __init__(self, pieces=None, nth_move=0):
        self.pieces = pieces if pieces is not None else [[0 for _ in range(9)] for _ in range(2)]
        self.nth_move = nth_move
        self.turn = nth_move & 1
        self.move_size = 9

    def has_ended(self):
        return self.has_won() or self.has_lost() or self.has_drawn()

    def has_won(self):
        return self._judge(self.pieces[self.turn])

    def has_lost(self):
        return self._judge(self.pieces[self.turn ^ 1])

    def has_drawn(self):
        return not self.has_won() and not self.has_lost() and sum(self.pieces[0]) + sum(self.pieces[1]) == 9

    def legal_moves(self):
        return [i for i in range(9) if not self.pieces[0][i] and not self.pieces[1][i]]

    def play(self, move):
        # Ensure a copy of pieces is used to prevent unintended side effects.
        new_pieces = deepcopy(self.pieces)
        new_pieces[self.turn][move] = 1
        return TictactoeBoard(new_pieces, self.nth_move + 1)

    def to_tensor(self):
        # A 3-channel tensor is passed as input to the neural network.
        #
        # The first two layers are binary representations of
        # the current player's pieces and the opponent's pieces.
        #
        # The last layer is filled with a value indicating the current player's turn.
        x = torch.tensor([self.pieces[self.turn], self.pieces[self.turn ^ 1]], dtype=torch.float)
        x = x.reshape(2, 3, 3)
        t = torch.full((1, 3, 3), fill_value=self.turn)
        x = torch.cat((x, t))
        return x.unsqueeze(0)

    def _judge(self, piece):
        """Return a boolean indicating whether the player having `piece` has won or not."""
        # Horizontal
        for x in range(3):
            if all([piece[3 * x + y] for y in range(3)]):
                return True
        # Vertical
        for y in range(3):
            if all([piece[3 * x + y] for x in range(3)]):
                return True
        # Diagonal (\)
        if piece[0] and piece[4] and piece[8]:
            return True
        # Diagonal (/)
        if piece[2] and piece[4] and piece[6]:
            return True
        return False

    def __str__(self):
        separator = "\n+---+---+---+\n"
        s = separator
        for x in range(3):
            s += "|"
            for y in range(3):
                idx = 3 * x + y
                if self.pieces[0][idx]:
                    s += " O "
                elif self.pieces[1][idx]:
                    s += " X "
                else:
                    s += "   "
                s += "|"
            s += separator
        return s.strip()
