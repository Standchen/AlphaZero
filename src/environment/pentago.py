###################################################################
# THIS CODE MAY CONTAIN BUGS
# AND THE GAME IS NOT EXPECTED TO BE SOLVABLE ON TYPICAL MACHINES.
###################################################################

from copy import deepcopy

import torch

from .board import Board


class PentagoBoard(Board):
    """A subclass of `Board` that implements the pentago game environment.

    Attributes
    ----------
    size: int
        An even integer value that represents the size of the board.
    pieces : list[list[int]]
        A nested list representing the pieces on the board.
    nth_move : int
        The index of current board (starting from 0).
    turn : int
        An integer value of either 0 or 1 indicating the current player's turn.
        (0 - First player, 1 - Second player)
    move_size : int
        Set to `size**2 + 1`. where 1 represents a pass move.
    """

    def __init__(self, size=6, pieces=None, nth_move=0):
        assert size & 1 == 0
        self.size = size
        self.pieces = pieces if pieces is not None else [[0 for _ in range(self._board_area)] for _ in range(2)]
        self.nth_move = nth_move
        self.turn = nth_move & 1

        self._board_area = size**2

        # Caches
        self._ended_cache = None
        self._won_cache = None
        self._lost_cache = None
        self._drawn_cache = None

        self._lmoves_cache = None

    def has_ended(self):
        if self._ended_cache is None:
            self._ended_cache = self.has_won() or self.has_lost() or self.has_drawn()
        return self._ended_cache

    def has_won(self):
        if self._won_cache is None:
            self._won_cache = self._judge(self.pieces[self.turn])
        return self._won_cache

    def has_lost(self):
        if self._lost_cache is None:
            self._lost_cache = self._judge(self.pieces[self.turn ^ 1])
        return self._lost_cache

    def has_drawn(self):
        if self._drawn_cache is None:
            self._drawn_cache = (
                sum(self.pieces[0]) + sum(self.pieces[1]) == self._board_area
                and not self.has_won()
                and not self.has_lost()
            )
        return self._drawn_cache

    def legal_moves(self):
        if self._lmoves_cache is None:
            lmoves = []
            for i in range(self._board_area):
                if not self.pieces[0][i] and not self.pieces[1][i]:
                    for j in range(8):
                        lmoves.append(i + j * self._board_area)
            self._lmoves_cache = lmoves
        return self._lmoves_cache

    def play(self, move):
        pos = move % self._board_area
        r = move // self._board_area

        x, y = pos // self.size, pos % self.size
        quadrant, direction = r % 4, r // 4

        # Ensure a copy of pieces is used to prevent unintended side effects.
        new_pieces = deepcopy(self.pieces)
        assert new_pieces[self.turn][self.size * x + y] == 0
        new_pieces[self.turn][self.size * x + y] = 1

        # Create new board and rotate it manually.
        # Since it needs to be rotated in current player's perspective,
        # the turn of `new_board` is temporarily switched while it's being rotated,
        # and then recovered.
        new_board = PentagoBoard(size=self.size, pieces=new_pieces, nth_move=self.nth_move + 1)
        new_board.turn ^= 1
        new_board._rotate(quadrant=quadrant, direction=direction)
        new_board.turn ^= 1
        return new_board

    def to_tensor(self):
        # A 3-channel tensor is passed as input to the neural network.
        #
        # The first two layers are binary representations of
        # the current player's pieces and the opponent's pieces.
        #
        # The last layer is filled with a value indicating the current player's turn.
        x = torch.tensor([self.pieces[self.turn], self.pieces[self.turn ^ 1]], dtype=torch.float)
        t = torch.full((1, self.size, self.size), fill_value=self.turn)
        x = torch.cat((x, t))
        return x.unsqueeze(0)

    def _rotate(self, quadrant: int, direction: int):
        """
        Rotate the specified quadrant of the board in the given direction.

        The board is divided into 4 quadrants, numbered as follows:
        0 | 2
        -----
        1 | 3

        Parameters
        ----------
        quadrant : int
            The quadrant number (0, 1, 2, or 3) to be rotated.
        direction : int
            The direction of rotation:
            - 0: Clockwise
            - 1: Counter-clockwise

        Raises
        ------
        ValueError
            If the direction is not 0 or 1.
        """
        assert 0 <= quadrant < 4
        h = self.size // 2
        px = h * (quadrant & 1)
        py = h * ((quadrant & 0b10) >> 1)
        tmp = [self.pieces[self.turn][self.size * (px + x) + (py + y)] for x in range(h) for y in range(h)]

        if direction == 0:
            for x in range(h):
                for y in range(h):
                    self.pieces[self.turn][self.size * (px + x) + (py + y)] = tmp[h * (h - 1 - y) + x]
        elif direction == 1:
            for x in range(h):
                for y in range(h):
                    self.pieces[self.turn][self.size * (px + x) + (py + y)] = tmp[h * y + (h - 1 - x)]
        else:
            raise ValueError

    def _judge(self, piece):
        """Check if the player with the given `piece` has won.

        Parameters
        ----------
        piece : list[int]
            A list of integers repreenting certain player's pieces.

        Returns
        -------
        bool
            True if the player with the given `piece` has won, False otherwise.
        """
        # Horizontal
        for x in range(self.size):
            for y in range(self.size - 4):
                if self._judge_direction(piece, x, y, 0, 1):
                    return True
        # Vertical
        for x in range(self.size - 4):
            for y in range(self.size):
                if self._judge_direction(piece, x, y, 1, 0):
                    return True
        # Diagonal (\)
        for x in range(self.size - 4):
            for y in range(self.size - 4):
                if self._judge_direction(piece, x, y, 1, 1):
                    return True
        # Diagonal (/)
        for x in range(self.size - 4):
            for y in range(4, self.size):
                if self._judge_direction(piece, x, y, 1, -1):
                    return True
        return False

    def _judge_direction(self, piece, x, y, dx, dy):
        """Determine if the player with the given `piece` has won in a specific direction.

        Parameters
        ----------
        x : int
            The x-coordinate of the starting position.
        y : int
            The y-coordinate of the starting position.
        dx : int
            The x-coordinate direction to check.
        dy : int
            The y-coordinate direction to check.

        Returns
        -------
        bool
            True if the player with the given `piece` has won in the specified direction, False otherwise.
        """
        i, j = x, y
        for _ in range(5):
            if not (0 <= i < self.size and 0 <= j < self.size):
                return False
            if piece[i][j] != 1:
                return False
            i += dx
            j += dy
        return True

    def __str__(self):
        separator = "\n+" + ("---+" * self.size) + "\n"
        s = separator
        for x in range(self.size):
            s += "|"
            for y in range(self.size):
                if self.pieces[0][self.size * x + y]:
                    s += " O "
                elif self.pieces[1][self.size * x + y]:
                    s += " X "
                else:
                    s += "   "
                s += "|"
            s += separator
        return s.strip()
