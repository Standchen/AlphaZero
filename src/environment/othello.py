from copy import deepcopy
import functools

import torch

from .board import Board


class OthelloBoard(Board):
    """A subclass of `Board` that implements the othello game environment.

    Attributes
    ----------
    size: int
        An even integer value that represents the size of the board.
    pieces : list[list[list[int]]]
        A nested list representing the pieces on the board.
    nth_move : int
        The index of current board (starting from 0).
    turn : int
        An integer value of either 0 or 1 indicating the current player's turn.
        (0 - First player, 1 - Second player)
    prev_passed : bool
        A boolean indicating whether the opponent passed on their turn.
    move_size : int
        Set to `size**2 + 1`. where 1 represents a pass move.
    """

    def __init__(self, size, pieces=None, nth_move=0, prev_passed=False):
        assert size & 1 == 0, f"Othello board size must be even, but found {size}."
        self.size = size

        if pieces is not None:
            self.pieces = pieces
        else:
            # Initial board of othello looks like:
            # .......................
            # ...+---+---+---+---+...
            # ...|   |   |   |   |...
            # ...+---+---+---+---+...
            # ...|   | X | O |   |...
            # ...+---+---+---+---+...
            # ...|   | O | X |   |...
            # ...+---+---+---+---+...
            # ...|   |   |   |   |...
            # ...+---+---+---+---+...
            # .......................
            h = size // 2
            self.pieces = [[[0 for _ in range(size)] for _ in range(size)] for _ in range(2)]
            self.pieces[0][h - 1][h] = self.pieces[0][h][h - 1] = 1
            self.pieces[1][h - 1][h - 1] = self.pieces[1][h][h] = 1

        # No two pieces can be placed at same position.
        assert all(
            not (self.pieces[0][x][y] and self.pieces[1][x][y]) for x in range(self.size) for y in range(self.size)
        )

        self.nth_move = nth_move
        self.turn = nth_move & 1
        self.prev_passed = prev_passed

        self.move_size = (self.size**2) + 1

        # (sum(# of first player pieces), sum(# of second player pieces))
        self._counts = (
            sum(sum(el) for el in self.pieces[0]),
            sum(sum(el) for el in self.pieces[1]),
        )

        # Pass move
        self._pass_move = self.size**2

    @functools.cache
    def has_ended(self):
        if self._counts[0] + self._counts[1] == self.size**2:
            return True
        elif self._counts[0] == 0 or self._counts[1] == 0:
            return True
        elif self.prev_passed and self.legal_moves()[0] == self._pass_move:
            return True
        return False

    def has_won(self):
        if not self.has_ended():
            return False
        return self._counts[self.turn] > self._counts[self.turn ^ 1]

    def has_lost(self):
        if not self.has_ended():
            return False
        return self._counts[self.turn] < self._counts[self.turn ^ 1]

    def has_drawn(self):
        if not self.has_ended():
            return False
        return self._counts[self.turn] == self._counts[self.turn ^ 1]

    @functools.cache
    def legal_moves(self):
        lmoves = [self.size * x + y for x in range(self.size) for y in range(self.size) if self._is_legal_move(x, y)]
        if not lmoves:
            lmoves = [self._pass_move]
        return lmoves

    def play(self, move):
        # Ensure a copy of pieces is used to prevent unintended side effects.
        new_pieces = deepcopy(self.pieces)

        if move == self._pass_move:
            return OthelloBoard(size=self.size, pieces=new_pieces, nth_move=self.nth_move + 1, prev_passed=True)
        else:
            x, y = move // self.size, move % self.size
            assert new_pieces[self.turn][x][y] == 0
            new_pieces[self.turn][x][y] = 1

            # Create new board and flip it manually.
            # Since it needs to be flipped in current player's perspective,
            # the turn of `new_board` is temporarily switched while it's being flipped,
            # and then recovered.
            new_board = OthelloBoard(size=self.size, pieces=new_pieces, nth_move=self.nth_move + 1, prev_passed=False)
            new_board.turn ^= 1
            new_board._flip(x, y)
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
        x = x.reshape(2, self.size, self.size)
        t = torch.full((1, self.size, self.size), fill_value=self.turn)
        x = torch.cat((x, t))
        return x.unsqueeze(0)

    def _is_legal_direction(self, x, y, dx, dy):
        """Determine whether the given direction is legal for placing a piece.

        Check if there is a legal sequence of opponent's pieces between the
        current player's pieces in the specified direction.

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
        tuple or None
            Return the position (i, j) of the other end piece if the direction is legal.
            If no such position exists, returns `None`.
        """
        # Out of board
        if not (0 <= x + dx < self.size and 0 <= y + dy < self.size):
            return None

        # There should be at least one piece of the opponent
        # between the current player's pieces.
        if self.pieces[self.turn ^ 1][x + dx][y + dy] != 1:
            return None

        i, j = x + 2 * dx, y + 2 * dy
        while 0 <= i < self.size and 0 <= j < self.size:
            if self.pieces[self.turn][i][j] == 1:
                return i, j
            elif self.pieces[self.turn][i][j] == 0 and self.pieces[self.turn ^ 1][i][j] == 0:
                return None
            i += dx
            j += dy
        return None

    def _is_legal_move(self, x, y):
        """Determine whether the given position is legal for placing a piece.

        Parameters
        ----------
        x : int
            The x-coordinate of the starting position.
        y : int
            The y-coordinate of the starting position.
        """
        # If already occupied, not a valid move.
        if self.pieces[0][x][y] or self.pieces[1][x][y]:
            return False

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if self._is_legal_direction(x, y, dx, dy) is not None:
                    return True
        return False

    def _flip(self, x, y):
        """Flip opponent's pieces between the current player's pieces in all valid directions.

        Parameters
        ----------
        x : int
            The x-coordinate of the starting position.
        y : int
            The y-coordinate of the starting position.
        """
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (tp := self._is_legal_direction(x, y, dx, dy)) is not None:
                    i, j = tp
                    x_it, y_it = x + dx, y + dy
                    while x_it != i or y_it != j:
                        assert self.pieces[self.turn][x_it][y_it] == 0
                        assert self.pieces[self.turn ^ 1][x_it][y_it] == 1
                        self.pieces[self.turn][x_it][y_it] = 1
                        self.pieces[self.turn ^ 1][x_it][y_it] = 0
                        x_it += dx
                        y_it += dy

        # Update counts.
        self._counts = (
            sum(sum(el) for el in self.pieces[0]),
            sum(sum(el) for el in self.pieces[1]),
        )

    def __str__(self):
        separator = "\n+" + ("---+" * self.size) + "\n"
        s = separator
        for x in range(self.size):
            s += "|"
            for y in range(self.size):
                if self.pieces[0][x][y]:
                    s += " O "
                elif self.pieces[1][x][y]:
                    s += " X "
                else:
                    s += "   "
                s += "|"
            s += separator
        return s.strip()
