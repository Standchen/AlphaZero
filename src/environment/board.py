from abc import *


class Board(metaclass=ABCMeta):
    """Abstract class for subclasses implementing board environments.

    Note
    ----
    Subclass must have the following instance variables:
    - `nth_move`
    - `move_size`
    """

    @abstractmethod
    def has_ended(self):
        """Return a boolean indicating whether the game has ended."""
        pass

    @abstractmethod
    def has_won(self):
        """Return a boolean indicating whether the game has won."""
        pass

    @abstractmethod
    def has_lost(self):
        """Return a boolean indicating whether the game has lost."""
        pass

    @abstractmethod
    def has_drawn(self):
        """Return a boolean indicating whether the game has drawn."""
        pass

    @abstractmethod
    def legal_moves(self):
        """Return a list containing legal moves.

        Ensure that the returned list is not modified,
        since this method may cache the returning list for optimization purposes.
        """
        pass

    @abstractmethod
    def play(self, move):
        """Return a new board object reflecting the play of `move`."""
        pass

    @abstractmethod
    def to_tensor(self):
        """Return a `torch.tensor` object that can be directly fed into the neural network."""
        pass

    @abstractmethod
    def __str__(self):
        """Return a string representation of the board's shape and arrangement of pieces."""
        pass
