from functools import partial

import torch.multiprocessing as mp

from mcts.mcts import *


class Fight:
    """A class to facilitate a competition between two players using different strategies in a board game.

    Strategy can be either an `MCTS` instance or a callable (`Board -> int`).

    Parameters
    ----------
    board_class : class
        The board class or a callable for instantiating it.
    num_fight : int
        The number of fights to be performed between the two players.
    play_0 : `MCTS` or callable
        The first player's strategy, which can be an `MCTS` instance or a callable.
    play_1 : `MCTS` or callable
        The second player's strategy, which can be an `MCTS` instance or a callable.
    num_thread : int, optional, default=10
        The number of threads to be used for parallelizing the fights.
    verbose_level : int, optional, default=0
        The verbosity level of the output.
        - 0: Print only the win counts.
        - 1: Print win/lose outcomes for each fight.
        - 2: Print board notations for every move in each fight.
    """

    def __init__(
        self,
        board_class,
        num_fight,
        play_0,
        play_1,
        num_thread=10,
        verbose_level=0,
    ):
        self.board_class = board_class
        self.num_fight = num_fight

        self.play_0 = play_0
        self.play_1 = play_1

        self.num_thread = num_thread
        self.verbose_level = verbose_level

        self.name_0 = self._play_name(play_0)
        self.name_1 = self._play_name(play_1)

    def fight(self):
        """Conduct fights between the two players and return the win counts for each player.

        Draws are considered as 0.5 points for both players.
        If `num_thread > 1`, multiprocessing is utilized to exploit parallelism.

        Returns
        -------
        float
            The number of wins (including draws as 0.5) for player 0.
        float
            The number of wins (including draws as 0.5) for player 1.
        """
        if self.num_thread == 1:
            # If no parallelism, runs in the main process.
            win_count_0, win_count_1, output = self._fight(num_play=self.num_fight)
        else:
            # If parallelism is enabled, spawn child processes.
            play_per_thread = [self.num_fight // self.num_thread] * self.num_thread
            play_per_thread[-1] += self.num_fight % self.num_thread

            mp.set_sharing_strategy("file_system")
            ctx = mp.get_context("spawn")

            pool = ctx.Pool(processes=self.num_thread)
            res = pool.map(self._fight, play_per_thread)
            pool.close()
            pool.join()

            # Aggregate the results.
            win_count_0, win_count_1, output = 0.0, 0.0, []
            for wc_0, wc_1, out in res:
                win_count_0 += wc_0
                win_count_1 += wc_1
                output.extend(out)

        # Print the output & results.
        output = "\n".join(output)
        print(output)

        print(f"[*] {self.name_0}: {100 * win_count_0 / self.num_fight:.2f}%")
        print(f"[*] {self.name_1}: {100 * win_count_1 / self.num_fight:.2f}%")

        return win_count_0, win_count_1

    def _fight(self, num_play):
        """A simple wrapper for `_fight_no_grad` method with `torch.no_grad()` context manager."""
        with torch.no_grad():
            return self._fight_no_grad(num_play=num_play)

    def _fight_no_grad(self, num_play):
        """Child processes execute this method to perform fights between the two players for `num_play` times.
        Gradient calculation is disabled.

        Parameters
        ----------
        num_play : int
            The number of fights to be performed by this child process.

        Returns
        -------
        float
            The number of wins (including draws as 0.5) for player 0.
        float
            The number of wins (including draws as 0.5) for player 1.
        list[str]
            The list of strings for verbose output.
        """
        playfunc_0 = self._play_to_function(self.play_0)
        playfunc_1 = self._play_to_function(self.play_1)

        output = []
        win_count_0, win_count_1 = 0, 0
        for i in range(num_play):
            # Alternate first player for each fight.
            turn = i & 1
            board = self.board_class()
            if self.verbose_level >= 2:
                if turn & 1:
                    first_player, second_player = self.name_1, self.name_0
                else:
                    first_player, second_player = self.name_0, self.name_1
                output.append(f"[*] {first_player}: O")
                output.append(f"[*] {second_player}: X")

            # Fight.
            while True:
                if board.has_lost():
                    if turn & 1:
                        if self.verbose_level >= 1:
                            output.append(f"[*] ({i&1}) {self.name_0} Won :)")
                        win_count_0 += 1
                    else:
                        if self.verbose_level >= 1:
                            output.append(f"[*] ({i&1}) {self.name_1} Won :)")
                        win_count_1 += 1
                    break

                elif board.has_drawn():
                    if self.verbose_level >= 1:
                        output.append(f"[*] ({i&1}) Drawn")
                    win_count_0 += 0.5
                    win_count_1 += 0.5
                    break

                if turn & 1:
                    move = playfunc_1(board)
                else:
                    move = playfunc_0(board)

                board = board.play(move)
                turn ^= 1

                if self.verbose_level >= 2:
                    output.append(f"{board}")
                    output.append("-" * 20 + "")
            if self.verbose_level >= 2:
                output.append("#" * 50)

        return win_count_0, win_count_1, output

    def _play_to_function(self, play):
        """Convert the given play strategy into a callable function.

        If the strategy is an `MCTS` instance, return its `play` method.
        If the strategy is already callable, return it as is.

        Parameters
        ----------
        play : `MCTS` or callable
            The player's strategy, which can be an `MCTS` instance or a callable.

        Returns
        -------
        callable
            The callable representing the player's strategy.

        Raises
        ------
        ValueError
            If the play strategy is neither an `MCTS` instance nor callable.
        """
        if isinstance(play, MCTS):
            # The model must reside in CPU before spawning.
            assert not next(play.model.parameters()).is_cuda
            play.model.to("cuda")
            return play.play
        elif callable(play):
            return play
        else:
            raise ValueError

    def _play_name(self, play):
        """Get the name of the given `play` strategy.

        If the strategy is an `MCTS` instance, return `"MCTS"`.
        If the strategy is a `function.partial` object, return its `func.__name__` attribute.
        If the strategy is a function, return its `__name__` attribute.

        Parameters
        ----------
        play : `MCTS` or callable
            The player's strategy, which can be an `MCTS` instance or a callable.

        Returns
        -------
        str
            The name representing the player's strategy.

        Raises
        ------
        ValueError
            If the play strategy is neither an `MCTS` instance nor callable.
        """
        if isinstance(play, MCTS):
            return f"MCTS"
        elif isinstance(play, partial):
            return play.func.__name__
        elif callable(play):
            return play.__name__
        else:
            raise ValueError
