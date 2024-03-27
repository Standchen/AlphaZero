from functools import partial
import subprocess

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim

from environment.tictactoe import TictactoeBoard
from environment.othello import OthelloBoard
from fight import Fight
from mcts.mcts import *
from mcts.network.augment import *
from mcts.network.network import TictactoeDualNetwork, OthelloDualNetwork
from mcts.network.process import *
from strategy.strategy import *


TMP_PATH = "../data/tmp/"
HISTORY_PATH = "../data/history/"
MODEL_PATH = "../models/"


class Trainer:
    """Trainer class for training an AI using Monte Carlo Tree Search (MCTS) and a neural network."""

    def __init__(
        self,
        num_generation: int,
        num_epoch: int,
        lr: float,
        batch_size: int,
        num_play: int,
        num_thread: int,
        num_subproc: int,
        update_per: int,
        num_fight: int,
        board_class,
        model_class,
        load_generation: int | None = None,
        augment_fn: callable = None,
        num_evaluation: int = 50,
        temp_threshold: int | float = float("inf"),
        dirichlet_alpha: float | None = 0.3,
        dirichlet_weight: float = 0.25,
    ):
        """
        Initialize the Trainer class.

        Parameters
        ----------
        num_generation : int
            The number of generations to train.
        num_epoch : int
            The number of epochs to optimize in each generation.
        lr : float
            The learning rate for the optimizer.
        batch_size : int
            The batch size used for training.
        num_play : int
            The number of self-play in each generation.
        num_thread : int
            The number of threads used within a single subprocess.
        num_subproc : int
            The number of subprocesses used to make history.
        update_per : int | None
            The number of generations between best model updates.
            If `None`, skip updating.
        num_fight : int
            The number of fights to perform in each best model update.
        board_class
            The board class or a callable for instantiating it.
        model_class
            The neural network class or a callable for instantiating it.
        load_generation : int | None
            The generation of the model to load for training.
        augment_fn : callable
            The function used for data augmentation.
        num_evaluation : int
            The number of evaluations in the MCTS algorithm.
        temp_threshold : int | float
            The threshold of temperature scheduler for boltzmann scaling.
            It should be either integer or `float("inf")`.
        dirichlet_alpha : float | None
            The Dirichlet noise alpha value.
        dirichlet_weight : float
            The Dirichlet noise weight (0 to 1).
        """
        # Set attributes.
        self.num_generation = num_generation
        self.num_epoch = num_epoch
        self.lr = lr
        self.batch_size = batch_size

        self.num_play = num_play
        self.num_thread = num_thread
        self.num_subproc = num_subproc

        self.update_per = update_per
        self.num_fight = num_fight

        self.board_class = board_class
        self.model_class = model_class
        self.load_generation = load_generation

        self.augment_fn = augment_fn

        self.num_evaluation = num_evaluation
        self.temp_threshold = temp_threshold

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

        # Initialize.
        self.pickle_path = self._subproc_preparation()
        self.model = self._initialize_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        """
        Train the model for the specified number of generations.

        In each generation, it creates history through self-play in multiple subprocesses and trains the model on it.
        Every `update_per` generations, the best model is updated if the current model outperforms it.

        Returns
        -------
        tuple
            A tuple containing the losses, policy_losses, and value_losses.
        """
        if self.num_play % self.num_thread:
            print(f"[*] Warning: Multiprocessing might be unbalanced ({self.num_play = }, {self.num_thread = }).")

        losses, policy_losses, value_losses = [], [], []
        for generation in (pbar := tqdm(range(self.start_generation, self.start_generation + self.num_generation))):
            pbar.set_description(f"Generation {generation}")

            self._make_history(generation)
            history = self._read_history(generation)

            # Optimize model with the history.
            # Make sure the model resides in CPU after optimization.
            self.model.to("cuda")
            loss, policy_loss, value_loss = optimize_model(
                self.model,
                self.optimizer,
                self.num_epoch,
                self.batch_size,
                history,
                self.augment_fn,
            )
            self.model.to("cpu")

            # Save loss information.
            losses.extend(loss)
            policy_losses.extend(policy_loss)
            value_losses.extend(value_loss)

            # Update progress bar with loss info.
            pbar.set_postfix_str(
                f"loss = {loss[-1]:.3f} (policy_loss = {policy_loss[-1]:.3f}, value_loss = {value_loss[-1]:.3f})"
            )

            # Save model
            save_model(self.model, generation=generation, path=MODEL_PATH)

            # Update best model.
            self._update_best_model(generation=generation)

        return losses, policy_losses, value_losses

    def _subproc_preparation(self):
        """Pickle data for subprocessing."""
        os.makedirs(TMP_PATH, exist_ok=True)
        pickle_path = f"{TMP_PATH}/subproc_data.pkl"

        with open(pickle_path, "wb") as f:
            data = {
                "board_class": self.board_class,
                "model_class": self.model_class,
                "num_evaluation": self.num_evaluation,
                "temp_threshold": self.temp_threshold,
                "dirichlet_alpha": self.dirichlet_alpha,
                "dirichlet_weight": self.dirichlet_weight,
                "num_thread": self.num_thread,
            }
            pickle.dump(data, f)
        return pickle_path

    def _initialize_model(self):
        """Initialize the model for training.

        If `load_generation` is specified, load corresponding model weight.
        """
        # Instantiate the model.
        model = self.model_class()

        # Load model weight if specified.
        if self.load_generation is None:
            print("[*] No checkpoint is specified to load. Start from scratch.")
            self.start_generation = 0
        elif isinstance(self.load_generation, int):
            print(f"[*] Generation {self.load_generation} is specified to load.")
            self.start_generation = self.load_generation + 1
            model.load_state_dict(load_model(generation=self.load_generation, path=MODEL_PATH))
        else:
            raise ValueError
        return model

    def _make_history(self, generation):
        """Create a history for the current generation using multiple subprocesses."""
        # First, save current model as temporary one.
        save_model(self.model, generation=generation, attr="tmp", path=MODEL_PATH)

        play_per_subproc = [self.num_play // self.num_subproc] * self.num_subproc
        play_per_subproc[-1] += self.num_play % self.num_subproc

        # Start subprocesses.
        subprocs = []
        for subproc_no in range(self.num_subproc):
            cmd = " ".join(
                [
                    "python3 ./history_maker.py",
                    f"--generation {generation}",
                    f"--subproc-no {subproc_no}",
                    f"--num-play {play_per_subproc[subproc_no]}",
                    f"--pickle-path {self.pickle_path}",
                ]
            )
            subprocs.append(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE))

        # Wait them to terminate.
        for subproc_no, subproc in enumerate(subprocs):
            print(f"[*] Waiting subproc {subproc_no} (Total {self.num_subproc}).")
            subproc.wait()
            print(f"[*] Subproc {subproc_no} finished.")

    def _read_history(self, generation):
        """Read history for given `generation`."""
        history = []
        for subproc_no in range(self.num_subproc):
            h = read_history(generation=generation, subproc_no=subproc_no, path=HISTORY_PATH)
            history.extend(h)
        return history

    def _update_best_model(self, generation):
        """Update the best model if current model performs better."""
        if self.update_per is None:
            pass
        elif generation == 0:
            # The initial model is saved as the best model at first
            save_model(self.model, generation=generation, attr="best", path=MODEL_PATH)
        elif generation != 0 and generation % self.update_per == 0:
            print("[*] Evaluate current model.")

            curr_model = self.model_class()
            curr_model.load_state_dict(self.model.state_dict())
            curr_mcts = MCTS(
                self.board_class,
                curr_model,
                self.num_evaluation,
                self.temp_threshold,
                dirichlet_alpha=None,
                dirichlet_weight=0,
            )

            best_model = self.model_class()
            best_model.load_state_dict(load_model(generation=None, attr="best", path=MODEL_PATH))
            best_mcts = MCTS(
                self.board_class,
                best_model,
                self.num_evaluation,
                self.temp_threshold,
                dirichlet_alpha=None,
                dirichlet_weight=0,
            )

            curr_win, best_win = Fight(
                self.board_class,
                num_fight=self.num_fight,
                play_0=curr_mcts,
                play_1=best_mcts,
                num_thread=10,
            ).fight()

            # If current model wins, update the best model.
            print(f"[*] Last: {100 * curr_win / self.num_fight:.2f}%")
            print(f"[*] Best: {100 * best_win / self.num_fight:.2f}%")

            if curr_win > best_win:
                print(f"[*] Update the best model at generation {generation}")
                save_model(self.model, generation=generation, attr="best", path=MODEL_PATH)
            else:
                print(f"[*] Skip updating the best model at generation {generation}")


if __name__ == "__main__":
    board_class = partial(OthelloBoard, size=6)
    model_class = OthelloDualNetwork

    trainer = Trainer(
        num_generation=80,
        num_epoch=10,
        lr=1e-4,
        batch_size=128,
        num_play=160,
        num_thread=20,
        num_subproc=4,
        update_per=None,
        num_fight=60,
        board_class=board_class,
        model_class=model_class,
        load_generation=None,
        augment_fn=augment_othello,
        num_evaluation=25,
        temp_threshold=float("inf"),
        dirichlet_alpha=0.0,
        dirichlet_weight=0.0,
    )

    losses, policy_losses, value_losses = trainer.train()

    # Plot losses
    it = np.arange(len(losses))
    plt.plot(it, losses, label="Loss")
    plt.plot(it, policy_losses, label="Policy loss")
    plt.plot(it, value_losses, label="Value loss")
    plt.legend(loc="upper right")
    plt.show()

    it = np.arange(len(policy_losses))
    plt.plot(it, policy_losses, label="Policy loss")
    plt.legend(loc="upper right")
    plt.show()

    it = np.arange(len(value_losses))
    plt.plot(it, value_losses, label="Value loss")
    plt.legend(loc="upper right")
    plt.show()

    # Assessment
    model = model_class()
    model.load_state_dict(load_model(generation=None, attr="best", path=MODEL_PATH))

    mcts = MCTS(
        board_class,
        model,
        num_evaluation=trainer.num_evaluation,
        temp_threshold=0,
        dirichlet_alpha=None,
        dirichlet_weight=0,
    )

    num_fight = 120
    play_0 = mcs_play
    play_1 = mcts
    win_count_0, win_count_1 = Fight(
        board_class,
        num_fight=num_fight,
        play_0=play_0,
        play_1=play_1,
        num_thread=10,
        verbose_level=1,
    ).fight()
