# AlphaZero General
A PyTorch implementation of DeepMind's AlphaZero[1], a self-play reinforcement learning algorithm to master various board games without prior human knowledge.
Unlike its predecessor, AlphaGo[2], which relied on human-generated data and domain-specific knowledge, AlphaGo Zero[3] trains its model solely on reinforcement learning from self-play, and results in better agent outperforming the precedent RL algorithms. AlphaZero expanded on this success by generalizing the capability to not only play Go but also other complex board games like chess and Shogi at superhuman levels, demonstrating that the approach taken by AlphaGo Zero could be adapted with minimal modifications to excel in other games.  
From AlphaGo to AlphaZero, the key is the utilization of MCTS (Monte Carlo Tree Search), enhanced by neural networks to predict the most promising moves and the likely winner of the games. These neural networks are iteratively trained on self-play data to better predict and produce high-quality outcomes in subsequent iterations.

This implementation offers flexibility, allowing the training and playing of games other than those already implemented (othello, tictactoe, and pentago). A key feature in this project is the highly optimized MCTS process engine *(batched MCTS)*. This leverages the parallel processing capabilities of GPUs, enabling it to perform self-play approximately *3x* faster by processing games in parallel.

# Optimization
## Batched MCTS
This optimization enhances the efficiency of generating self-play history for training and testing, by allowing for the concurrent execution of multiple agent processes to parallelize the procedure. Rather than processing each agent's data independently, this optimization aggregates tensors from multiple processes and batch them for GPU processing. This batched approach allows the GPU to process tensors more efficiently compared to sequential processing of individual, fine-grained tensors.

In this optimized framework, [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) is employed to manage concurrent process execution, streamlining the gathering and scattering of tensors across the processes. (In the code, these multiprocessing processes are referred to as `threads` to differentiate them from subprocesses, although this terminology is abused and used loosely for clarity.)  
Also, it is empirically found that deploying multiple instances—each is a [subprocess](https://docs.python.org/3/library/subprocess.html) spawned by the main process, running a set of aforementioned `threads` in it— results in better speed and GPU utilization compared to simply increasing the number of `threads` within a single instance.

The number of subprocesses (`num_subproc`) and the number of multiprocessing processes (`num_thread`) are configurable in `Trainer` class.

# Directory Structure
- **/data**
  - **/data/history**: Contains pickled self-play history data.
  - **/data/tmp**: Contains pickled metadata for subprocess.

- **/models**: Contains model checkpoints.

- **/src**
  - **/src/environment**: Provides game environments (othello, tictactoe, and pentago). See `board.py` for interface to implement your own environment.
  - **/src/mcts**: MCTS-related codes.
  - **/src/mcts/network/**: Neural network.
  - **/src/strategy/strategy.py**: Implements various play strategies (random, alpha-beta pruning, MCS).


# Train from scratch
To train the model from scratch with default configuration on Othello, run:
```
./src/train.sh
```
You can also customize the configuration. For example,
```
from functools import partial

from environment.othello import OthelloBoard
from training import Trainer
from mcts.network.network import OthelloDualNetwork

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
        num_evaluation=25,
        temp_threshold=float("inf"),
        dirichlet_alpha=0.0,
        dirichlet_weight=0.0,
    )

    losses, policy_losses, value_losses = trainer.train()
```

Used `python 3.10.13`, `torch==2.0.0+cu118` to train and test the model.

# TODO
- [ ] Add learning rate scheduler
- [ ] Add temperature scheduler
- [ ] Add buffered queue of recent self-play history
(To prevent overfitting, encourage exploration, and smooth learning over time.)
- [ ] Implement interactive playing interface

# References
[1] Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & others. (2017). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. *arXiv preprint arXiv:1712.01815*. Retrieved from https://arxiv.org/abs/1712.01815  
[2] Silver, D., Huang, A., Maddison, C. _et al._ Mastering the game of Go with deep neural networks and tree search. _Nature_ **529**, 484–489 (2016). https://doi.org/10.1038/nature16961  
[3] Silver, D., Schrittwieser, J., Simonyan, K. _et al._ Mastering the game of Go without human knowledge. _Nature_ **550**, 354–359 (2017). https://doi.org/10.1038/nature24270  
