from datetime import datetime
import glob
import os

import torch
import torch.nn.functional as F


def optimize_model(model, optimizer, num_epoch, batch_size, history, augment_fn):
    """Optimize the given neural network model using the provided history.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be optimized.
    optimizer : torch.optim.Optimizer
        The optimizer used for the optimization process.
    num_epoch : int
        The number of epochs to train the model.
    batch_size : int
        The batch size used for training.
    history : list
        The list of game states, policies, and values used for training.
    augment_fn : callable
        A function to augment the input data.

    Returns
    -------
    tuple
        A tuple containing lists of (total) losses, policy losses, and value losses.
    """
    board_batch, policy_batch, value_batch = zip(*history)

    board_batch = torch.cat(board_batch, dim=0).to("cuda")
    policy_batch = torch.stack(policy_batch, dim=0).to("cuda")
    value_batch = torch.tensor(value_batch, dtype=torch.float).unsqueeze(1).to("cuda")

    # Perform data augmentation if `augment_fn` is provided
    if augment_fn is not None:
        print("[*] Perform augmentation.")
        board_batch, policy_batch, value_batch = augment_fn(board_batch, policy_batch, value_batch)

    data_size = board_batch.shape[0]
    print(f"[*] {data_size = }")

    # Switch model to train mode.
    model.train()

    # Train the model for num_epoch times on the same history.
    losses, policy_losses, value_losses = [], [], []
    for epoch in range(num_epoch):
        # Shuffle
        shuffle_idx = torch.randperm(data_size)
        board_batch = board_batch[shuffle_idx]
        policy_batch = policy_batch[shuffle_idx]
        value_batch = value_batch[shuffle_idx]

        for i in range(0, data_size, batch_size):
            # Fetch data into minibatches.
            board_minibatch = board_batch[i : i + batch_size]
            policy_minibatch = policy_batch[i : i + batch_size]
            value_minibatch = value_batch[i : i + batch_size]

            # Process.
            out_policy, out_value = model(board_minibatch)

            # Calculate losses
            policy_loss = -torch.sum(torch.log(out_policy) * policy_minibatch) / policy_minibatch.shape[0]
            value_loss = F.mse_loss(input=out_value, target=value_minibatch)
            loss = policy_loss + value_loss

            # Perform optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            losses.append(loss := loss.item())
            policy_losses.append(policy_loss := policy_loss.item())
            value_losses.append(value_loss := value_loss.item())

        print(f"[*] Epoch {epoch:2}: {loss = :.3f} ({policy_loss = :.3f}, {value_loss = :.3f})")

    # Debugging print.
    print(f"[*] Length of history: {len(history)}")
    print(f"[*] Model policy output:\n{out_policy}")
    print(f"[*] Policy from MCTS:\n{policy_batch}")

    print(f"[*] Model value output:\n{out_value.squeeze(1)}")
    print(f"[*] Value from MCTS:\n{value_batch.squeeze(1)}")

    # Switch model back to eval mode.
    model.eval()

    return losses, policy_losses, value_losses


def save_model(model, generation: int, attr=None, path=None):
    """Save model to a file.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to save.
    generation : int
        The generation number of the model.
    attr : str | None
        An attribute of the model.
        - "best"
            Best model so far.
        - "tmp"
            Temporary model used in `history_maker.py`.
        - None
            Plain models.
    path : str
        The directory where the model will be saved.
    """
    os.makedirs(path, exist_ok=True)

    model_type = f"{attr}_model" if attr is not None else "model"

    now = datetime.now()
    timestamp = f"{now.year:04}{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}{now.second:02}"
    nonce = os.urandom(4).hex()

    # Remove best / tmp model if exists.
    if attr is not None:
        if filenames := glob.glob(os.path.join(path, f"{model_type}*.pt")):
            assert len(filenames) == 1, f"There should be one best / tmp model, but found {len(filenames)}."
            os.remove(filenames[0])

    filename = f"{model_type}_generation-{generation:06}_{timestamp}_{nonce}.pt"
    torch.save(model.state_dict(), os.path.join(path, filename))


def load_model(generation: int | None, attr=None, path=None):
    """Load model from a file.

    Parameters
    ----------
    generation : int | None
        The generation number of the model to load. If None, the latest model will be loaded.
    attr : str
        An attribute of the model. See `save_model` for description.
    path: str
        The directory where the model is stored.

    Returns
    -------
    torch.nn.Module
        The loaded neural network model.
    """
    model_type = f"{attr}_model" if attr is not None else "model"
    filenames = glob.glob(os.path.join(path, f"{model_type}*.pt"))

    if attr is not None:
        assert len(filenames) == 1, f"There should be one best / tmp model, but found {len(filenames)}."
    else:
        if generation is None:
            # If generation is not specified, load the latest.
            filenames.sort()
        else:
            filenames = glob.glob(os.path.join(path, f"model_generation-{generation:06}*.pt"))
            assert len(filenames) == 1, f"There should be only one model per generation, but found {len(filenames)}."

    print(f"[*] Load model: {filenames[-1]}")
    return torch.load(os.path.join(path, filenames[-1]))
