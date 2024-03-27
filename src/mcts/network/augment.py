import torch


class DataAugmentor:
    """A class for augmenting training data with various transformations.

    It applies transformations like vertical flip, horizontal flip,and rotation
    to the input data.

    Attributes
    ----------
    boards : torch.Tensor
        A board states tensor.
    policies : torch.Tensor
        A policy tensor containing move probabilities.
    invariant_policies : torch.Tensor
        A invariant policy tensor containing probabilities of special moves (e.g. pass move.)
    values : torch.Tensor
        A value tensor containing the value of each board state.
    """

    def __init__(self, board_batch, policy_batch, value_batch, num_invariants=0):
        """Initialize the DataAugmentor class with input data.

        Parameters
        ----------
        board_batch : torch.Tensor
            A board states tensor.
        policy_batch : torch.Tensor
            A policy tensor containing move probabilities.
        value_batch : torch.Tensor
            A value tensor containing the value of each board state.
        num_invariants : int, optional
            The number of invariant policy actions (e.g., pass move).
        """
        N, _, H, W = board_batch.shape
        _, P = policy_batch.shape
        assert P == H * W + num_invariants

        self.N = N

        self.boards = board_batch
        self.policies, self.invariant_policies = policy_batch.split((P - num_invariants, num_invariants), dim=1)
        self.policies = self.policies.reshape(N, 1, H, W)
        self.values = value_batch

    def vertical_flip(self, augmented_only=False):
        """Apply vertical flip transformation to the input data."""
        new_boards = self.boards.flip(dims=(2,))
        new_policies = self.policies.flip(dims=(2,))

        self._update(new_boards, new_policies, augmented_only)

    def horizontal_flip(self, augmented_only=False):
        """Apply horizontal flip transformation to the input data."""
        new_boards = self.boards.flip(dims=(3,))
        new_policies = self.policies.flip(dims=(3,))

        self._update(new_boards, new_policies, augmented_only)

    def rotate(self, ks=(1, 2, 3), augmented_only=False):
        """Rotate by 90, 180, 270 degrees."""
        new_boards = torch.cat([self.boards.rot90(k=k, dims=(2, 3)) for k in ks])
        new_policies = torch.cat([self.policies.rot90(k=k, dims=(2, 3)) for k in ks])

        self._update(new_boards, new_policies, augmented_only)

    def claim(self):
        """Return the augmented data.

        Returns
        -------
        tuple
            A tuple containing the augmented board states tensor, policy tensor, and value tensor.
        """
        M, _, _, _ = self.policies.shape
        assert M % self.N == 0
        ratio = M // self.N
        res = (
            self.boards,
            torch.cat(
                (self.policies.reshape(M, -1), self.invariant_policies.repeat((ratio, 1))),
                dim=1,
            ),
            self.values.repeat((ratio, 1)),
        )
        return res

    def _update(self, new_boards, new_policies, augmented_only):
        if augmented_only:
            self.boards = new_boards
            self.policies = new_policies
        else:
            self.boards = torch.cat((self.boards, new_boards))
            self.policies = torch.cat((self.policies, new_policies))


def augment_8(board_batch, policy_batch, value_batch):
    augmentor = DataAugmentor(board_batch, policy_batch, value_batch)
    augmentor.rotate()
    augmentor.horizontal_flip()
    return augmentor.claim()


def augment_othello(board_batch, policy_batch, value_batch):
    N = board_batch.shape[0]

    augmentor_1 = DataAugmentor(board_batch, policy_batch, value_batch, num_invariants=1)
    augmentor_1.rotate(ks=(2,))
    res_1 = augmentor_1.claim()
    assert res_1[0].shape[0] == 2 * N

    augmentor_2 = DataAugmentor(board_batch, policy_batch, value_batch, num_invariants=1)
    augmentor_2.rotate(ks=(1, 3), augmented_only=True)
    augmentor_2.horizontal_flip(augmented_only=True)
    res_2 = augmentor_2.claim()
    assert res_2[0].shape[0] == 2 * N

    return tuple(torch.cat((batch_1, batch_2), dim=0) for batch_1, batch_2 in zip(res_1, res_2))
