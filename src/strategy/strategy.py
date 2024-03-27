import random


def random_play(board):
    """Choose a random legal move for `board`."""
    return random.choice(board.legal_moves())


def alpha_beta_pruning_play(board):
    """Select the best move for `board` using the Alpha-Beta Pruning algorithm."""
    best_score, best_move = -float("inf"), None
    for move in board.legal_moves():
        score = -_alpha_beta(board=board.play(move), alpha=-float("inf"), beta=float("inf"))
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def _alpha_beta(board, alpha, beta):
    """Helper function for `alpha_beta_pruning_play`.

    It recursively search to figure out the exact action value.
    """
    if board.has_lost():
        return -1
    elif board.has_drawn():
        return 0

    for move in board.legal_moves():
        score = -_alpha_beta(board.play(move), -beta, -alpha)
        alpha = max(alpha, score)
        if alpha >= beta:
            return alpha
    return alpha


def mcs_play(board, num_playout=10):
    """Choose the best move using Monte Carlo Search (MCS) algorithm."""
    leg_moves = board.legal_moves()
    values = [0 for _ in range(len(leg_moves))]
    for i, move in enumerate(leg_moves):
        for _ in range(num_playout):
            values[i] += -_playout(board.play(move))

    best_val, best_move = -float("inf"), None
    for i in range(len(values)):
        if values[i] > best_val:
            best_val = values[i]
            best_move = leg_moves[i]
    return best_move


def _playout(board):
    """Helper function for `mcs_play`.

    Performs a random playout starting from the given board position.
    """
    if board.has_lost():
        return -1
    elif board.has_drawn():
        return 0

    random_move = random.choice(board.legal_moves())
    return -_playout(board.play(random_move))
