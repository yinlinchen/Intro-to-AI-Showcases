import chess

class Node:
    def __init__(self, board, move=None, parent=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []

    def add_child(self, move):
        new_board = self.board.copy()
        new_board.push(move)
        return Node(new_board, move, self)

def evaluate_node(node):
    if node.board.is_game_over():
        if node.board.is_checkmate():
            return float('-inf') if node.board.turn == chess.WHITE else float('inf')
        else:
            return 0

    piece_values = {'P': 1, 'N': 3, 'B': 3.25, 'R': 5, 'Q': 9, 'K': 100}
    score = 0
    for square in chess.SQUARES:
        piece = node.board.piece_at(square)
        if piece is not None:
            value = piece_values[piece.symbol().upper()]
            score += value if piece.color == chess.WHITE else -value


    mobility_score = len(list(node.board.legal_moves))
    node.board.push(chess.Move.null())
    mobility_score -= len(list(node.board.legal_moves))
    node.board.pop()
    score += mobility_score * 0.1

    white_king_square = node.board.king(chess.WHITE)
    black_king_square = node.board.king(chess.BLACK)
    white_king_safety = -len(list(node.board.attackers(chess.BLACK, white_king_square))) * 0.5
    black_king_safety = -len(list(node.board.attackers(chess.WHITE, black_king_square))) * 0.5
    score += white_king_safety - black_king_safety
    score += evaluate_tactics(node)
    return score


def evaluate_tactics(node):
    board = node.board
    tactical_score = 0

    # Example: Check for forks
    for square in chess.SQUARES:
        attackers = board.attackers(chess.WHITE, square)
        if len(attackers) > 1:
            # White has a fork opportunity
            tactical_score += 1

        attackers = board.attackers(chess.BLACK, square)
        if len(attackers) > 1:
            # Black has a fork opportunity
            tactical_score -= 1

    # Similar logic can be applied for pins, skewers, and discovered attacks
    return tactical_score


def minimax(node, depth, alpha, beta, maximizing_player):
    if depth == 0 or node.board.is_game_over():
        return evaluate_node(node), node.move

    moves = list(node.board.legal_moves)
    moves.sort(key=lambda move: node.board.is_capture(move) or node.board.gives_check(move), reverse=True)

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for move in moves:
            child = node.add_child(move)
            eval, _ = minimax(child, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in moves:
            child = node.add_child(move)
            eval, _ = minimax(child, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

class ChessAgent:
    def __init__(self, depth=4):
        self.depth = depth

    def choose_move(self, board):
        root_node = Node(board)
        _, best_move = minimax(root_node, self.depth, float('-inf'), float('inf'), False)
        return best_move
