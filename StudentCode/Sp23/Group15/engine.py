import chess
import chess.polyglot
import random
import math
import os

MAX = math.inf
MIN = -math.inf
DEPTH = 5
HI_CENTRAL = 0.3
LO_CENTRAL = 0.1
hi_central = [chess.E4, chess.D4, chess.D5, chess.E5]
lo_central = [chess.C3, chess.D3, chess.E3, chess.C4, chess.F4, chess.C5, chess.F5, chess.C6, chess.D6, chess.E6]
white_castled = False
black_castled = False

def random_move(board):
    return random.choice(list(board.legal_moves))

class BaseHeuristic(object):
    # AlphaZero: 
    # Knight: 3.05, Bishop: 3.33, Rook: 5.63, Queen: 9.5
    def __init__(self):
        self.values = {chess.PAWN: 1,
             chess.KNIGHT: 3,
             chess.BISHOP: 3,
             chess.ROOK: 5,
             chess.QUEEN: 9,
             chess.KING: 0}
        self.transposition = {}
        
    def __call__(self, board):
        k = chess.polyglot.zobrist_hash(board)
        if k not in self.transposition:
            val = self.val(board)
            self.transposition[k] = val
            return val
        
        return self.transposition[k]
    
    def val(self, board):
        if board.is_game_over():
            res = board.result()
            if res == "1-0":
                return 10000
            elif res == "0-1":
                return -10000
            else:
                return 0
            
        value = 0
        piece_locs = board.piece_map()
        for square, symbol in piece_locs.items():
            if symbol.color == chess.WHITE:
                value += self.values[symbol.piece_type]
            else:
                value -= self.values[symbol.piece_type]
                
        value += self.mobility_val(board) + self.king_safety(board)
                
        return value
    
    def mobility_val(self, board):
        turn = board.turn
        value = 0
        
        board.turn = chess.WHITE
        value += 0.1 * board.legal_moves.count()
        board.turn = chess.BLACK
        value -= 0.1 * board.legal_moves.count()
        board.turn = turn
        return value
    
    def king_safety(self, board):
        value = 0
        if not black_castled:
            if board.has_castling_rights(chess.BLACK):
                value -= 0.3
            else:
                value += 0.7
                
        if not white_castled:
            if board.has_castling_rights(chess.WHITE):
                value += 0.3
            else:
                value -= 0.7
                
        return value
        
    
    def centrality_val(self, board):
        turn = board.turn
        value = 0
        attacked = []
        for attacker in chess.SquareSet(board.occupied_co[chess.WHITE]):
            attacks = board.attacks(attacker)
            for s in attacks:
                attacked.append((s, attacker, board.piece_at(attacker).piece_type))
            
        for a, s, p in attacked:
            if a in hi_central or s in hi_central:
                if p == chess.KNIGHT or p == chess.BISHOP:
                    value += 0.3
                elif p == chess.PAWN:
                    value += 0.5
            elif a in lo_central or s in lo_central:
                if p == chess.KNIGHT or p == chess.BISHOP:
                    value += 0.5
                elif p == chess.PAWN:
                    value += 0.3

        attacked = []
        for attacker in chess.SquareSet(board.occupied_co[chess.BLACK]):
            attacks = board.attacks(attacker)
            for s in attacks:
                attacked.append((s, attacker, board.piece_at(attacker).piece_type))
            
        for a, s, p in attacked:
            if a in hi_central or s in hi_central:
                if p == chess.KNIGHT or p == chess.BISHOP:
                    value -= 0.3
                elif p == chess.PAWN:
                    value -= 0.5
            elif a in lo_central or s in lo_central:
                if p == chess.KNIGHT or p == chess.BISHOP:
                    value -= 0.5
                elif p == chess.PAWN:
                    value -= 0.3
                
                
        return value
        
    #TODO: centrality, king safety, pawn structure, king tropism
                
# TODO: MTCS, Quiescence, Beam, Parallelism, NEGAMAX
def alpha_beta(heuristic, board, depth, a, b, root=False):
    if depth >= DEPTH or board.is_game_over():
        return heuristic(board)
    val = 0
    if root:
        moves = []
    turn = board.turn
    if turn == chess.WHITE:
        val = -10000
    else:
        val = 10000
    
    move_vals = []
    for move in board.legal_moves:
        board.push(move)
        true_val = 0
        if board.is_castling(move):
            true_val += 0.2
        move_vals.append((heuristic(board) + true_val, move))
        board.pop()
    
#     if depth > 3:
#         move_vals = move_vals[:10]
    move_vals.sort(key=lambda x: x[0], reverse=board.turn)
    branches = [m[1] for m in move_vals]
    
    
    for move in branches:
        board.push(move)
        ab_val = alpha_beta(heuristic, board, depth + 1, a, b)
        board.pop()
        
        if root:
            moves.append((ab_val, move))
        
        if turn == chess.WHITE: # maximizer
            val = max(val, ab_val)
            a = max(a, val)
            if a >= b: # prune for beta
                break
        else:
            val = min(val, ab_val)
            b = min(b, val)
            if a >= b:
                break # prune for alpha
                
    if root:
        return val, moves
    else:
        return val
    
def engine_move(heuristic, board):
    val, moves = alpha_beta(heuristic, board, 0, -10000, 10000, root=True)
    moves.sort(key=lambda x: x[0], reverse=board.turn)
    # TODO: tiebreaks
    print(moves)
    val = moves[0][0]
    pawns = []
    minor = []
    rem = []
    for v, m in moves:
        if v == val:
            pa = board.piece_at(m.from_square).piece_type
            if pa == chess.PAWN:
                pawns.append(m)
            elif pa == chess.KNIGHT or pa == chess.BISHOP:
                minor.append(m)
            else:
                rem.append(m)
                
    if pawns:
        return random.choice(pawns)
    if minor:
        return random.choice(minor)
    
    m = random.choice(rem)
    if board.is_castling(m):
        if board.turn == chess.WHITE:
            white_castled = True
        else:
            black_castled = True
    return m
    