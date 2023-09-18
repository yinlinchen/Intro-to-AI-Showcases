#!/usr/bin/env python
# coding: utf-8

# In[1]:


import chess
import chess.polyglot
import random
import math
import os


# In[2]:


MAX = math.inf
MIN = -math.inf
DEPTH = 5
HI_CENTRAL = 0.3
LO_CENTRAL = 0.1
hi_central = [chess.E4, chess.D4, chess.D5, chess.E5]
lo_central = [chess.C3, chess.D3, chess.E3, chess.C4, chess.F4, chess.C5, chess.F5, chess.C6, chess.D6, chess.E6]
white_castled = False
black_castled = False


# In[3]:


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
        
    def __call__(self, board, m):
        k = chess.polyglot.zobrist_hash(board)
        if k not in self.transposition:
            val = self.val(board, m)
            self.transposition[k] = val
            return val
        
        return self.transposition[k]
    
    def val(self, board, m):
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
                
                
        if (m < 15):
            value += self.mobility_val(board, 0.1) + self.king_safety(board, 0.3)
        else:
            value += self.mobility_val(board, 0.2) + self.king_safety(board, 0.1)
                
        return value
    
    def mobility_val(self, board, weight):
        turn = board.turn
        value = 0
        
        board.turn = chess.WHITE
        value += weight * board.legal_moves.count()
        board.turn = chess.BLACK
        value -= weight * board.legal_moves.count()
        board.turn = turn
        return value
    
    def king_safety(self, board, weight):
        value = 0
        if not black_castled:
            if board.has_castling_rights(chess.BLACK):
                value -= weight
            else:
                value += 1 - weight
                
        if not white_castled:
            if board.has_castling_rights(chess.WHITE):
                value += weight
            else:
                value -= 1 - weight
                
        return value
    
    def pawn_structure(self, board):
        value = 0
        structure = {}
        for attacker_square in chess.SquareSet(board.occupied_co[chess.BLACK]):
            piece = board.piece_at(attacker_square).piece_type
            file = chess.square_name(attacker_square)[0]
            if piece == chess.PAWN:
                structure[file] = structure.get(file, 0) + 1
                
        for file in structure:
            if structure[file] > 1:
                value += 0.3 * structure[file]
                
        for attacker_square in chess.SquareSet(board.occupied_co[chess.BLACK]):
            piece = board.piece_at(attacker_square).piece_type
            file = chess.square_name(attacker_square)[0]
            if piece == chess.ROOK and not structure[file]:
                value -= 0.3
                
        structure = {}
        for attacker_square in chess.SquareSet(board.occupied_co[chess.WHITE]):
            piece = board.piece_at(attacker_square).piece_type
            file = chess.square_name(attacker_square)[0]
            if piece == chess.PAWN:
                structure[file] = structure.get(file, 0) + 1
                
        for file in structure:
            if structure[file] > 1:
                value -= 0.3 * structure[file]
                
        for attacker_square in chess.SquareSet(board.occupied_co[chess.WHITE]):
            piece = board.piece_at(attacker_square).piece_type
            file = chess.square_name(attacker_square)[0]
            if piece == chess.ROOK and not structure[file]:
                value += 0.3
            
            
    
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
    
                
        


# In[4]:


# TODO: MTCS, Beam, Parallelism, NEGAMAX
def alpha_beta(heuristic, board, depth, a, b, m, root=False):
    quiescence = False
    m += 1
    if depth >= DEPTH or board.is_game_over() and not quiescence:
        return heuristic(board, m)
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
        move_vals.append((heuristic(board, m) + true_val, move))
        board.pop()
    
    move_vals.sort(key=lambda x: x[0], reverse=board.turn)
    branches = [m[1] for m in move_vals]
    
    
    for move in branches:
        board.push(move)
        if board.is_capture(move):
            quiescence = True
        ab_val = alpha_beta(heuristic, board, depth + 1, a, b, m)
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


# In[5]:


import random
def engine_move(heuristic, board):
    val, moves = alpha_beta(heuristic, board, 0, -10000, 10000, 0, root=True)
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
    


# In[9]:


board = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
display(board)
board.push(engine_move(BaseHeuristic(), board))
display(board)


# In[10]:


board = chess.Board("r1bqkb1r/ppp2ppp/2np1n2/1B2p3/4P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 1 5")
display(board)
board.push(engine_move(BaseHeuristic(), board))
display(board)


# In[11]:


board = chess.Board("rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")
display(board)
board.push(engine_move(BaseHeuristic(), board))
display(board)


# In[12]:


board = chess.Board("rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3")
display(board)
board.push(engine_move(BaseHeuristic(), board))
display(board)


# In[13]:


board = chess.Board("rnbqkb1r/1p3ppp/p2ppn2/6B1/3NP3/2N5/PPP2PPP/R2QKB1R w KQkq - 0 7")
display(board)
board.push(engine_move(BaseHeuristic(), board))
display(board)


# In[14]:


board = chess.Board("rnbqkb1r/1pp2ppp/p3pn2/1N1p4/3P1B2/8/PPP1PPPP/R2QKBNR w KQkq - 0 5")
display(board)
board.push(engine_move(BaseHeuristic(), board))
display(board)


# In[15]:


board = chess.Board("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
display(board)
board.push(engine_move(BaseHeuristic(), board))
display(board)


# In[16]:


board = chess.Board("rnbqkbnr/ppppp2p/5p2/6p1/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3")
display(board)
board.push(engine_move(BaseHeuristic(), board))
display(board)


# In[17]:


board = chess.Board("r1bqk2r/ppp2ppp/3p1nn1/4p3/1bP1P3/2NP1N1P/PP2BPP1/R1BQK2R w KQkq - 1 8")
display(board)
board.push(engine_move(BaseHeuristic(), board))
display(board)


# In[18]:


board = chess.Board("rnbqkbnr/pppp2pp/8/4p3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 0 4")
display(board)
board.push(engine_move(BaseHeuristic(), board))
display(board)


# In[19]:


board = chess.Board("8/8/4k3/8/3K4/4P3/8/8 w - - 0 1")
display(board)
board.push(engine_move(BaseHeuristic(), board))
display(board)


# In[20]:


board = chess.Board("k7/4Q3/8/2K5/8/8/8/8 w - - 0 1")
for i in range(10):
    display(board)
    board.push(engine_move(BaseHeuristic(), board))
    
display(board)


# In[21]:


board = chess.Board("8/8/3k4/7R/6R1/8/8/2K5 w - - 0 1")
for i in range(10):
    display(board)
    board.push(engine_move(BaseHeuristic(), board))
    
display(board)


# In[ ]:




