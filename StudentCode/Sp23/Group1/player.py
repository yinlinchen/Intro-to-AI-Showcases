import random
import time
import copy
from collections import defaultdict

from uttt_main import UTTT
from window import Window

class Player:

    def make_move(self, state: UTTT):
        return


class RandomPlayer(Player):
    def __init__(self, index=0):
        self.index = index

    def make_move(self, state: UTTT, window=None):
        start_time = time.perf_counter()

        random_index = random.randrange(len(state.valid_moves))
        board_ai = state.valid_moves[random_index][0]
        cell_ai = state.valid_moves[random_index][1]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        state.make_move(board_ai, cell_ai)
        print("Random AI has made the move (", board_ai,", ", cell_ai,") in ", elapsed_time, " seconds")
        return board_ai, cell_ai


class UserPlayer(Player):
    def make_move(self, state: UTTT, window=None, terminal=False):
        if terminal:
            input_str = input("Make Your Move: ")
            board, cell = map(int, input_str.split())
            if(state.make_move(board, cell) is False):
                print("That move was invalid, please make a valid move")
        # window move
        else:
            board, cell = window.make_move()
            if(state.make_move(board, cell) is False):
                print("That move was invalid, please make a valid move")
        return board, cell


class MonteCarloPlayer(Player):
    def __init__(self, sim_count):
        self.sim_count = sim_count

    def make_move(self, state: UTTT, window=None):
        start_time = time.perf_counter()

        simulation_scores = self.simulate_games(state)
        index_choice = simulation_scores.index(max(simulation_scores))
        board_ai = state.valid_moves[index_choice][0]
        cell_ai = state.valid_moves[index_choice][1]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        state.make_move(board_ai, cell_ai)
        print("Monte Carlo AI has made the move (", board_ai,", ", cell_ai,") in ", elapsed_time, " seconds")
        return board_ai, cell_ai

    def simulate_games(self, state: UTTT):
        possible_moves = state.valid_moves

        games = [0] * len(possible_moves)
        for i in range(len(possible_moves)):
            for _ in range(self.sim_count):
                temp_game = state.generate_next_uttt(possible_moves[i][0], possible_moves[i][1])
                games[i] += MonteCarloPlayer.run_simulation(temp_game)
        return games

    def run_simulation(temp_state: UTTT):
        player_id = 3 - temp_state.player_turn
        while temp_state.winner == 0:
            random_index = random.randrange(len(temp_state.valid_moves))
            board_ai = temp_state.valid_moves[random_index][0]
            cell_ai = temp_state.valid_moves[random_index][1]
            temp_state.make_move(board_ai, cell_ai)
        return 1 if player_id == temp_state.winner else 0 if temp_state.winner == -1 else -1
    
class MiniMaxPlayer(Player):
    def __init__(self, max_depth, player_num):
        self.max_depth = max_depth
        self.player_num = player_num
        self.other_player_num = 1 if player_num == 2 else 2
        self.board_values = [0.2, 0.17, 0.2, 0.17, 0.22, 0.17, 0.2, 0.17, 0.2]

        positions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        position_neighbors = [[1, 3, 4], [0, 2, 3, 4, 5], [1, 4, 5], [0, 1, 4, 6, 7]]

    def make_move(self, state: UTTT, window=None):
        start_time = time.perf_counter()

        move, value = self.minimax(state, True, 0, -1000000, 1000000)
        board_ai, cell_ai = move

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        state.make_move(board_ai, cell_ai)
        print("Mini Max AI has made the move (", board_ai,", ", cell_ai,") in ", elapsed_time, " seconds")
        return board_ai, cell_ai
    
    def uttt_heuristic(self, state: UTTT):
        
        evaluation = 0
        for idx, single_board in enumerate(state.game_grid):
            evaluation += self.board_heuristic(single_board) * self.board_values[idx]

        # Evaluate Groups of 3
        row_1 = self.row_heuristic(state.simplified_grid[0:3]) * 10
        row_2 = self.row_heuristic(state.simplified_grid[3:6]) * 10
        row_3 = self.row_heuristic(state.simplified_grid[6:9]) * 10

        col_1 = self.row_heuristic(state.simplified_grid[0:9:3]) * 10
        col_2 = self.row_heuristic(state.simplified_grid[1:9:3]) * 10
        col_3 = self.row_heuristic(state.simplified_grid[2:9:3]) * 10

        dia_1 = self.row_heuristic(state.simplified_grid[0:9:4]) * 20 
        dia_2 = self.row_heuristic(state.simplified_grid[2:8:2]) * 20

        evaluation += row_1 + row_2 + row_3 + col_1 + col_2 + col_3 + dia_1 + dia_2

        return evaluation
        
    def board_heuristic(self, board):

        evaluation = 0
        for idx, val in enumerate(board):
            if val == self.player_num:
                evaluation += self.board_values[idx]
        
        # Evaluate Groups of 3
        row_1 = self.row_heuristic(board[0:3])
        row_2 = self.row_heuristic(board[3:6])
        row_3 = self.row_heuristic(board[6:9])

        col_1 = self.row_heuristic(board[0:9:3])
        col_2 = self.row_heuristic(board[1:9:3])
        col_3 = self.row_heuristic(board[2:9:3])

        dia_1 = self.row_heuristic(board[0:9:4])
        dia_2 = self.row_heuristic(board[2:8:2])

        evaluation += row_1 + row_2 + row_3 + col_1 + col_2 + col_3 + dia_1 + dia_2
    
        #evaluation += self.win_heuristic(board)
        
        return evaluation

    def row_heuristic(self, row):

        evaluation = 0
        player_squares = row.count(self.player_num)
        other_player_squares = row.count(self.other_player_num)

        # Two in a Row (Player)
        if player_squares == 2 and other_player_squares == 0:
            evaluation += 10
        
        # Two in a Row (Other)
        if player_squares == 0 and other_player_squares == 2:
            evaluation -= 10

        # Three in a Row (Player)
        if player_squares == 3 and other_player_squares == 0:
            evaluation += 1000
        
        # Three in a Row (Other)
        if player_squares == 3 and other_player_squares == 0:
            evaluation -= 1000
        
        # Blocked 2 in a Row (Player)
        if player_squares == 2 and other_player_squares == 1:
            evaluation -= 50
        
        # Blocked 2 in a Row (Other)
        if player_squares == 1 and other_player_squares == 2:
            evaluation += 50
        
        return evaluation
        
    def win_heuristic(self, board):
        row_1 = 100 if board[0:3].count(self.player_num) == 3 else 0
        row_2 = 100 if board[3:6].count(self.player_num) == 3 else 0
        row_3 = 100 if board[6:9].count(self.player_num) == 3 else 0

        col_1 = 100 if board[0:9:3].count(self.player_num) == 3 else 0
        col_2 = 100 if board[1:9:3].count(self.player_num) == 3 else 0
        col_3 = 100 if board[2:9:3].count(self.player_num) == 3 else 0

        dia_1 = 100 if board[0:9:4].count(self.player_num) == 3 else 0
        dia_2 = 100 if board[2:8:2].count(self.player_num) == 3 else 0

        return row_1 + row_2 + row_3 + col_1 + col_2 + col_3 + dia_1 + dia_2
        
    def minimax(self, state: UTTT, max_player, depth, alpha, beta):

        # Base Case
        if depth == self.max_depth:
            return None, self.uttt_heuristic(state)
        elif len(state.valid_moves) == 0:
            return None, self.uttt_heuristic(state)
        elif state.winner > 0:
            return None, self.uttt_heuristic(state)

        possible_moves = state.valid_moves

        depth += 1

        # Maxmizing Player
        if max_player:
            max_value = -100000
            max_action = None
            for move in possible_moves:
                succsessor = state.generate_next_uttt(move[0], move[1])
                curr_action, curr_max = self.minimax(succsessor, False, depth, alpha, beta)
                if curr_max > max_value:
                    max_value = curr_max
                    max_action = move
                alpha = max(max_value, alpha)
                if beta < alpha:
                    break
            return max_action, max_value
        # Minimizing Player
        else:
            min_value = 100000
            min_action = None
            for move in possible_moves:
                succsessor = state.generate_next_uttt(move[0], move[1])
                curr_action, curr_min = self.minimax(succsessor, True, depth, alpha, beta)
                if curr_min < min_value:
                    min_value = curr_min
                    min_action = move
                beta = min(min_value, beta)
                if beta < alpha:
                    break
            return min_action, min_value
    