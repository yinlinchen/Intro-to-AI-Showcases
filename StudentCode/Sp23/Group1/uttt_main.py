import copy
import random
import time

class UTTT:
    def __init__(self, game_grid=None, simplified_grid=None, active_board=None, valid_moves=None, winner=0, player_turn=1):
        """
        Initializes the UTTT variable.

        If no arguments are provided, the UTTT variable is initialized to default settings.

        If arguments are provided, the UTTT variable is initialized given each variable.

        :param game_grid: An array of length 9 holding an array of length 9 in each index representing the whole game grid
        :param simplified_grid: An array of length 9 holding the value of which player has won the corresponding board (0 if not won yet)
        :param active_board: Which board valid moves are limited to given the previous move (None if no limitations)
        :param winner: player that has won the game (0 if no one has won yet)
        :param player_turn: which player's turn it is to act
        """

        self.active_board = active_board
        self.winner = winner
        self.player_turn = player_turn
        self.finished = 0

        if game_grid is None:
            self.game_grid = [[0] * 9 for _ in range(9)]
        else:
            self.game_grid = game_grid

        if simplified_grid is None:
            self.simplified_grid = [0] * 9
        else:
            self.simplified_grid = simplified_grid

        if valid_moves is None:
            self.valid_moves = self.get_valid_moves()
        else:
            self.valid_moves = valid_moves

        self.last_move = None

        
        # I want to make valid_moves live in the object and get updated in make_move()
        

    def get_valid_moves(self):
        """
        Returns all valid moves the player can make on their next move taking into account the active board (None if all are active) and which spaces have already been filled

        Args:
            None

        Returns:
            Array of Tuples: Returns an array of tuples (board, cell) of all legal moves the player can make

        """
        
        valid_moves = []
        if self.active_board is None:
            for board, winner in enumerate(self.simplified_grid):
                if (winner == 0):
                    for cell in range(9):
                        if self.game_grid[board][cell] == 0:
                            valid_moves.append((board, cell))
        else:
            for cell in range(9):
                if self.game_grid[self.active_board][cell] == 0:
                    valid_moves.append((self.active_board, cell))

        if len(valid_moves) == 0:
            count_1 = sum(1 for mark in self.simplified_grid if mark == 1)
            count_2 = sum(1 for mark in self.simplified_grid if mark == 2)

            if count_1 > count_2:
                self.winner = 1
            elif count_2 > count_1:
                self.winner = 2
            else:
                self.winner = -1
            
        return valid_moves
    
    # to check if there is a draw, use this conditional:
    # if len(valid_moves) == 0 and self.winner == 0:

    def make_move(self, board, cell):
        """
        Updates the UTTT object according to the move made

        Args:
            board (int) : The board number the player is making the move on
            cell (int) : The cell in the board the player is making the move on
            valid_moves (array of (int, int) tuples) : An array of all valid (board, cell) moves

        Returns:
            Boolean: Returns true if the move is valid and false otherwise

        """
        if (board, cell) in self.valid_moves:
            self.game_grid[board][cell] = self.player_turn
            if self.check_board_win(board, self.player_turn):
                self.simplified_grid[board] = self.player_turn

                # INPUTTING -1 AS THE BOARD NUMBER CHECKS THE SIMPLIFIED BOARD TO SEE IF SOMEONE WON THE FULL GAME
                if (self.check_board_win(-1, self.player_turn)):
                    self.winner = self.player_turn
                    return True # technically not necessary
                
            # make the current board the same as the cell if that board has any free spaces, otherwise place no restriction
            self.active_board = cell if (self.simplified_grid[cell] == 0 and 0 in self.game_grid[cell]) else None
            self.player_turn = 3 - self.player_turn

            self.valid_moves = self.get_valid_moves()
            self.last_move = board, cell

            return True, 
        return False
    

    # This assume the move you are making is valid and not take in valid_moves as an input. Therefore the validity of the move MUST be checked before the function is called
    # If it is better for this to be changed to self validate the moves, how should I handle an invalid move?
    def generate_next_uttt(self, board, cell):
        """
        Generates the next UTTT object given that the 

        Args:
            board (int) : The board number the player is making the move on
            cell (int) : The cell in the board the player is making the move on
            valid_moves (array of (int, int) tuples) : An array of all valid (board, cell) moves

        Returns:
            UTTT: The new UTTT generated from the given move

        """
        next_game_state = UTTT(copy.deepcopy(self.game_grid), copy.deepcopy(self.simplified_grid), self.active_board, self.valid_moves, self.winner, self.player_turn)
        next_game_state.make_move(board, cell)
        return next_game_state

    def check_board_win(self, board, player):
        """
        Checks to see if a given player has won the given board (therefore winning them the game)

        Args:
            board (int) : The board that the player may have won **IF -1 THEN SIMPLIFIED BOARD**
            player (int) : The player that you are checking if they won

        Returns:
            bool: Returns true if the given player has won the board and false otherwise

        """
        # I was thinking about adding this variable: winning_combinations = [{0,1,2}, {3,4,5}, {6,7,8}, {0,3,6}, {1,4,7}, {2,5,8}, {0,4,8}, {2,4,6}]
        # and using that to short hand the return statement but couldn't find a way

        if board == -1: 
            b = self.simplified_grid 
        else: 
            b = self.game_grid[board]
        
        return ((b[0] == b[1] == b[2] == player) or
                (b[3] == b[4] == b[5] == player) or
                (b[6] == b[7] == b[8] == player) or
                (b[0] == b[3] == b[6] == player) or
                (b[1] == b[4] == b[7] == player) or
                (b[2] == b[5] == b[8] == player) or
                (b[0] == b[4] == b[8] == player) or
                (b[2] == b[4] == b[6] == player))
    
    def print_board(self):
        print()
        rows = []
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                sub_rows = []
                for k in range(3):
                    sub_rows.append(str(self.game_grid[i+k][j:j+3]).replace('0', '_').replace('1', 'X').replace('2', 'O'))
                rows.append(" ".join(sub_rows))
            if i != 6:
                rows.append("---------+---------+---------")
        print("\n".join(rows))
