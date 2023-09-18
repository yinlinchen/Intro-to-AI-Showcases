from aggressive_player import *
from defensive_player import *
from fast_player import *

class MixedStrategyPlayer(StrategyPlayer):

    def __init__(self, id):

        # Init base strategy player
        StrategyPlayer.__init__(self, id, PlayerKind.Mixed)

        #creating new aggressive, defensive and fast players that will be used by the mixed strategy player
        self.Aggressive = AggressivePlayer(id)
        self.Defensive = DefensivePlayer(id)
        self.Fast = FastPlayer(id)

    def select_nonrandom_new_state(self, board_state, successors, timestamp):

        #prioritize defensive, then aggressive then fast the random for strategic play

        curr_succ_ind = self.Defensive.select_nonrandom_new_state(board_state, successors, timestamp)

        if (curr_succ_ind != -1):
            return curr_succ_ind

        curr_succ_ind = self.Aggressive.select_nonrandom_new_state(board_state, successors, timestamp)

        if (curr_succ_ind != -1):
            return curr_succ_ind

        curr_succ_ind = self.Fast.select_nonrandom_new_state(board_state, successors, timestamp)

        if (curr_succ_ind != -1):
            return curr_succ_ind

        return -1

