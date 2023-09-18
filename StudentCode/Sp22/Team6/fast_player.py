from strategy_player import *

class FastPlayer(StrategyPlayer):
    def __init__(self, id):

        # Init a base strategy player
        StrategyPlayer.__init__(self, id, PlayerKind.Fast)

    def select_nonrandom_new_state(self, board_state, successors, ts):
        successor_ind = -1

        for i in range(0, len(successors)):
            if successor_ind == -1:
                new_location = 0
            else:
                new_location = successors[successor_ind]["action"][1]

            succ_new_loc = successors[i]["action"][1]
            if (succ_new_loc > new_location):
                successor_ind = i

        if successor_ind != -1:
            return successor_ind

        return -1

