from strategy_player import *

class AggressivePlayer(StrategyPlayer):

    def __init__(self, id):

        #Init of the base strategy player
        StrategyPlayer.__init__(self, id, PlayerKind.Aggressive)

    def select_nonrandom_new_state(self, board_state, successor_dict, ts):
        for i in range(len(successor_dict)):
            next_state = successor_dict[i]["new_state"]

            for p in range(3):
                if (p == self.id):
                    continue

                if ((board_state[p].state[0] + 0.25) == next_state[p].state[0]):
                    return i

        return -1

