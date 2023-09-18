from strategy_player import *

class DefensivePlayer(StrategyPlayer):

    def __init__(self, id):

        #Init of the base strategy player
        StrategyPlayer.__init__(self, id, PlayerKind.Defensive)


    def get_knocking_range_count(self, rel_board_state_simple, curr_board_state):
        players_in_knock_rng = 0

        for player in range(4):
            position = rel_board_state_simple[self.id][player]
            # dealing with blockades
            if (curr_board_state[self.id].state[position] > 0.25):
                continue

            for i in range(4):
                if (i == self.id):
                    continue

                for j in range(4):
                    different_player_pos = rel_board_state_simple[i][j]
                    if (different_player_pos != -1 and position <= 51 and (position not in SafeSquares)):
                        range_knock = (position - different_player_pos + 52) % 52
                        if (range_knock >= 1 and range_knock <= 6):
                            players_in_knock_rng = players_in_knock_rng + 1

        return players_in_knock_rng

    def select_nonrandom_new_state(self, board_state, successors, ts):

        range_knock_all = []

        for i in range(len(successors)):

            curr_succ = successors[i]
            next_state_board = curr_succ["new_state"]

            next_state_simple = self.get_simple_board_state(next_state_board)
            next_simple_relative_state = self.get_simple_relative_board_state(next_state_simple)

            successor_knocking_range_count\
                = self.get_knocking_range_count(next_simple_relative_state, next_state_board)

            range_knock_all.append(successor_knocking_range_count)

        knock_low = min(range_knock_all)
        knock_min_range = range_knock_all.count(knock_low)

        #defensive strategy invalid if knock range count is not the same for all players
        if (knock_min_range != len(range_knock_all)):
            return range_knock_all.index(knock_low)

        return -1