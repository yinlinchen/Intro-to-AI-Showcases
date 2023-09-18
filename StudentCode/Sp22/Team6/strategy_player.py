from player import *
import random
import math

class StrategyPlayer(Player):

    def __init__(self, id, kind):

        # Init base player for strategy
        Player.__init__(self, id, kind)

    def select_nonrandom_new_state(self, board_state, successors, timestamp):
        return -1

    def select_new_state(self, st_board_curr, succ_dict, timestamp):
        tot_ply = 4
        comm_sq = 52
        tot_sq = 58
        one_piece_val = 0.25
        d_max = 6

        ind_succ = self.select_nonrandom_new_state(st_board_curr, succ_dict, timestamp)

        if (ind_succ == -1):
            ind_succ = random.randint(0, len(succ_dict) - 1)

        new_board_state = succ_dict[ind_succ]["new_state"]
        if new_board_state[self.id].state[tot_sq] == 4*one_piece_val:
            try:
                st_board_curr[(self.id + 1) % tot_ply].cummulative_rew += -4 * one_piece_val
            except AttributeError:
                pass

            try:
                st_board_curr[(self.id + 2) % tot_ply].cummulative_rew += -4*one_piece_val
            except AttributeError:
                pass

            try:
                st_board_curr[(self.id + 3) % tot_ply].cummulative_rew += -4 * one_piece_val
            except AttributeError:
                pass
        diff = new_board_state[(self.id - 1) % tot_ply].state[0] - st_board_curr[(self.id - 1) % tot_ply].state[0]

        if diff > 0 and timestamp > 0 and st_board_curr[(self.id - 1) % tot_ply].timestamp == timestamp - 1:
            try:
                st_board_curr[(self.id - 1) % tot_ply].cummulative_rew += -1 * one_piece_val
            except AttributeError:
                pass
        for i in range(tot_ply):
            try:
                st_board_curr[(self.id - i) % tot_ply].reward()
            except AttributeError:
                pass
        return ind_succ

