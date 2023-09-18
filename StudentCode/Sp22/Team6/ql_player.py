import copy
import math
import random

from player import Player
from player import PlayerKind


class QLPlayer(Player):

    # QL Parameters
    learning_rate = 0.5
    discount_rate = 0.95

    def __init__(self, id, train=False, nn=None, epsilon=0.0):

        # Init Base player
        Player.__init__(self, id, PlayerKind.QLearning)


        self.old_board_state = None
        self.old_to_new_action = None
        self.new_board_state = None
        self.train = train
        self.nn = nn
        self.epsilon = epsilon
        self.cummulative_rew = 0.0

    def reward(self):

        tot_sq = 58
        tot_ply = 4
        d_max_val = 6

        if not self.train or self.cummulative_rew == 0.0:
            self.cummulative_rew = 0.0
            return

        if self.old_board_state is not None and self.old_to_new_action is not None and self.new_board_state is not None:

            simp = True
            old_inputs = self.board_state_and_action_to_nn_inputs(self.old_board_state, self.old_to_new_action)
            prev_q_val = self.nn.evaluate(old_inputs)

            is_fin = False
            min_q_val_appr = 0
            max_q_val_appr = 0

            for player in self.new_board_state:
                if player.state[tot_sq] == 1:
                    is_fin = True
                    break

            if not is_fin:
                min_q_val_appr = float("inf")
                max_q_val_appr = float("-inf")

                if simp:
                    next_player = self.new_board_state[(self.id + 1) % tot_ply]

                    for dice in range(1, d_max_val + 1):
                        new_successors = next_player.get_next_states(dice, self.new_board_state)

                        if new_successors is not None:
                            for s in new_successors:
                                new_inputs = next_player.board_state_and_action_to_nn_inputs(self.new_board_state,
                                                                                             s['action'])

                                new_q_est = self.nn.evaluate(new_inputs)

                                if new_q_est > max_q_val_appr:
                                    max_q_val_appr = new_q_est

                                if new_q_est < min_q_val_appr:
                                    min_q_val_appr = new_q_est
                else:

                    cur_state1 = self.new_board_state
                    next_player1 = cur_state1[(self.id + 1) % tot_ply]

                    for dice1 in range(1, d_max_val + 1):
                        new_successors1 = next_player1.get_next_states(dice1, cur_state1)

                        if new_successors1 is None:
                            continue

                        for s1 in new_successors1:
                            cur_state2 = s1['new_state']
                            next_player2 = cur_state2[(self.id + 2) % tot_ply]

                            for dice2 in range(1, 6 + 1):
                                new_successors2 = next_player2.get_next_states(dice2, cur_state2)

                                if new_successors2 is None:
                                    continue

                                for s2 in new_successors2:
                                    cur_state3 = s2['new_state']
                                    next_player3 = cur_state3[(self.id + 3) % tot_ply]

                                    for dice3 in range(1, d_max_val + 1):
                                        new_successors3 = next_player3.get_next_states(dice3, cur_state3)

                                        if new_successors3 is None:
                                            continue

                                        for s3 in new_successors3:
                                            cur_state4 = s3['new_state']
                                            next_player4 = cur_state4[(self.id + tot_ply) % tot_ply]

                                            for dice4 in range(1, d_max_val + 1):
                                                new_successors4 = next_player4.get_next_states(dice4, cur_state4)

                                                if new_successors4 is None:
                                                    continue

                                                for s4 in new_successors4:
                                                    new_inputs = self.board_state_and_action_to_nn_inputs(cur_state4,
                                                                                                          s4['action'])

                                                    new_q_est = self.nn.evaluate(new_inputs)

                                                    if new_q_est > max_q_val_appr:
                                                        max_q_val_appr = new_q_est

            if max_q_val_appr == float("-inf"):
                max_q_val_appr = 0

            if min_q_val_appr == float("inf"):
                min_q_val_appr = 0


            if simp:
                #Bellman Equation for Q-Learning
                q_updated_val = prev_q_val + QLPlayer.learning_rate * (self.cummulative_rew - QLPlayer.discount_rate * max_q_val_appr - prev_q_val)
            else:
                q_updated_val = prev_q_val + QLPlayer.learning_rate * (self.cummulative_rew - QLPlayer.discount_rate * max_q_val_appr - prev_q_val)


            self.nn.train_with_datapoint(old_inputs, q_updated_val)


        self.cummulative_rew = 0.0

    def select_new_state(self, curr_st_board, succ_list, ts):
        tot_sq = 58
        tot_ply = 4
        one_piece_val = 0.25
        ply_sq = 52
        d_max = 6

        self.old_board_state = copy.deepcopy(curr_st_board)


        if self.train and random.uniform(0, 1) < self.epsilon:
            succ_ind = random.randint(0, len(succ_list) - 1)
        else:

            all_q_vals = list()

            for succ_ind in range(len(succ_list)):
                action = succ_list[succ_ind]["action"]
                val_to_app  = self.board_state_and_action_to_nn_inputs(self.old_board_state, action)
                all_q_vals.append(self.nn.evaluate(val_to_app))

            max_q_value = max(all_q_vals)

            if math.isnan(max_q_value):
                exit()
            else:
                possible_succ = list()

                for i in range(len(all_q_vals)):
                    if all_q_vals[i] == max_q_value:
                        possible_succ.append(i)

                succ_ind = possible_succ[random.randint(0, len(possible_succ) - 1)]


        self.old_to_new_action = (succ_list[succ_ind]["action"][0], succ_list[succ_ind]["action"][1])
        self.new_board_state = succ_list[succ_ind]["new_state"]

        cur_board_state = curr_st_board
        action = self.old_to_new_action
        new_board_state = self.new_board_state


        if new_board_state[self.id].state[tot_sq] == 1.0:

            self.cummulative_rew += 1.0


            try:
                cur_board_state[(self.id + 1) % tot_ply].cummulative_rew += -(4*one_piece_val)
            except AttributeError:
                pass

            try:
                cur_board_state[(self.id + 2) % tot_ply].cummulative_rew += -(4*one_piece_val)
            except AttributeError:
                pass

            try:
                cur_board_state[(self.id + 3) % tot_ply].cummulative_rew += -(4*one_piece_val)
            except AttributeError:
                pass


        if new_board_state[self.id].state[0] < cur_board_state[self.id].state[0]:
            self.cummulative_rew += one_piece_val


        inp_pice_location = action[0]


        if 1 <= inp_pice_location <= 51:
            #pieces are safe at these sqaures
            safe_sqs = [1, 9, 14, 22, 27, 35, 40, 48]
            if inp_pice_location not in safe_sqs:
                if cur_board_state[self.id].state[inp_pice_location] == one_piece_val:

                    can_be_att = False

                    for new_player in range(1, 4):

                        src_piece_loc_in_np = (inp_pice_location - 13 * new_player) % ply_sq


                        for other_player in range(1, ply_sq):
                            if cur_board_state[(self.id + new_player) % tot_ply].state[other_player] == 0:
                                continue

                            if 0 < src_piece_loc_in_np - other_player <= d_max:
                                can_be_att = True
                                break

                        if can_be_att:
                            self.cummulative_rew += 0.2
                            break


        for new_player in range(1, 4):
            diff = new_board_state[(self.id - new_player) % 4].state[0] - cur_board_state[(self.id - new_player) % 4].state[0]

            if diff > 0:
                self.cummulative_rew += 0.15 * diff * 4

        past_st = cur_board_state[(self.id - 1) % 4].state[0]
        future_st = new_board_state[(self.id - 1) % 4].state[0]

        if future_st-past_st > 0 and ts > 0 and cur_board_state[(self.id - 1) % tot_ply].timestamp == ts - 1:
            try:
                cur_board_state[(self.id - 1) % tot_ply].cummulative_rew += -(one_piece_val)
            except AttributeError:
                pass




        for l in range(1, ply_sq):
            safe_sqs = [1, 9, 14, 22, 27, 35, 40, 48]
            if l in safe_sqs:
                continue

            if new_board_state[self.id].state[l] >= 0.5 > cur_board_state[self.id].state[l]:
                self.cummulative_rew += 0.05
                break


        for i in range(tot_ply):
            try:
                cur_board_state[(self.id - i) % tot_ply].reward()
            except AttributeError:
                pass

        return succ_ind
