import random
import math
import copy

from player import Player
from player import PlayerKind


class QLPlayer(Player):

    # QL Parameters
    debug = False
    learning_rate = 0.8
    discount_rate = 0.95

    def __init__(self, id, train=False, nn=None, epsilon=0.0):

        # Initialize a generic player
        Player.__init__(self, id, PlayerKind.QLearning)

        self.old_board_state = None
        self.old_to_new_cat = None
        self.old_to_new_action = None
        self.new_board_state = None
        self.train = train
        self.nn = nn
        self.epsilon = epsilon
        self.cum_reward = 0.0

    def board_state_and_category_to_nn_inputs(self, board_state, category):
        tot_ply = 4

        inputs = [0.0, ] * 237

        i = 0

        for p_order in range(tot_ply):
            for s in board_state[(self.id + p_order) % 4].state:
                inputs[i] = s
                i += 1

        if category == Player.DEFENSIVE_MOVE:
            inputs[236] = 0.20
        elif category == Player.AGGRESSIVE_MOVE:
            inputs[236] = 0.40
        elif category == Player.FAST_MOVE:
            inputs[236] = 0.60
        elif category == Player.RELEASE_MOVE:
            inputs[236] = 0.80
        elif category == Player.RANDOM_MOVE:
            inputs[236] = 1.00

        return inputs

    def reward(self):
        tot_ply = 4
        comm_sq = 52
        tot_sq = 58
        one_piece_val = 0.25
        d_max = 6

        if not self.train or self.cum_reward == 0.0:
            self.cum_reward = 0.0
            return

        if self.old_board_state is not None and self.old_to_new_cat is not None and self.new_board_state is not None:
            prev_inp = self.board_state_and_category_to_nn_inputs(self.old_board_state, self.old_to_new_cat)

            prev_q_val = self.nn.evaluate(prev_inp)

            is_fin = False
            q_val_min_apprx = 0
            q_val_max_apprx = 0

            for p in self.new_board_state:
                if p.state[tot_sq] == 1:
                    is_fin = True
                    break

            if not is_fin:
                q_val_min_apprx = float("inf")
                q_val_max_apprx = float("-inf")

                following_player = self.new_board_state[(self.id + 1) % tot_ply]

                action_cat = list()

                for d_val in range(1, d_max + 1):
                    next_succ = following_player.get_next_states(d_val, self.new_board_state)

                    if next_succ is not None:
                        for s in next_succ:
                            if s['categories'] & Player.DEFENSIVE_MOVE > 0:
                                action_cat.append(Player.DEFENSIVE_MOVE)

                            if s['categories'] & Player.AGGRESSIVE_MOVE > 0:
                                action_cat.append(Player.AGGRESSIVE_MOVE)

                            if s['categories'] & Player.FAST_MOVE > 0:
                                action_cat.append(Player.FAST_MOVE)

                            if s['categories'] & Player.RELEASE_MOVE > 0:
                                action_cat.append(Player.RELEASE_MOVE)

                            if s['categories'] & Player.RANDOM_MOVE > 0:
                                action_cat.append(Player.RANDOM_MOVE)

                action_cat = list(set(action_cat))

                for i in action_cat:
                    new_inputs = following_player.board_state_and_category_to_nn_inputs(self.new_board_state, i)
                    new_q_est = self.nn.evaluate(new_inputs)

                    if new_q_est > q_val_max_apprx:
                        q_val_max_apprx = new_q_est

                    if new_q_est < q_val_min_apprx:
                        q_val_min_apprx = new_q_est

            if q_val_max_apprx == float("-inf"):
                q_val_max_apprx = 0

            if q_val_min_apprx == float("inf"):
                q_val_min_apprx = 0

            q_val_updated = prev_q_val + QLPlayer.learning_rate * (self.cum_reward - QLPlayer.discount_rate * q_val_min_apprx - prev_q_val)

            self.nn.train_with_datapoint(prev_inp, q_val_updated)

        self.cum_reward = 0.0

    def select_new_state(self, curr_st_board, succ_list, ts):
        tot_ply = 4
        comm_sq = 52
        tot_sq = 58
        one_piece_val = 0.25
        d_max = 6

        self.old_board_state = copy.deepcopy(curr_st_board)

        action_cat = []

        for s in succ_list:
            if s['categories'] & Player.DEFENSIVE_MOVE > 0:
                action_cat.append(Player.DEFENSIVE_MOVE)

            if s['categories'] & Player.AGGRESSIVE_MOVE > 0:
                action_cat.append(Player.AGGRESSIVE_MOVE)

            if s['categories'] & Player.FAST_MOVE > 0:
                action_cat.append(Player.FAST_MOVE)

            if s['categories'] & Player.RELEASE_MOVE > 0:
                action_cat.append(Player.RELEASE_MOVE)

            if s['categories'] & Player.RANDOM_MOVE > 0:
                action_cat.append(Player.RANDOM_MOVE)

            for s in succ_list:
                print (str(s['action']) + " [" + str(s['categories']) + "] ")

            print ("")

        action_cat = list(set(action_cat))

        if self.train and random.uniform(0, 1) < self.epsilon:
            self.old_to_new_cat = action_cat[random.randint(0, len(action_cat) - 1)]
        else:
            q_values = list()

            for i in action_cat:
                q_values.append(self.nn.evaluate(self.board_state_and_category_to_nn_inputs(self.old_board_state, i)))

            max_q_value = max(q_values)

            if math.isnan(max_q_value):
                exit()
            else:
                cat_candidates = []

                for i in range(len(q_values)):
                    if q_values[i] == max_q_value:
                        cat_candidates.append(i)

                self.old_to_new_cat = action_cat[cat_candidates[random.randint(0, len(cat_candidates) - 1)]]

        s_with_category = []

        for i in range(len(succ_list)):
            if succ_list[i]['categories'] & self.old_to_new_cat > 0:
                s_with_category.append(i)

        succ_val_ind = s_with_category[random.randint(0, len(s_with_category) - 1)]

        self.old_to_new_action = (succ_list[succ_val_ind]["action"][0], succ_list[succ_val_ind]["action"][1])
        self.new_board_state = succ_list[succ_val_ind]["new_state"]

        cur_board_state = curr_st_board
        action = self.old_to_new_action
        new_board_state = self.new_board_state

        if new_board_state[self.id].state[tot_sq] == 4*one_piece_val:
            self.cum_reward += 4*one_piece_val

            try:
                cur_board_state[(self.id + 1) % tot_ply].cummulative_rew += -4 * one_piece_val
            except AttributeError:
                pass

            try:
                cur_board_state[(self.id + 2) % tot_ply].cummulative_rew += -4 * one_piece_val
            except AttributeError:
                pass

            try:
                cur_board_state[(self.id + 3) % tot_ply].cummulative_rew += -4 * one_piece_val
            except AttributeError:
                pass

        if new_board_state[self.id].state[0] < cur_board_state[self.id].state[0]:
            self.cum_reward += 5.25

        src_piece_loc = action[0]

        if 1 <= src_piece_loc <= 51:
            safe_sqs = [1, 9, 14, 22, 27, 35, 40, 48]
            if src_piece_loc not in safe_sqs:
                if cur_board_state[self.id].state[src_piece_loc] == one_piece_val:
                    can_be_att = False

                    for new_playr in range(1, tot_ply):
                        src_piece_loc_in_np = (src_piece_loc - 13 * new_playr) % comm_sq
                        for og_playr in range(1, comm_sq):
                            if cur_board_state[(self.id + new_playr) % 4].state[og_playr] == 0:
                                continue

                            if 0 < src_piece_loc_in_np - og_playr <= d_max:
                                can_be_att = True
                                break

                        if can_be_att:
                            self.cum_reward += 5.2
                            break
        for new_playr in range(1, tot_ply):
            curr_st_val = cur_board_state[(self.id - new_playr) % 4].state[0]
            new_st_val = new_board_state[(self.id - new_playr) % 4].state[0]

            if new_st_val - curr_st_val > 0:
                self.cum_reward += 4.15 * (new_st_val - curr_st_val) * 4
        diff = new_board_state[(self.id - 1) % 4].state[0] - cur_board_state[(self.id - 1) % 4].state[0]

        if diff > 0 and ts > 0 and cur_board_state[(self.id - 1) % 4].timestamp == ts - 1:
            try:
                cur_board_state[(self.id - 1) % 4].cummulative_rew += -4.25
            except AttributeError:
                pass

        cur_closest = tot_sq-1

        while True:
            if cur_closest == 0:
                break
            elif cur_board_state[self.id].state[cur_closest] > 0:
                if new_board_state[self.id].state[cur_closest] < cur_board_state[self.id].state[cur_closest]:
                    self.cum_reward += 3.1
                break
            else:
                cur_closest -= 1
        for play_location in range(1, comm_sq):
            safe_sqs = [1, 9, 14, 22, 27, 35, 40, 48]
            if play_location in safe_sqs:
                continue

            if new_board_state[self.id].state[play_location] >= (2*one_piece_val) > cur_board_state[self.id].state[play_location]:
                self.cum_reward += 1.05
                break
        for i in range(tot_ply):
            try:
                cur_board_state[(self.id - i) % tot_ply].reward()
            except AttributeError:
                pass

        return succ_val_ind
