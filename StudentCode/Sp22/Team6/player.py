import copy


class PlayerKind:
    #Indexes for different types of players
    Random = 0
    QLearning = 1
    Fast = 2
    Aggressive = 3
    Defensive = 4
    Mixed = 5
    Human = 6

def GetKind(kind):
    return {
        PlayerKind.Random: 'R',
        PlayerKind.QLearning: 'Q',
        PlayerKind.Fast: 'F',
        PlayerKind.Aggressive: 'A',
        PlayerKind.Defensive: 'D',
        PlayerKind.Mixed: 'M',
        PlayerKind.Human: 'H',
    }[kind]

def GetFullKind(kind):
    return {
        PlayerKind.Random: 'Random',
        PlayerKind.QLearning: 'Q-Learning',
        PlayerKind.Fast: 'Fast',
        PlayerKind.Aggressive: 'Aggressive',
        PlayerKind.Defensive: 'Defensive',
        PlayerKind.Mixed: 'Mixed-Strategy',
        PlayerKind.Human: 'Human',
    }[kind]

SafeSquares = [0, 1, 9, 14, 22, 27, 35, 40, 48]


class Player(object):
    DEFENSIVE_MOVE = 1
    AGGRESSIVE_MOVE = 2
    FAST_MOVE = 4
    RELEASE_MOVE = 8
    RANDOM_MOVE = 16

    def __init__(self, id, kind):
        self.id = id

        self.kind = kind
        self.state = [0.0, ] * 59

        self.board_state = None

        self.action = None

        self.timestamp = -1

    def __deepcopy__(self, memo):

        ply_copy = type(self)(self.id)
        for i in range(len(self.state)):
            ply_copy.state[i] = self.state[i]

        return ply_copy

    def get_c_track_pieces_next_player(self, board_state, order, position):
        state_to_ret = 52
        next_pos = (position - 13 * order) % 52
        if next_pos == 0:
            return board_state[(self.id + order) % 4].state[state_to_ret]

        state_to_ret = next_pos
        return board_state[(self.id + order) % 4].state[state_to_ret]

    def set_c_track_pieces_next_player(self, board_state, seq, position, val_update):
        tot_playable_squares = 52
        tot_players = 4

        next_pos = (position - 13 * seq) % tot_playable_squares

        if next_pos == 0:
            board_state[(self.id + seq) % tot_players].state[tot_playable_squares] = val_update

        board_state[(self.id + seq) % tot_players].state[next_pos] = val_update

    def get_simple_state(self, state):
        ret_state = list()

        for i in range(59):
            nPlayers = state[i] * 4
            while (nPlayers > 0):
                ret_state.append(i)
                nPlayers = nPlayers - 1

        return ret_state

    def get_simple_board_state(self, board_state):
        ret_board_st = dict()

        for i in range(4):
            ret_board_st[i] = self.get_simple_state(board_state[i].state)

        return ret_board_st

    def get_relative_position(self, rel_player_id, rel_player_pos):

        tot_players = 4
        play_sq = 52

        if (rel_player_id == self.id):
            return rel_player_pos

        if rel_player_pos >= play_sq or rel_player_pos == 0:
            #other player not in common play area
            return -1

        rel_pos = (rel_player_pos + 13 * ((rel_player_id - self.id + tot_players) % tot_players)) % play_sq

        if rel_pos != 0:
            return rel_pos
        else:
            return play_sq

    def get_simple_relative_board_state(self, simple_board_state):

        simple_relative_board_state = dict()

        for player in range(4):
            if (player == self.id):
                list_of_players_states = list(simple_board_state[player])
                simple_relative_board_state[player] = list_of_players_states

            simple_relative_board_state[player] = list()

            for pos in simple_board_state[player]:
                simple_relative_board_state[player].append(self.get_relative_position(player, pos))

        return simple_relative_board_state

    def transition_is_defensive(self, last_state_board, action, new_st):

        player = action[0]
        list_of_safe = [1, 9, 14, 22, 27, 35, 40, 48]
        tot_play_sq = 52
        single_piece_val = 0.25

        if 1 <= player <= 51:
            if player not in list_of_safe:
                if last_state_board[self.id].state[player] == single_piece_val:
                    can_be_att = False

                    for i in range(1, 4):
                        src_piece_loc_in_np = (player - 13 * i) % tot_play_sq

                        for j in range(1, tot_play_sq):
                            if last_state_board[(self.id + i) % 4].state[j] == 0:
                                continue

                            if 0 < src_piece_loc_in_np - j <= 6:
                                can_be_att = True
                                break

                        if can_be_att:
                            return True

        return False

    def transition_is_aggressive(self, last_state_board, action, new_st):

        for i in range(1, 4):
            last_st_for_plyr = last_state_board[(self.id - i) % 4].state[0]
            new_st_for_plyr = new_st[(self.id - i) % 4].state[0]

            if new_st_for_plyr - last_st_for_plyr > 0:
                return True

        return False

    def transition_is_fast(self, last_state_board, action, new_st):
        piece_close_home = 57

        while True:
            if piece_close_home == 0:
                break
            elif last_state_board[self.id].state[piece_close_home] > 0:
                new_st_of_piece = new_st[self.id].state[piece_close_home]
                last_st_of_piece = last_state_board[self.id].state[piece_close_home]
                if new_st_of_piece < last_st_of_piece:
                    return True
                break
            else:
                piece_close_home -= 1

        return False

    def transition_is_release(self, old_board_state, action, new_board_state):
        return new_board_state[self.id].state[0] < old_board_state[self.id].state[0]

    def get_next_states(self, d_roll, curr_st_board):

        tot_sq = 58
        common_sq = 52
        one_piece_val = 0.25

        if curr_st_board is None:
            return None

        for p in curr_st_board:
            if p.state[tot_sq] == 1:
                return None

        succ = list()

        cur_player_board = curr_st_board[self.id].state

        if cur_player_board[0] > 0 and d_roll == 6:
            updated_succ = copy.deepcopy(curr_st_board)

            updated_succ[self.id].state[0] -= one_piece_val
            updated_succ[self.id].state[1] += one_piece_val

            new_successor_w_action = dict()
            new_successor_w_action['new_state'] = updated_succ
            new_successor_w_action['action'] = (0, 1)
            new_successor_w_action['categories'] = Player.RANDOM_MOVE


            succ.append(new_successor_w_action)


        for location in range(1, tot_sq):
            if cur_player_board[location] == 0:
                continue


            updated_succ = copy.deepcopy(curr_st_board)


            is_block = False

            temp_min_location = location + 1
            temp_max_location = temp_min_location + d_roll


            if temp_min_location <= common_sq < temp_max_location:
                temp_max_location += 1

            for temp_locations in range(temp_min_location, temp_max_location):

                if temp_locations == common_sq or (temp_locations >= tot_sq):
                    continue


                if temp_locations in SafeSquares:
                    continue


                if curr_st_board[self.id].state[temp_locations] >= 2*one_piece_val:
                    #2 pieces create a blockade
                    is_block = True
                    break


                if temp_locations < common_sq:
                    for io in range(1, 4):
                        if self.get_c_track_pieces_next_player(updated_succ, io, temp_locations) >= 2*one_piece_val:
                            is_block = True
                            break

                    if is_block:
                        break

            if is_block:
                continue


            new_loc = location + d_roll


            if location <= common_sq <= new_loc:
                new_loc += 1

            if new_loc <= tot_sq:
                updated_succ[self.id].state[location] -= one_piece_val
                updated_succ[self.id].state[new_loc] += one_piece_val


                if new_loc < common_sq and new_loc not in SafeSquares:
                    for io in range(1, 4):
                        knocked_opp = self.get_c_track_pieces_next_player(updated_succ, io, new_loc)

                        if self.get_c_track_pieces_next_player(updated_succ, io, new_loc) > 0:
                            self.set_c_track_pieces_next_player(updated_succ, io, new_loc, 0)
                            updated_succ[(self.id + io) % 4].state[0] += knocked_opp


                new_successor_w_action = dict()

                new_successor_w_action['new_state'] = updated_succ
                new_successor_w_action['action'] = (location, new_loc)
                new_successor_w_action['categories'] = Player.RANDOM_MOVE

                succ.append(new_successor_w_action)

        if len(succ) > 0:

            #checking if transition is def, agg, fast or release
            for s in succ:
                if self.transition_is_defensive(curr_st_board, s['action'], s['new_state']):
                    s['categories'] += Player.DEFENSIVE_MOVE

                if self.transition_is_aggressive(curr_st_board, s['action'], s['new_state']):
                    s['categories'] += Player.AGGRESSIVE_MOVE

                if self.transition_is_fast(curr_st_board, s['action'], s['new_state']):
                    s['categories'] += Player.FAST_MOVE

                if self.transition_is_release(curr_st_board, s['action'], s['new_state']):
                    s['categories'] += Player.RELEASE_MOVE

            return succ
        else:
            return None

    def board_state_and_action_to_nn_inputs(self, curr_st_board, action):

        compact = False
        one_piece_val = 0.25
        tot_sq = 58.0

        if not compact:
            nn_inps = [0.0, ] * 238

            i = 0

            for play_ord in range(4):
                for s in curr_st_board[(self.id + play_ord) % 4].state:
                    nn_inps[i] = s
                    i += 1

            nn_inps[236] = action[0] / tot_sq
            nn_inps[237] = action[1] / tot_sq

            return nn_inps
        else:
            nn_inps = [0.0, ] * 18
            ith_inp = 0

            tmp = copy.deepcopy(curr_st_board)

            for k in range(4):
                for i in range(len(tmp[(self.id + k) % 4].state)):
                    while tmp[(self.id + k) % 4].state[i] > 0.0:
                        nn_inps[ith_inp] = i
                        ith_inp += 1
                        tmp[(self.id + k) % 4].state[i] -= one_piece_val

            nn_inps[16] = action[0]
            nn_inps[17] = action[1]

            for i in range(len(nn_inps)):
                nn_inps[i] /= tot_sq

            return nn_inps

    def move(self, d_roll, ply_on_board, ts):

        successors = self.get_next_states(d_roll, ply_on_board)

        if successors is not None:
            successor = successors[self.select_new_state(ply_on_board, successors, ts)]

            self.board_state = copy.deepcopy(successor["new_state"])
            self.action = (successor["action"][0], successor["action"][1])
            self.timestamp = ts


            for p in range(len(ply_on_board)):
                for l in range(len(ply_on_board[p].state)):
                    ply_on_board[p].state[l] = self.board_state[p].state[l]

    def select_new_state(self, board_state, successors, timestamp):
        return 0
