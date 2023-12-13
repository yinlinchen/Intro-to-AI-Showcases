# Contains heuristic evaluation functions for the GameState

from game import GameState
from piece import Piece, Cat, Kitten, PlayerID
from board import Board, Point, Pair, Triple, on_bed
import util


# Tuning constants for scores.
WIN_BONUS = 1000.0
PLY_PENALTY = 0.1

# A factor such that current player score counts more than other's negative,
# to encourage offensive
ANTI_AGGRESSION = 0.8

# a cat must be worth more than 3x kittens,
# and more than pending triple to make promotion worth it
CAT_MULTIPLIER = 25.0
KITTEN_MULTIPLIER = 1.0
BOARD_MULTIPLIER = 2.0
CENTER_MULTIPLIER = 2.5

# Encourage moves that will lead to better moves:
# These are, in a way, shortcuts to evaluate strong positions
# at prior depths.
PENDING_TRIPLE_BONUS = 4.0     # encourage getting triples
PENDING_PROMOTION_BONUS = 3.5  # encourage getting all pieces onto board

STRANDED_CAT_PENALTY = 1.0

def eval_piece_count(id: PlayerID, state: GameState):
    """
    A simple evaluation heuristic that provides a score
    based on the number of piece counts.
    Cats are worth more than Kittens.
    
    This assumes MAX is player 1, and thus subtracts the score
    of player 2.

    For all heuristics, a win is still factored in,
    as is the ply penalty.
    """

    p1_score = _get_piece_score(PlayerID.ONE, state)
    p2_score = _get_piece_score(PlayerID.TWO, state)

    # apply anti-aggression measure, adjusting conservative vs offensive play
    if id == PlayerID.TWO:
        p1_score *= ANTI_AGGRESSION
    elif id == PlayerID.ONE:
        p2_score *= ANTI_AGGRESSION
    ply_penalty = _get_ply_penalty(id, state)
    win_bonus =_get_win_bonus(state)

    # incentivize the win, no point in continuing for a "better" win
    if win_bonus != 0:
        return win_bonus + ply_penalty
    return p1_score - p2_score + ply_penalty


def eval_board_bonus(id: PlayerID, state: GameState):
    """
    A simple evaluation heuristic that provides a score
    based on the number of piece counts.
    A piece on the board is considered worth more than a piece
    in hand.

    This assumes MAX is player 1, and thus subtracts the score
    of player 2.

    For all heuristics, a win is still factored in.
    as is the ply penalty.
    """
    p1_score = _get_piece_score_with_board_bonus(PlayerID.ONE, state)
    p2_score = _get_piece_score_with_board_bonus(PlayerID.TWO, state)
    
    # apply anti-aggression measure, adjusting conservative vs offensive play
    if id == PlayerID.TWO:
        p1_score *= ANTI_AGGRESSION
    elif id == PlayerID.ONE:
        p2_score *= ANTI_AGGRESSION

    ply_penalty = _get_ply_penalty(id, state)
    win_bonus =_get_win_bonus(state)
    if win_bonus != 0:
        return win_bonus + ply_penalty
    return p1_score - p2_score + ply_penalty
    


def eval_territory(id: PlayerID, state: GameState):
    """
    A more complex heuristic.
    This factors the piece count such that cats are worth more
    than kittens, but also factors in the value of placement on the board.
    - Center pieces are considered stronger
    - Having 2 pieces in a row gets a bonus (pending triple) unless
      already blocked by the opponent.
    - Having the "L" form of pending triple gets a bonus.
    - Having 7 on the board gets a bonus.
    - Bonuses are stronger if cats.
    
    This assumes MAX is player 1, and thus subtracts the score
    of player 2.

    For all heuristics, a win is still factored in.
    """
    p1_score = _get_territory_score(PlayerID.ONE, state)
    p2_score = _get_territory_score(PlayerID.TWO, state)

    # apply anti-aggression measure, adjusting conservative vs offensive play
    if id == PlayerID.TWO:
        p1_score *= ANTI_AGGRESSION
    elif id == PlayerID.ONE:
        p2_score *= ANTI_AGGRESSION

    ply_penalty = _get_ply_penalty(id, state)
    win_bonus =_get_win_bonus(state)
    if win_bonus != 0:
        return win_bonus + ply_penalty
    return p1_score - p2_score + ply_penalty
    


def eval_stranding(id: PlayerID, state: GameState):
    """
    A more complex heuristic.
    This is the similar as eval_territory, but additionally
    will provide a slight penalty when cats are stranded
    away from other pieces, if no more cats are
    remaining in the player's hand. This is a more advanced
    strategy that can orphan the opponent's cats, such that
    they lose the ability to block cats or to ever reclaim their
    cats without further promotion.
    
    This assumes MAX is player 1, and thus subtracts the score
    of player 2.

    For all heuristics, a win is still factored in.
    """
    p1_score = _get_territory_score(PlayerID.ONE, state)
    p1_score += _get_stranded_cat_penalty(PlayerID.ONE, state)
    p2_score = _get_territory_score(PlayerID.TWO, state)
    p2_score += _get_stranded_cat_penalty(PlayerID.TWO, state)

    # apply anti-aggression measure, adjusting conservative vs offensive play
    if id == PlayerID.TWO:
        p1_score *= ANTI_AGGRESSION
    elif id == PlayerID.ONE:
        p2_score *= ANTI_AGGRESSION

    ply_penalty = _get_ply_penalty(id, state)
    win_bonus =_get_win_bonus(state)
    if win_bonus != 0:
        return win_bonus + ply_penalty
    return p1_score - p2_score + ply_penalty


def _get_win_bonus(state: GameState) -> float:
    """
    Helper to return the win bonus, given the winner,
    if the state is terminal.
    """
    win_bonus = 0
    if state.is_terminal():
        win_bonus = WIN_BONUS if state.winner == PlayerID.ONE else -WIN_BONUS
    return win_bonus


def _get_ply_penalty(id: PlayerID, state: GameState) -> float:
    """
    Helper to return ply penalty.
    """
    penalty = state.plies * PLY_PENALTY
    if id == PlayerID.ONE:
        return -penalty
    return penalty


def _get_piece_score(id: PlayerID, state: GameState) -> float:
    """
    Helper to get the piece score.
    Cats are worth more than Kittens.
    """
    on_board = state.get_pieces_on_board(id)
    cat_score = 0
    kitten_score = 0
    for (piece, _, _) in on_board:
        if type(piece) is Cat:
            cat_score += 1
        else:
            kitten_score += 1
    
    player = state.get_player(id)
    cat_score += player.hand.num_cats()
    kitten_score += player.hand.num_kittens()
    cat_score *= CAT_MULTIPLIER
    kitten_score *= KITTEN_MULTIPLIER
    score = cat_score + kitten_score
    return score


def _get_piece_score_with_board_bonus(id: PlayerID, state: GameState) -> float:
    """
    Helper to get the piece score, factoring in a bonus for pieces
    placed on the board.

    Cats are worth more than Kittens.
    """
    on_board = state.get_pieces_on_board(id)
    cat_score = 0
    kitten_score = 0
    for (piece, _, _) in on_board:
        if type(piece) is Cat:
            cat_score += 1
        else:
            kitten_score += 1

    # now, factor in the board scores
    cat_score = cat_score * CAT_MULTIPLIER * BOARD_MULTIPLIER
    kitten_score = kitten_score * KITTEN_MULTIPLIER * BOARD_MULTIPLIER
    
    # now add in the the hand scores
    player = state.get_player(id)
    cat_score += player.hand.num_cats() * CAT_MULTIPLIER
    kitten_score += player.hand.num_kittens() * KITTEN_MULTIPLIER
    score = cat_score + kitten_score
    return score


def _get_territory_score(id: PlayerID, state: GameState) -> float:
    """
    Helper for territory score of this player.
    - Center pieces are considered stronger
    - Having 2 pieces in a row gets a bonus (pending triple) unless
      already blocked by the opponent.
    - Having the "L" form of pending triple gets a bonus.
    - Having 7 on the board gets a bonus.
    - Bonus is stronger if cats.
    """
    on_board = state.get_pieces_on_board(id)
    score = 0
    
    # track so we don't double-count pending triples, but still allow a piece
    # to count towards more than one!
    matched_doubles = {}
    matched_Ls = []
    all_cats = True
    for (piece, x, y) in on_board:
        if type(piece) is Kitten:
            all_cats = False

        # check triples
        pending_score = _get_pending_triple_score(piece, x, y, matched_doubles, state)
        if pending_score > 0:
            score += pending_score
            continue

        # check Ls
        pending_score = _get_pending_L_score(piece, x, y, matched_Ls, state)
        if pending_score > 0:
            score += pending_score
            continue


        bonus = CAT_MULTIPLIER if type(piece) is Cat else KITTEN_MULTIPLIER
        if _is_piece_in_center(x, y):
            bonus *= CENTER_MULTIPLIER
        else:
            bonus *= BOARD_MULTIPLIER

        score += bonus
    

    player = state.get_player(id)

    # check if 7 pieces are on board
    if len(on_board) == 7:
        bonus = PENDING_PROMOTION_BONUS
        if all_cats and not player.hand.has_kitten():
            bonus *= CAT_MULTIPLIER  # close to a win!
        score += bonus


    # now add in the the hand scores
    score += player.hand.num_cats() * CAT_MULTIPLIER
    score += player.hand.num_kittens() * KITTEN_MULTIPLIER
    return score
        

def _get_pending_triple_score(piece: Piece, x: int, y: int,
                              matched: dict[Point,Pair], state: GameState) -> float:
    """
    Gets score bonus for pending triples for this piece, if any.
    """
    adjacent = Board.get_completion_points(x, y)
    already_matched = matched.get((x, y), [])
    
    # this is bounded by low max of 7 in outer loop
    score = 0.0
    for pair in adjacent:
        pending = False
        pending_pt = None
        for pt in pair:
            # we already factored a bonus for these 2 pts
            if pt in already_matched:
                continue

            # do we have another in a row?
            adj_piece = state.board[pt[0]][pt[1]]
            if adj_piece is None:
                continue

            if adj_piece.same_player(piece):
                pending = True
                pending_pt = pt
            else:
                # blocked by opponent already
                pending = False
                break

        if pending:
            # reverse track so we don't count twice for same triple
            if pending_pt in matched:
                matched[pending_pt].append((x,y))
            else:
                matched[pending_pt] = [(x,y)]
            
            # update score
            # NOTE: using a cat multiplier here is detrimental, as
            # can be stronger than actually completing the triple and getting
            # a new cat
            score += PENDING_TRIPLE_BONUS
    
    return score


def _get_pending_L_score(piece: Piece, x: int, y: int,
                         matched: list[Triple], state: GameState) -> float:
    """
    Checks for a diagonal boop that would complete a triple.
    """

    # For example, a piece at (5, 1) would complete an L-based
    # triple for Player1:
    #   |------|------|------|------|------|------|
    #  5|      |      |      | K(2) |      |      |
    #   |------|------|------|------|------|------|
    #  4|      |      |      |      | K(1) |      |
    #   |------|------|------|------|------|------|
    #  3|      |      |      |      |      |      |
    #   |------|------|------|------|------|------|
    #  2| K(2) |      | K(1) |      | K(1) |      |
    #   |------|------|------|------|------|------|
    #  1| K(2) |      |      |      |      |      |
    #   |------|------|------|------|------|------|
    #  0|      |      |      |      |      |      |
    #   |------|------|------|------|------|------|
    #      0      1      2      3      4      5
    
    # The pieces at (2,2), (4,2), and (4, 4) make
    # a backwards L shape.

    score = 0.0
    comps = Board.get_L_compeltion_points(x, y)
    for pair in comps:
        corner = pair[0]
        other = pair[1]
        triple = [(x, y), corner, other]

        # make sure we didn't match this already:
        # we expect this to be empty or nearly empty
        if util.any_match(triple, matched):
            continue
        
        # check if the middle is blocked, in which case this is moot
        mid_x = (x + other[0])//2
        mid_y = (y + other[1])//2
        if state.board[mid_x][mid_y] is not None:
            continue

        # check if pieces are there and are ours
        valid_L = True
        for pt in pair:
            other_piece = state.board[pt[0]][pt[1]]
            if other_piece is None or not other_piece.same_player(piece):
                valid_L = False
                break

        if not valid_L:
            continue

        # check if the square needed to boop in other direction is blocked
        (bx, by) = Board.boop_vector(mid_x, mid_y, corner[0], corner[1])
        next_x = corner[0] + bx
        next_y = corner[1] + by
        if on_bed(next_x, next_y) and state.board[next_x][next_y] is not None:
            continue

        # we got one!
        score += PENDING_TRIPLE_BONUS

        # track so we don't count twice
        matched.append(triple)
    
    return score


def _get_stranded_cat_penalty(id: PlayerID, state: GameState) -> float:
    """
    Helper to return penalty for cats that the player can't recover
    on their own.
    """
    on_board = state.get_pieces_on_board(id)
    penalty = 0.0
    if state.get_player(id).hand.has_cat():
        return penalty
 
    for (piece, x, y) in on_board:
        if type(piece) is Kitten:
            continue

        # if a cat, check if any surrounding squares have pieces
        surrounding = Board.surrounding_squares(x, y)
        found = False
        for (sx, sy) in surrounding:
            other = state.board[sx][sy]
            if other is None:
                continue
            if not other.same_player(piece):
                continue
            found = True
            break
        if not found:
            penalty += STRANDED_CAT_PENALTY

    return -penalty if id == PlayerID.ONE else penalty


def _is_piece_in_center(x, y) -> bool:
    """
    Returns True if piece is in a center square, False otherwise.
    """
    return x >= 2 and x <= 3 and y >= 2 and y <= 3