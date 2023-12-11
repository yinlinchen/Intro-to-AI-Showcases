from game import Game, GameState
from player import Player
from piece import PlayerID, Cat, Kitten
import util

def test_init_gamestate():
    p1 = Player(PlayerID.ONE)
    p2 = Player(PlayerID.TWO)
    gs = GameState(p1, p2)
    assert gs.player1 == p1
    assert gs.player2 == p2
    assert gs.turn == PlayerID.ONE
    assert gs.current_player() == p1
    assert gs.board is not None
    assert len(gs.board.empty_squares()) == 36


def test_advance_turn():
    p1 = Player(PlayerID.ONE)
    p2 = Player(PlayerID.TWO)
    gs = GameState(p1, p2)
    assert gs.turn == PlayerID.ONE
    assert gs.current_player() == p1
    assert gs.other_player() == p2
    
    gs.advance_turn()
    assert gs.turn == PlayerID.TWO
    assert gs.current_player() == p2
    assert gs.other_player() == p1

    gs.advance_turn()
    assert gs.turn == PlayerID.ONE
    assert gs.current_player() == p1
    assert gs.other_player() == p2


def test_copy():
    # apply changes, then copy the state
    p1 = Player(PlayerID.ONE)
    p2 = Player(PlayerID.TWO)
    gs = GameState(p1, p2)
    
    p2.hand.get_kitten()
    p2.hand.add_piece(Cat(p2.id))

    gs.board[2][3] = p1.hand.get_kitten()
    gs.advance_turn()
    gs.must_promote = True


    copy = gs.copy()
    assert not copy.player1 is p1 # new ref
    assert copy.player1.id == p1.id

    assert not copy.player2 is p2
    assert copy.player2.id == p2.id

    assert not copy.player1.hand is p1.hand
    assert not copy.player2.hand is p2.hand
    assert not copy.player1.decisions is p1.decisions
    assert not copy.player2.decisions is p2.decisions

    assert copy.player1.hand.num_cats() == p1.hand.num_cats()
    assert copy.player2.hand.num_cats() == p2.hand.num_cats()

    assert copy.player1.hand.num_kittens() == p1.hand.num_kittens()
    assert copy.player2.hand.num_kittens() == p2.hand.num_kittens()

    assert copy.turn == gs.turn
    assert not copy.board is gs.board
    assert copy.board == gs.board
    assert copy.must_promote == gs.must_promote

    assert copy.winner == gs.winner
    piece = copy.board[2][3]
    assert type(piece) is Kitten
    assert piece.get_player() == p1.id



def simulate_early_game():
    # helper to create GameState after a few rounds of play,
    # without relying on the methods under test
    p1 = Player(PlayerID.ONE)
    p2 = Player(PlayerID.TWO)
    gs = GameState(p1, p2)

    p1_moves = [(1,1), (2,2), (4,4)]
    p2_moves = [(0,1), (0,2), (3,5)]

    for (x, y) in p1_moves:
        gs.board[x][y] = p1.hand.get_kitten()

    for (x, y) in p2_moves:
        gs.board[x][y] = p2.hand.get_kitten()

    return gs


def simulate_advanced_game():
    # similar helper, with addition of Cats in play
    gs = simulate_early_game()
    gs.player1.hand.get_kitten()
    gs.player2.hand.add_piece(Cat(PlayerID.TWO))

    gs.board[0][0] = Cat(PlayerID.ONE)
    gs.advance_turn()
    return gs


def test_get_legal_actions_kittens_only_on_board():
    # mark up the board, ensure only legal actions permitted for this player
    gs = simulate_early_game()
    actions = gs.get_legal_actions()
    # expect all blank squares available, which we know works from board unit test
    spaces = gs.board.empty_squares()

    # since p1 should have only kittens, only expect kittens.
    # action format is (Piece, x, y)
    expected = list(map(lambda pt: (Kitten(PlayerID.ONE), pt[0], pt[1]), spaces))
    assert util.list_match(expected, actions)


def test_get_legal_actions_cats():
    gs = simulate_advanced_game()
    actions = gs.get_legal_actions()
    spaces = gs.board.empty_squares()
    expected = []
    # player 2 has a cat too!
    for pt in spaces:
        expected.append((Kitten(PlayerID.TWO), pt[0], pt[1]))
        expected.append((Cat(PlayerID.TWO), pt[0], pt[1]))
    
    assert util.list_match(expected, actions)


def test_apply_action_basic_move():
    gs = simulate_early_game()

    # this move causes no boops
    num_kittens = gs.player1.hand.num_kittens()
    action = (Kitten(PlayerID.ONE), 0, 5)

    gs.apply_action(action)

    # check the state
    assert gs.player1.hand.num_kittens() == num_kittens-1
    assert not gs.has_pending_decision()
    assert gs.winner is None
    assert not gs.must_promote
    assert gs.turn == PlayerID.TWO

    piece = gs.board[0][5]
    assert piece == Kitten(PlayerID.ONE)


def test_apply_action_causes_boops():
    gs = simulate_early_game()
    
    action = (Kitten(PlayerID.ONE), 2, 5)

    gs.apply_action(action)

    # check the state
    assert not gs.has_pending_decision()
    assert gs.winner is None
    assert not gs.must_promote
    assert gs.turn == PlayerID.TWO

    piece = gs.board[2][5]
    assert piece == Kitten(PlayerID.ONE)

    piece = gs.board[3][5]
    assert piece is None

    piece = gs.board[4][5]
    assert piece == Kitten(PlayerID.TWO)


def test_apply_action_causes_boops_off_bed():
    gs = simulate_early_game()

    action = (Kitten(PlayerID.ONE), 3, 4)

    p2_kittens = gs.player2.hand.num_kittens()

    gs.apply_action(action)

    # check the state
    assert not gs.has_pending_decision()
    assert gs.winner is None
    assert not gs.must_promote
    assert gs.turn == PlayerID.TWO

    piece = gs.board[3][4]
    assert piece == Kitten(PlayerID.ONE)

    piece = gs.board[3][5] # booped off bed!
    assert piece is None

    piece = gs.board[5][4]
    assert piece == Kitten(PlayerID.ONE)

    assert gs.player2.hand.num_kittens() == p2_kittens + 1


def test_apply_action_makes_triple():
    gs = simulate_early_game()

    # this will lead to a triple for Player 1
    action = (Kitten(PlayerID.ONE), 3, 3)
    p1_kittens = gs.player1.hand.num_kittens()
    p1_cats = gs.player1.hand.num_cats()

    gs.apply_action(action)

    # check the state
    assert not gs.has_pending_decision()
    assert gs.winner is None
    assert not gs.must_promote
    assert gs.turn == PlayerID.TWO
    
    # this is the triple
    piece = gs.board[3][3]
    assert piece is None

    piece = gs.board[2][2]
    assert piece is None

    piece = gs.board[1][1]
    assert piece is None

    # this got booped to (5, 5)
    piece = gs.board[4][4]
    assert piece is None

    piece = gs.board[5][5]
    assert piece == Kitten(PlayerID.ONE)

    # verify hand has cats now!
    assert gs.player1.hand.num_kittens() == p1_kittens - 1  # one of these became a cat!
    assert gs.player1.hand.num_cats() == p1_cats + 3  # we promoted 3 cats!


def test_apply_action_then_requires_decision():
    gs = simulate_early_game()

    # adding extra pieces to set this up
    gs.board[4][2] = Kitten(PlayerID.ONE)
    gs.board[0][5] = Kitten(PlayerID.TWO)

    # this will lead to a DECISION! for Player 1
    action = (Kitten(PlayerID.ONE), 5, 1)
    p1_kittens = gs.player1.hand.num_kittens()
    p1_cats = gs.player1.hand.num_cats()

    gs.apply_action(action)

    # check the state
    assert gs.winner is None
    assert not gs.must_promote
    assert gs.turn == PlayerID.ONE  # did NOT change yet, pending!
    assert gs.has_pending_decision() # this time, yes!

    # these are the two sets of triples, overlapping here
    piece = gs.board[3][3]
    assert piece == Kitten(PlayerID.ONE)

    piece = gs.board[2][2]
    assert piece == Kitten(PlayerID.ONE)

    piece = gs.board[1][1]
    assert piece == Kitten(PlayerID.ONE)

    piece = gs.board[4][4]
    assert piece == Kitten(PlayerID.ONE)

    # this new piece is still here
    piece = gs.board[5][1]
    assert piece == Kitten(PlayerID.ONE)

    # verify hand has decisions now!
    assert gs.player1.hand.num_kittens() == p1_kittens - 1  # piece on board
    assert gs.player1.hand.num_cats() == p1_cats  # no change yet!
   
    assert gs.player1.pending_decision()

    expected_decisions = [
        [(1, 1), (2, 2), (3, 3)],
        [(2, 2), (3, 3), (4, 4)]
    ]
    assert util.all_have_match(expected_decisions, gs.player1.decisions)


def create_gs_pending_both_decisions():
    # helper to create the game state where next move (at 3,3)
    # leads to pending decisions for player 1 and player 2
    p1 = Player(PlayerID.ONE)
    p2 = Player(PlayerID.TWO)
    gs = GameState(p1, p2)

    gs.board[1][1] = p1.hand.get_kitten()
    gs.board[1][2] = p1.hand.get_kitten()
    gs.board[1][4] = p1.hand.get_kitten()

    gs.board[2][3] = p1.hand.get_kitten()

    gs.board[4][3] = p2.hand.get_kitten()

    gs.board[5][1] = p2.hand.get_kitten()
    gs.board[5][2] = p2.hand.get_kitten()
    gs.board[5][4] = p2.hand.get_kitten()
    return gs


def test_apply_action_then_requires_multiple_decisions():
    gs = create_gs_pending_both_decisions()

    # this will lead to a DECISION for both players!
    action = (Kitten(PlayerID.ONE), 3, 3)
    p1_kittens = gs.player1.hand.num_kittens()
    p1_cats = gs.player1.hand.num_cats()
    p2_kittens = gs.player2.hand.num_kittens()
    p2_cats = gs.player2.hand.num_cats()

    gs.apply_action(action)

    # check the state
    assert gs.winner is None
    assert not gs.must_promote
    assert gs.turn == PlayerID.ONE  # did NOT change yet, pending!
    assert gs.has_pending_decision() # this time, yes!

    # this is the new piece
    assert gs.board[3][3] == Kitten(PlayerID.ONE)
    
    # these should all still be here
    for i in range (1, 5):
        assert gs.board[1][i] == Kitten(PlayerID.ONE)
        assert gs.board[5][i] == Kitten(PlayerID.TWO)

    # verify hand has decisions now
    assert gs.player1.hand.num_kittens() == p1_kittens - 1  # piece on board
    assert gs.player1.hand.num_cats() == p1_cats  # no change yet!

    assert gs.player2.hand.num_kittens() == p2_kittens  # no change!
    assert gs.player2.hand.num_cats() == p2_cats  # no change yet!
   
    assert gs.player1.pending_decision()
    assert gs.player2.pending_decision()

    expected_p1 = [
        [(1, 1), (1, 2), (1, 3)],
        [(1, 2), (1, 3), (1, 4)]
    ]
    expected_p2 = [
        [(5, 1), (5, 2), (5, 3)],
        [(5, 2), (5, 3), (5, 4)]
    ]

    assert util.all_have_match(expected_p1, gs.player1.decisions)
    assert util.all_have_match(expected_p2, gs.player2.decisions)

    # test the state-level query
    assert util.all_have_match(expected_p1, gs.get_legal_selections(PlayerID.ONE))
    assert util.all_have_match(expected_p2, gs.get_legal_selections(PlayerID.TWO))


def create_gs_pending_promotion():
    # helper to create the game state where next p1 move
    # leads to a default promotion
    p1 = Player(PlayerID.ONE)
    p2 = Player(PlayerID.TWO)
    gs = GameState(p1, p2)

    gs.board[0][0] = p1.hand.get_kitten()
    gs.board[0][1] = p2.hand.get_kitten()
    gs.board[0][2] = p1.hand.get_kitten()

    gs.board[2][0] = p1.hand.get_kitten()
    gs.board[2][1] = p2.hand.get_kitten()
    gs.board[2][2] = p1.hand.get_kitten()
    

    gs.board[4][0] = p1.hand.get_kitten()
    gs.board[4][1] = p2.hand.get_kitten()
    gs.board[4][2] = p1.hand.get_kitten()

    gs.board[0][5] = p1.hand.get_kitten()
    return gs


def test_apply_action_then_requires_promotion():
    gs = create_gs_pending_promotion()

    action = (Kitten(PlayerID.ONE), 5, 5)

    p1_cats = gs.player1.hand.num_cats()

    gs.apply_action(action)

    assert gs.winner is None
    assert gs.must_promote  # now True!
    assert gs.turn == PlayerID.ONE  # did NOT change yet, pending!
    assert gs.has_pending_decision() # this time, yes!

    assert gs.board[5][5] == Kitten(PlayerID.ONE)

    assert not gs.player1.hand.has_kitten()  # all on board
    assert gs.player1.hand.num_cats() == p1_cats  # no change yet!
    assert not gs.player1.pending_decision()  # by this api, no
    
    promotions = gs.get_legal_promotions()
    expected = [(0, 0), (0, 2), (2, 0), (2, 2), (4, 0), (4, 2), (0, 5), (5, 5)]
    assert util.list_match(expected, promotions)


def test_apply_action_leads_to_winner_triple():
    p1 = Player(PlayerID.ONE)
    p2 = Player(PlayerID.TWO)
    gs = GameState(p1, p2)

    # promoting 3 cats (2 on board, 1 must be in hand)
    p1.hand.get_kitten()
    p1.hand.get_kitten()
    p1.hand.get_kitten()

    p1.hand.add_piece(Cat(PlayerID.ONE))

    gs.board[0][0] = Cat(PlayerID.ONE)
    gs.board[0][1] = Cat(PlayerID.ONE)

    action = (Cat(PlayerID.ONE), 0, 2)
    gs.apply_action(action)

    for i in range(0, 3):
        assert gs.board[0][i] == Cat(PlayerID.ONE)

    assert gs.winner == PlayerID.ONE
    assert gs.turn == PlayerID.ONE
    assert gs.is_terminal()
    assert not gs.has_pending_decision()


def test_apply_actions_leads_to_winner_eight_cats():
    p1 = Player(PlayerID.ONE)
    p2 = Player(PlayerID.TWO)
    gs = GameState(p1, p2)

    gs.board[0][0] = p1.hand.get_cat()
    gs.board[0][2] = p1.hand.get_cat()

    gs.board[2][0] = p1.hand.get_cat()
    gs.board[2][2] = p1.hand.get_cat()
    
    gs.board[4][0] = p1.hand.get_cat()
    gs.board[4][2] = p1.hand.get_cat()

    gs.board[0][5] = p1.hand.get_cat()

    for i in range(0, 8):
        p1.hand.get_kitten()

    p1.hand.add_piece(Cat(PlayerID.ONE))

    action = (Cat(PlayerID.ONE), 5, 5)
    gs.apply_action(action)

    assert gs.winner == PlayerID.ONE
    assert gs.turn == PlayerID.ONE
    assert gs.is_terminal()
    assert not gs.has_pending_decision()


def test_resolve_selection():
    # might as well test both
    gs = create_gs_pending_both_decisions()
    p1_selection = [(1, 1), (1, 2), (1, 3)]
    p2_selection = [(5, 1), (5, 2), (5, 3)]

    # cause the pending decisions with move to 3, 3
    action = (Kitten(PlayerID.ONE), 3, 3)
    gs.apply_action(action)
    
    # we have decisions to resolve:
    assert gs.has_pending_decision()
    assert gs.turn == PlayerID.ONE
    assert gs.player1.pending_decision()
    assert gs.player2.pending_decision()

    # resolve p1
    gs.resolve_selection(PlayerID.ONE, p1_selection)

    # verify board
    for pt in p1_selection:
        assert gs.board[pt[0]][pt[1]] is None

    # We expect that the turn has NOT yet advanced!
    assert gs.turn == PlayerID.ONE
    assert gs.has_pending_decision()
    
    assert not gs.player1.pending_decision()
    assert gs.player2.pending_decision()
    assert gs.has_pending_decision()

    # resolve p2
    gs.resolve_selection(PlayerID.TWO, p2_selection)
    assert not gs.player2.pending_decision()

    for pt in p2_selection:
        assert gs.board[pt[0]][pt[1]] is None

    # ready to move on
    assert gs.turn == PlayerID.TWO
    assert not gs.has_pending_decision()


def test_resolve_promotion():
    gs = create_gs_pending_promotion()

    cats = gs.player1.hand.num_cats()
    action = (Kitten(PlayerID.ONE), 5, 5)

    gs.apply_action(action)

    assert gs.must_promote
    assert not gs.player1.hand.has_kitten()
    assert gs.turn == PlayerID.ONE
    assert gs.has_pending_decision()

    gs.resolve_promotion(0, 0) # we have a kitten here

    # board updated
    assert gs.board[0][0] is None

    # hand updated
    assert not gs.player1.hand.has_kitten()
    assert gs.player1.hand.num_cats() == cats + 1  # 1 was promoted!
    
    # ready to move on
    assert not gs.has_pending_decision()
    assert gs.turn == PlayerID.TWO


def test_generate_successors_action():
    gs = simulate_early_game()

    orig_board = gs.board.copy()
    orig_p1 = gs.player1.copy()
    orig_p2 = gs.player2.copy()

    # this causes boops
    action = (Kitten(PlayerID.ONE), 2, 5)

    successor = gs.generate_successor_from_action(action)

    # check the state
    assert not successor.has_pending_decision()
    assert successor.winner is None
    assert not successor.must_promote
    assert successor.turn == PlayerID.TWO

    piece = successor.board[2][5]
    assert piece == Kitten(PlayerID.ONE)

    piece = successor.board[3][5]
    assert piece is None

    piece = successor.board[4][5]
    assert piece == Kitten(PlayerID.TWO)

    # verify we didn't touch the original board
    assert orig_board == gs.board

    # check hands
    assert orig_p1.hand.num_cats() == gs.player1.hand.num_cats()
    assert orig_p1.hand.num_kittens() == gs.player1.hand.num_kittens()
    assert orig_p2.hand.num_cats() == gs.player2.hand.num_cats()
    assert orig_p2.hand.num_kittens() == gs.player2.hand.num_kittens()


def test_generate_successors_with_triple():
    gs = simulate_early_game()
    
    # this will lead to a triple for Player 1
    action = (Kitten(PlayerID.ONE), 3, 3)

    p1_kittens = gs.player1.hand.num_kittens()
    p1_cats = gs.player1.hand.num_cats()
    orig_board = gs.board.copy()

    # generate the successor
    successor = gs.generate_successor_from_action(action)

    # check the state
    assert not successor.has_pending_decision()
    assert successor.winner is None
    assert not successor.must_promote
    assert successor.turn == PlayerID.TWO
    
    # this is the triple
    piece = successor.board[3][3]
    assert piece is None

    piece = successor.board[2][2]
    assert piece is None

    piece = successor.board[1][1]
    assert piece is None

    # this got booped to (5, 5)
    piece = successor.board[4][4]
    assert piece is None

    piece = successor.board[5][5]
    assert piece == Kitten(PlayerID.ONE)

    # verify hand has cats now!
    assert successor.player1.hand.num_kittens() == p1_kittens - 1  # one of these became a cat!
    assert successor.player1.hand.num_cats() == p1_cats + 3  # we promoted 3 cats!

    # verify the original is unchanged
    assert gs.board == orig_board
    assert p1_kittens == gs.player1.hand.num_kittens()
    assert p1_cats == gs.player1.hand.num_cats()


def test_generate_successors_selection():
    gs = create_gs_pending_both_decisions()
    orig_board = gs.board.copy()
    
    # this will lead to a DECISION for both players!
    action = (Kitten(PlayerID.ONE), 3, 3)

    # get the successor
    successor = gs.generate_successor_from_action(action)

    # verify new piece on board, but pending decicions, e.g. triples
    # still in place
    assert successor.board != orig_board

    assert gs.board[3][3] is None
    assert successor.board[3][3] == Kitten(PlayerID.ONE)
    assert successor.board[1][3] == Kitten(PlayerID.ONE)
    assert successor.board[5][3] == Kitten(PlayerID.TWO)

    assert successor.player1.pending_decision()
    assert successor.player2.pending_decision()
    assert not gs.player1.pending_decision()
    assert not gs.player2.pending_decision()
    
    # should be one still, need to check pending selections
    successor = successor.generate_successor_from_selection(PlayerID.ONE, successor.player1.decisions[0])

    # should still have player 2 pending
    assert successor.board != orig_board  # now changes
    assert not successor.player1.pending_decision()
    assert successor.player1.hand.num_cats() == gs.player1.hand.num_cats() + 3
    assert successor.player2.pending_decision()

    # resolve final decision and check
    successor = successor.generate_successor_from_selection(PlayerID.TWO, successor.player2.decisions[0])

    assert successor.board != orig_board  # now changes
    assert not successor.player1.pending_decision()
    assert successor.player1.hand.num_cats() == gs.player1.hand.num_cats() + 3
    assert not successor.player2.pending_decision()
    assert successor.player2.hand.num_cats() == gs.player2.hand.num_cats() + 3
    assert successor.turn == PlayerID.TWO
    
    # should still be unchanged
    assert gs.board == orig_board
    assert gs.turn == PlayerID.ONE


def test_generate_successors_promotion():
    gs = create_gs_pending_promotion()
    p1_cats = gs.player1.hand.num_cats()
    orig_board = gs.board.copy()

    action = (Kitten(PlayerID.ONE), 5, 5)
    successor = gs.generate_successor_from_action(action)

    assert successor.winner is None
    assert successor.must_promote  # now True!
    assert successor.turn == PlayerID.ONE  # did NOT change yet, pending!
    assert successor.has_pending_decision() # this time, yes!

    assert successor.board[5][5] == Kitten(PlayerID.ONE)
    assert gs.board[5][5] is None

    assert not successor.player1.hand.has_kitten()  # all on board
    assert successor.player1.hand.num_cats() == p1_cats  # no change yet!
    assert successor.has_pending_decision()
    assert successor.turn == PlayerID.ONE

    # generate successor based on picking a promotion
    successor = successor.generate_successor_from_promotion(0, 0)
    
    # verify
    assert not successor.player1.hand.has_kitten()  # all on board
    assert successor.player1.hand.num_cats() == p1_cats + 1 # promoted to a cat
    assert not successor.has_pending_decision() # resolve
    assert successor.turn == PlayerID.TWO
    assert successor.board[5][5] == Kitten(PlayerID.ONE)
    assert successor.board[0][0] is None


    # original should be unchanged
    assert gs.player1.hand.num_cats() == p1_cats
    assert gs.board == orig_board
    assert gs.turn == PlayerID.ONE
    assert gs.board[0][0] == Kitten(PlayerID.ONE)