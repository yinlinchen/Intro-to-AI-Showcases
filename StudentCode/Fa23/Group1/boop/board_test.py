from board import Board, off_bed
from piece import Kitten, Cat, PlayerID
from util import list_match, all_have_match


def test_board_init():
    board = Board()
    for i in range(6):
        for j in range(6):
            assert board[i][j] is None


def test_get_item():
    board = Board()
    board[3][5] = Kitten(PlayerID.ONE)
    kitten = board[3][5]
    assert kitten is not None
    assert type(kitten) is Kitten


def test_out_of_range():
    board = Board()
    try:
        _ = board[1][7]
        assert False
    except IndexError:
        pass


def test_copy_board():
    board = Board()
    for i in range(6):
        board[i][i] = Kitten(PlayerID.ONE)
   
    board2 = board.copy()
    assert board == board2
    assert not board is board2
    for i in range(6):
        assert board2[i][i] is not None

    # modify first board
    for i in range(6):
        board[i][i] = None


    # test indeed distinct boards
    for i in range(6):
        assert board[i][i] is None
        assert board2[i][i] is not None


def test_off_bed():
    assert not off_bed(0, 0)
    assert not off_bed(5, 5)
    assert not off_bed(3, 3)
    assert not off_bed(1, 4)
    assert not off_bed(4, 1)
    assert off_bed(-1, 3)
    assert off_bed(3, -1)
    assert off_bed(2, 6)
    assert off_bed(6, 2)
    assert off_bed(6, 6)
    assert off_bed(7, 7)
    assert off_bed(-1, -1)
    assert off_bed(-1, 6)
    assert off_bed(6, -1)


def test_boop_vector():
    assert Board.boop_vector(0, 0, 1, 1) == (1, 1)
    assert Board.boop_vector(1, 1, 0, 0) == (-1, -1)
    assert Board.boop_vector(0, 1, 1, 1) == (1, 0)
    assert Board.boop_vector(2, 1, 1, 1) == (-1, 0)
    assert Board.boop_vector(1, 0, 1, 1) == (0, 1)
    assert Board.boop_vector(1, 2, 1, 1) == (0, -1)
    assert Board.boop_vector(0, 2, 1, 1) == (1, -1)
    assert Board.boop_vector(2, 2, 1, 1) == (-1, -1)
    assert Board.boop_vector(2, 0, 1, 1) == (-1, 1)

    assert Board.boop_vector(2, 3, 3, 4) == (1, 1)
    assert Board.boop_vector(2, 4, 3, 4) == (1, 0)
    assert Board.boop_vector(4, 4, 3, 4) == (-1, 0)
    assert Board.boop_vector(3, 3, 3, 4) == (0, 1)
    assert Board.boop_vector(3, 5, 3, 4) == (0, -1)
    assert Board.boop_vector(2, 5, 3, 4) == (1, -1)
    assert Board.boop_vector(4, 5, 3, 4) == (-1, -1)
    assert Board.boop_vector(4, 3, 3, 4) == (-1, 1)


def test_completion_points():
    # given a pt, should return a list of pairs of pts
    # that complete triples, with nothing invalid
    
    # boundaries
    comps = Board.get_completion_points(0, 0)
    expected = [
        [(1, 0), (2, 0)], # row
        [(0, 1), (0, 2)], # col
        [(1, 1), (2, 2)], # diag
    ]
    assert all_have_match(expected, comps)


    comps = Board.get_completion_points(5, 5)
    expected = [
        [(3, 5), (4, 5)], # row
        [(5, 3), (5, 4)], # col
        [(3, 3), (4, 4)], # diag
    ]
    assert all_have_match(expected, comps)


    # try one in the middle
    comps = Board.get_completion_points(3, 3)
    expected =[
        [(2, 3), (4, 3)], # row with pt in mid
        [(1, 3), (2, 3)], # row with pt at left
        [(4, 3), (5, 3)], # row with pt at right

        [(3, 2), (3, 4)], # col with pt in mid
        [(3, 1), (3, 2)], # col with pt at top
        [(3, 4), (3, 5)], # col with pt at bottom

        [(2, 4), (4, 2)], # diag pt mid -> \
        [(2, 4), (1, 5)], # diag lower right -> \
        [(4, 2), (5, 1)], # diag upper left -> \

        [(2, 2), (4, 4)], # diag pt mid -> /
        [(4, 4), (5, 5)], # diag lower left -> /
        [(2, 2), (1, 1)], # diag upper right -> /
    ]
    assert all_have_match(expected, comps)


def test_overlaps():
    t1 = [(0, 0), (0, 1), (0, 2)]
    t2 = [(1, 0), (1, 1), (1, 2)]
    t3 = [(0, 2), (1, 1), (2, 0)]
    t4 = [(0, 0), (1, 1), (2, 2)]
    t5 = [(3, 3), (4, 3), (5, 3)]
    t6 = [(5, 2), (5, 3), (5, 4)]
    
    assert not Board.overlaps(t1, t2)
    assert Board.overlaps(t1, t3)
    assert Board.overlaps(t2, t3)
    assert Board.overlaps(t1, t4)
    assert Board.overlaps(t2, t4)
    assert Board.overlaps(t3, t4)
    assert not Board.overlaps(t5, t1)
    assert not Board.overlaps(t5, t2)
    assert not Board.overlaps(t5, t3)
    assert not Board.overlaps(t5, t4)
    assert Board.overlaps(t5, t5)
    assert Board.overlaps(t5, t6)


def test_get_overlapping():
    # test if triples have overlap, for when player
    # must resolve multiple triples in one move
    assert Board.get_overlapping([]) == []

    case = [
        [(0, 0), (0, 1), (0, 2)], # no overlap
        [(1, 0), (1, 1), (1, 2)]
    ]

    assert Board.get_overlapping(case) == []

    case = [
        [(0, 0), (0, 1), (0, 2)], # one overlap
        [(0, 0), (1, 1), (2, 2)]
    ]

    res = Board.get_overlapping(case)
    assert all_have_match(case, res)

    case = [
        [(0, 0), (0, 1), (0, 2)], # all overlap (at different pts)
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)]
    ]
    res = Board.get_overlapping(case)
    assert all_have_match(case, res)

    case = [
        [(0, 0), (0, 1), (0, 2)], # outer overlap the middle one (but not each other)
        [(0, 0), (1, 1), (2, 2)],
        [(1, 2), (2, 2), (3, 2)]
    ]
    res = Board.get_overlapping(case)
    assert all_have_match(case, res)


    case = [
        [(0, 0), (0, 1), (0, 2)], # two overlap, one does not
        [(3, 3), (4, 3), (5, 3)],
        [(0, 0), (1, 1), (2, 2)]
    ]

    res = Board.get_overlapping(case)
    expected = [case[0], case[2]]
    assert all_have_match(expected, res)

    case = [
        [(0, 0), (0, 1), (0, 2)], # two distinct cases of overlaps
        [(0, 0), (1, 1), (2, 2)],
        [(3, 3), (4, 3), (5, 3)],
        [(5, 2), (5, 3), (5, 4)]
    ]

    res = Board.get_overlapping(case)
    assert all_have_match(case, res)


def test_empty_squares():
    board = Board()
    empties = board.empty_squares()
    assert len(empties) == 36
    assert (0,0) in empties
    assert (3,4) in empties
    assert (5,5) in empties

    board[3][4] = Kitten(PlayerID.ONE)
    empties = board.empty_squares()
    assert len(empties) == 35
    assert (3,4) not in empties
    assert (0,0) in empties
    assert (5,5) in empties
    assert (2,3) in empties
    

    board[4][5] = Cat(PlayerID.TWO)
    empties = board.empty_squares()
    assert len(empties) == 34
    assert (3,4) not in empties
    assert (4,5) not in empties
    assert (0,0) in empties
    assert (5,5) in empties
    assert (2,3) in empties

    board[3][4] = None
    board[4][5] = None
    empties = board.empty_squares()
    assert len(empties) == 36
    assert (3,4) in empties
    assert (4,5) in empties


def test_surrounding_squares():
    # test boundary cases
    sqs = Board.surrounding_squares(0, 0)
    expected = [(0, 1), (1, 0), (1, 1)]
    assert list_match(expected, sqs)

    sqs = Board.surrounding_squares(5, 5)
    expected = [(4, 5), (5, 4), (4, 4)]
    assert list_match(expected, sqs)


    # test a middle case
    sqs = Board.surrounding_squares(2, 3)
    expected = [(1, 3), (3, 3), (1, 2), (2, 2), (3, 2), (3, 4), (2, 4), (1, 4)]
    assert list_match(expected, sqs)


def test_num_cats_in_triple():
    # test boundary cases
    board = Board()
    t1 = [(0, 0), (0, 1), (0, 2)]
    t2 = [(1, 0), (1, 1), (1, 2)]
    t3 = [(0, 2), (1, 1), (2, 0)]
    t4 = [(0, 0), (1, 1), (2, 2)]
    t5 = [(3, 3), (4, 3), (5, 3)]
    t6 = [(5, 2), (5, 3), (5, 4)]

    for t in [t1, t2, t3, t4, t5, t6]:
        assert board.num_cats(t) == 0

    # add some kittens
    board[1][1] = Kitten(PlayerID.ONE)
    board[3][3] = Kitten(PlayerID.ONE)

    
    for t in [t1, t2, t3, t4, t5, t6]:
        assert board.num_cats(t) == 0

    # now add some cats
    board[1][1] = Cat(PlayerID.ONE)
    for t in [t2, t3, t4]:
        assert board.num_cats(t) == 1
    
    for t in [t1, t5, t6]:
        assert board.num_cats(t) == 0

    board[0][0] = Cat(PlayerID.ONE)
    for t in [t1, t2, t3]:
        assert board.num_cats(t) == 1

    for t in [t5, t6]:
        assert board.num_cats(t) == 0

    assert board.num_cats(t4) == 2

    # last one: detect the triple!
    board[2][2] = Cat(PlayerID.ONE)
    assert board.num_cats(t4) == 3
    

def test_board_equality():
    b1 = Board()
    b2 = Board()
    assert b1 == b2

    b1[2][1] = Cat(1)
    assert b1 != b2
    
    b2[2][1] = Kitten(1)
    assert b1 != b2

    b2[2][1] = Cat(2)
    assert b1 != b2

    b2[2][1] = Cat(1)
    assert b1 == b2

    b2[4][5] = Kitten(1)
    assert b1 != b2
