from hand import Hand
from piece import Piece, Kitten, Cat, PlayerID

def test_hand_init():
    hand = Hand(PlayerID.ONE)
    assert len(hand.kittens) == 8
    for i in range(8):
        assert type(hand.kittens[i]) is Kitten
        assert hand.kittens[i].get_player() == PlayerID.ONE

    assert hand.cats == []


def test_has_kitten():
    hand = Hand(PlayerID.ONE)
    assert hand.has_kitten()

    hand.kittens = []
    assert not hand.has_kitten()

    hand.kittens = [Kitten(PlayerID.ONE)]
    assert hand.has_kitten()


def test_has_cat():
    hand = Hand(PlayerID.ONE)
    assert not hand.has_cat()

    hand.cats = [Cat(PlayerID.ONE), Cat(PlayerID.ONE)]
    assert hand.has_cat()


def test_num_kittens():
    hand = Hand(PlayerID.ONE)
    assert hand.num_kittens() == 8
    hand.kittens = [Kitten(PlayerID.ONE), Kitten(PlayerID.ONE), Kitten(PlayerID.ONE)]
    assert hand.num_kittens() == 3


def test_num_cats():
    hand = Hand(PlayerID.ONE)
    assert hand.num_cats() == 0
    hand.cats = [Cat(PlayerID.ONE), Cat(PlayerID.ONE), Cat(PlayerID.ONE)]
    assert hand.num_cats() == 3


def test_get_kitten():
    hand = Hand(PlayerID.TWO)
    for i in range(8):
        kitten = hand.get_kitten()
        assert kitten is not None
        assert kitten.get_player() == PlayerID.TWO
    kitten = hand.get_kitten()
    assert kitten is None


def test_get_cat():
    hand = Hand(PlayerID.ONE)
    assert hand.get_cat() is None
    hand.cats = [Cat(PlayerID.ONE), Cat(PlayerID.ONE), Cat(PlayerID.ONE)]
    for i in range(3):
        cat = hand.get_cat()
        assert cat is not None
        assert cat.get_player() == PlayerID.ONE


def test_add_piece():
    hand = Hand(PlayerID.ONE)
    err = False
    try:
        hand.add_piece(Kitten(PlayerID.ONE))
    except:
        err = True

    assert err

    for i in range(3):
        kitten = hand.get_kitten()
        assert type(kitten) is Kitten
        assert kitten.get_player() == PlayerID.ONE

    assert hand.num_kittens() == 5
    
    for i in range(3):
        hand.add_piece(Kitten(PlayerID.ONE))

    assert hand.num_kittens() == 8

    assert not hand.has_cat()
    for i in range(8):
        hand.add_piece(Cat(PlayerID.ONE))

    err = False
    try:
        hand.add_piece(Cat(PlayerID.ONE))
    except:
        err = True

    assert err

    for i in range(8):
        cat = hand.get_cat()
        assert type(cat) is Cat
        assert cat.get_player() == PlayerID.ONE

    assert not hand.has_cat()
    
    # try to add for wrong player
    err = False
    try:
        hand.add_piece(Cat(PlayerID.TWO))
    except:
        err = True

    assert err


def test_copy():
    hand = Hand(PlayerID.ONE)
    for i in range(3):
        kitten = hand.get_kitten()

    assert hand.num_kittens() == 5
    
    for i in range(4):
        hand.add_piece(Cat(PlayerID.ONE))

    assert hand.num_kittens() == 5
    assert hand.num_cats() == 4

    # test modifying doesn't affect original
    copy = hand.copy()
    assert copy.num_kittens() == 5
    assert copy.num_cats() == 4

    cat = copy.get_cat()
    assert type(cat) is Cat
    assert cat.get_player() == PlayerID.ONE

    kitten = copy.get_kitten()
    assert type(kitten) is Kitten
    assert kitten.get_player() == PlayerID.ONE

    assert hand.num_kittens() == 5
    assert hand.num_cats() == 4

    assert copy.num_kittens() == 4
    assert copy.num_cats() == 3