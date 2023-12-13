from player import Player
from piece import PlayerID, Cat, Kitten

def test_init():
    player = Player(PlayerID.ONE)
    assert player.hand is not None
    assert player.hand.num_kittens() == 8
    assert player.hand.num_cats() == 0
    assert player.id == PlayerID.ONE


def test_id():
    player = Player(PlayerID.ONE)
    assert player.id == PlayerID.ONE

    player = Player(PlayerID.TWO)
    assert player.id == PlayerID.TWO


def test_copy():
    player = Player(PlayerID.TWO)
    player.decisions = [[(0, 0), (1, 1), (2, 2)],
                        [(1, 1), (2, 2), (3, 3)]]
    player.hand.get_kitten()
    player.hand.add_piece(Cat(player.id))
    
    copy = player.copy()
    assert copy.id == PlayerID.TWO
    assert len(copy.decisions) == 2
    assert copy.pending_decision()

    # make sure hand is distinct ref (though decisions are shallow copy)
    assert copy.hand is not None
    assert copy.hand.num_kittens() == 7
    assert copy.hand.num_cats() == 1
    copy.hand.get_kitten()
    copy.hand.add_piece(Cat(copy.id))

    assert copy.hand.num_kittens() == 6
    assert copy.hand.num_cats() == 2

    assert player.hand.num_kittens() == 7
    assert player.hand.num_cats() == 1


def test_decisions():
    player = Player(PlayerID.ONE)
    assert len(player.decisions) == 0
    assert not player.pending_decision()

    player.decisions = [[(0, 0), (1, 1), (2, 2)],
                        [(1, 1), (2, 2), (3, 3)]]
    
    assert player.pending_decision()
