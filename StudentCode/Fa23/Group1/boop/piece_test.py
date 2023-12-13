# Testing game piece logic.

from piece import *

def test_kitten_boops_kitten():
    k1 = Kitten(PlayerID.ONE)
    k2 = Kitten(PlayerID.ONE)
    assert k1.can_boop(k2)


def test_kitten_does_not_boop_cat():
    kitten = Kitten(PlayerID.ONE)
    cat = Cat(PlayerID.ONE)
    assert not kitten.can_boop(cat)


def test_cat_boops_kitten():
    kitten = Kitten(PlayerID.ONE)
    cat = Cat(PlayerID.ONE)
    assert cat.can_boop(kitten)


def test_cat_boops_cat():
    c1 = Cat(PlayerID.ONE)
    c2 = Cat(PlayerID.ONE)
    assert c1.can_boop(c2)


def test_kitten_graduates_into_cat():
    kitten = Kitten(PlayerID.ONE)
    piece = kitten.graduate()
    assert type(piece) is Cat


def test_cat_graduating_stays_cat():
    cat = Cat(PlayerID.ONE)
    piece = cat.graduate()
    assert type(piece) is Cat


def test_same_player_cats():
    cat1_1 = Cat(PlayerID.ONE)
    cat1_2 = Cat(PlayerID.ONE)
    assert cat1_1.same_player(cat1_2)
    cat2_1 = Cat(PlayerID.TWO)
    cat2_2= Cat(PlayerID.TWO)
    assert cat2_1.same_player(cat2_2)

    assert not cat1_1.same_player(cat2_1)


def test_same_player_kittens():
    kitten1_1 = Kitten(PlayerID.ONE)
    kitten1_2 = Kitten(PlayerID.ONE)
    assert kitten1_1.same_player(kitten1_2)
    kitten2_1 = Kitten(PlayerID.TWO)
    kitten2_2= Kitten(PlayerID.TWO)
    assert kitten2_1.same_player(kitten2_2)
    
    assert not kitten1_1.same_player(kitten2_1)


def test_same_player_cat_kitten():
    cat1 = Cat(PlayerID.ONE)
    kitten1 = Kitten(PlayerID.ONE)
    assert kitten1.same_player(cat1)
    cat2 = Cat(PlayerID.TWO)
    kitten2 = Kitten(PlayerID.TWO)
    assert cat2.same_player(kitten2)
    
    assert not cat1.same_player(kitten2)
    assert not kitten2.same_player(cat1)
