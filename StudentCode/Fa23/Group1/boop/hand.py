# This class will represnt the set of cards in the players hand.

from piece import *

class Hand():
    def __init__(self, player: PlayerID):
        """
        Constructs hand with 8 kittens to start, for appropriate player.
        """
        self.kittens: list[Kitten] = [Kitten(player)]*8 # start with 8 kittens
        self.cats: list[Cat] = []
        self.player = player        

        # TODO: could track promotions in additional manner, so we can tell when
        # all 8 cats are on the board more easily...

    def copy(self):
        """
        Creates a copy of this hand.
        """
        hand = Hand(self.player)
        hand.kittens = [Kitten(self.player)]*self.num_kittens()
        hand.cats = [Cat(self.player)]*self.num_cats()
        return hand


    def num_kittens(self) -> int:
        """
        Returns number of kittens currently in hand.
        """
        return len(self.kittens)
    

    def num_cats(self) -> int:
        """
        Returns number of cats currently in hand.
        """
        return len(self.cats)
    

    def has_kitten(self) -> bool:
        """
        Returns whether hand has any kittens.
        """
        return len(self.kittens) > 0
    

    def has_cat(self) -> bool:
        """
        Returns whether hand has any cats.
        """
        return len(self.cats) > 0
    

    def get_kitten(self) -> Kitten|None:
        """
        Returns a kitten from hand, or None.
        """
        if self.num_kittens() == 0:
            return None
        
        return self.kittens.pop()
    

    def get_cat(self) -> Cat|None:
        """
        Returns a cat from hand, or None.
        """
        if self.num_cats() == 0:
            return None
        
        return self.cats.pop()
    
    
    def add_piece(self, piece: Piece):
        """
        Adds a game piece back to the hand.
        """
        if piece.get_player() != self.player:
            raise Exception("attempted to add wrong player's piece to hand")

        if type(piece) is Kitten:
            if len(self.kittens) == 8:
                raise Exception("too many kittens")

            self.kittens.append(piece)
        else:
            if len(self.cats) == 8:
                raise Exception("too many cats")

            self.cats.append(piece)

    def remove_piece(self, piece: Piece):
        """
        Remove a game piece back to the hand.
        """
        if piece.get_player() != self.player:
            raise Exception("attempted to remove wrong player's piece to hand")
        
        if type(piece) is Kitten:
            if len(self.kittens) == 0:
                raise Exception("too few kittens")
            
            self.kittens.remove(piece)
        else:
            if len(self.cats) == 0:
                raise Exception("too few cats")

            self.cats.remove(piece)
                    
