# This Class will represent the game piece in the Boop Game
from abc import ABC, abstractmethod
from enum import Enum

class PlayerID(Enum):
    ONE = 1,
    TWO = 2


class Piece(ABC):
    """
    Represents a game piece.
    Stores which player it belongs to.
    Reference equivalence should not be relied upon.
    """
    def __init__(self, id: PlayerID):
        self.player = id


    def __eq__(self, other: object) -> bool:
        """
        Overrides equality for convenience, as these are
        generally treated like values.
        """
        if self is None:
            return other is None
        
        return type(self) is type(other) and self.same_player(other)


    @abstractmethod
    def can_boop(self, other) -> bool:
        pass


    @abstractmethod
    def graduate(self):
        pass
    
    @abstractmethod
    def getImagePath(self):
        pass


    def get_player(self) -> PlayerID:
        return self.player


    def same_player(self, other):
        return other.get_player() == self.get_player()


    def same_type(self, other):
        return type(self) is type(other)


class Kitten(Piece):
    def __init__(self, id: PlayerID):
        super().__init__(id)

    def __repr__(self):
        return f"K({1 if self.player == PlayerID.ONE else 2})"

    def can_boop(self, other) -> bool:
        return type(other) is Kitten
    
    def graduate(self):
        return Cat(self.get_player())
    
    def getImagePath(self):        
        if self.player == PlayerID.ONE:
            name = 'kitten1.jpg'
        else:
            name = 'kitten2.png'
        
        return f'boop/game_assets/{name}'
        

class Cat(Kitten):
    def __init__(self, id: PlayerID):
        super().__init__(id)

    def __repr__(self):
        return f"C({1 if self.player == PlayerID.ONE else 2})"

    def can_boop(self, other) -> bool:
        # cats boop all
        return True

    def graduate(self):
        return self
    
    def getImagePath(self):
        if self.player == PlayerID.ONE:
            name = 'cat1.jpg'
        else:
            name = 'cat2.png'
        
        return f'boop/game_assets/{name}'