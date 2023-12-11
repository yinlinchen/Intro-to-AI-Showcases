# This Class will represent the Player in the Boop Game
from hand import Hand
from piece import PlayerID
from board import Triple

class Player():
    def __init__(self, id: PlayerID, hand=None):
        """
        Initializes player with id and hand.
        If hand not specified, initializes with default hand.
        """
        self.id = id
        self.hand = hand
        if self.hand is None:
            self.hand = Hand(id)
        self.decisions: list[Triple] = []
        self.agent = None
    
    def __repr__(self):
        return f"{1 if self.player == PlayerID.ONE else 2})"
    

    def set_agent(self, agent):
        """
        Sets an AI agent for this Player.
        """
        self.agent = agent


    def is_agent(self):
        """
        Returns True if Player is an AI agent, False otherwise
        """
        return self.agent is not None


    def copy(self):
        """
        Makes a copy of this player.
        Note that the decisions on the player's hand are a _shallow_ copy,
        as it is not expected we should modify the collection itself.
        """
        hand = self.hand.copy()
        player =  Player(self.id, hand)

        # NOTE: shallow copy, but this should be ok (and indeed, more performant)
        # as we don't intend to modify the decisions collection itself, but assign
        # a new list.
        # If necessary, we can change to a deep copy
        player.decisions = [d for d in self.decisions]
        player.agent = self.agent
        return player


    def pending_decision(self):
        """
        Returns whether player has a pending decision.
        """
        return len(self.decisions) > 0
    
    def getID(self):
        """
        Returns player ID
        """
        return self.id