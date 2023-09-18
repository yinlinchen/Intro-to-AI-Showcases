import random
import math

from player import *


class RandomPlayer(Player):

    def __init__(self, id):

        # Init a base strategy player
        Player.__init__(self, id, PlayerKind.Random)

    def select_new_state(self, board_state, successors, timestamp):
        return random.randint(0, len(successors) - 1)
