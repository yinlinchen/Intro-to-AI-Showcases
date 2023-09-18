
import random

from player import *


class HumanPlayer(Player):

    def __init__(self, id, name=""):

        # Initialize a generic player
        Player.__init__(self, id, PlayerKind.Human)

        self.name = name

    def select_new_state(self, board_state, successors, ts):

        possible = []

        print("====================> HUMAN PLAYER " + self.name + ":")
        print("Choices:                 ")

        for s in successors:
            print (" " + str(s['action'][0]))
            possible.append(s['action'][0])

        print()

        while True:
            source_pos = input("Enter the source position: ")

            if source_pos in possible:
                break

        print()

        for i in range(len(successors)):
            if source_pos == successors[i]['action'][0]:
                return i

        return random.randint(0, len(successors) - 1)
