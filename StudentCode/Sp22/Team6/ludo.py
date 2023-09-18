import random


class Ludo(object):

    def __init__(self, players):

        self.players = players

        for player in self.players:
            #All players at home position initially
            player.state = [0.0, ] * 59
            player.state[0] = 1.00

        self.active_plyr_turn = random.randint(0, 3)

    @staticmethod
    def player_wins(player):

        if player.state[58] == 1:
            return True

        return False

    def roll_dice(self):

        return random.randint(1, 6)

    def play(self):

        active_plyr = 0

        while True:
            current_p = self.players[self.active_plyr_turn]

            die_val = self.roll_dice()

            current_p.move(die_val, self.players, active_plyr)

            if Ludo.player_wins(current_p):
                return current_p

            self.active_plyr_turn = (self.active_plyr_turn + 1) % 4

            active_plyr += 1
