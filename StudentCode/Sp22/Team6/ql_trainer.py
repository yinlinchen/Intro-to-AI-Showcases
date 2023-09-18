import random
import math

from ql_player import QLPlayer
from rnd_player import RandomPlayer
from ludo import Ludo
from mixed_strategy_player import MixedStrategyPlayer
from nn import NN


class QLTrainer(Ludo):

    def __init__(self, num_episodes, nn_file_dst, nn_file_src=None, debug=False):
        Ludo.__init__(self, [])
        self.nn = NN(238, nn_file_src)
        self.num_episodes = num_episodes
        self.nn_file_dst = nn_file_dst
        self.debug = debug

    def train(self, test=False):
        wins = [0, ] * 4
        max_epsilon = 0.9
        epsilon = max_epsilon

        wrf = open("out.txt", "a")
        wrf.write ("Test Output\n")
        wrf.close()

        if self.debug:
            print ("====================================================================================================")
            print( "| TRAINING STARTED                                                                                 |")
            print ("====================================================================================================")
            print()
        if test:
            wrf = open("out.txt", "a")
            wrf.write ("%-10d %20.2f %20.2f" % (0,
                                           self.test(1000, "1_QL_AGAINST_3_RANDOM")[0],
                                           self.test(1000, "1_QL_AGAINST_3_EXPERT")[0]))
            wrf.close()

        for episode in range(self.num_episodes):
            if self.debug:
                print ("Training episode " + str(episode + 1) + "/" + str(self.num_episodes) + "...")

            # Players to train with
            self.players = [QLPlayer(id=0, train=True, nn=self.nn, epsilon=epsilon),
                            QLPlayer(id=1, train=True, nn=self.nn, epsilon=epsilon),
                            QLPlayer(id=2, train=True, nn=self.nn, epsilon=epsilon),
                            QLPlayer(id=3, train=True, nn=self.nn, epsilon=epsilon)]
            self.player_turn = 0
            for player in self.players:
                player.state = [0.0, ] * 59
                player.state[0] = 1.00

            self.play()

            for p in range(len(self.players)):
                if self.players[p].state[58] == 1.0:
                    wins[p] += 1
            if episode < 0.1 * self.num_episodes:
                epsilon = ((0.1 * self.num_episodes - 1) - episode) * max_epsilon / (0.1 * self.num_episodes - 1)
            else:
                epsilon = 0

            if self.debug:
                print()
            if test and episode > 0:
                if episode % 1000 == 0:
                    wrf = open("out.txt", "a")
                    wrf.write ("%-10d %20.2f %20.2f" % (episode,
                                                   self.test(1000, "1_QL_AGAINST_3_RANDOM")[0],
                                                   self.test(1000, "1_QL_AGAINST_3_EXPERT")[0]))
                    wrf.close()
            if episode % 1000 == 0:
                self.nn.write_to_file(self.nn_file_dst)

        wrf.close()
        self.nn.write_to_file(self.nn_file_dst)

        for w in range(len(wins)):
            wins[w] = wins[w] * 100.0 / self.num_episodes

        if self.debug:
            print ("Wins: " + str(wins))
            print()

            print ("====================================================================================================")
            print ("| TRAINING ENDED                                                                                   |")
            print ("====================================================================================================")
            print()

    def test(self, games, setting):
        wins = [0, ] * 4

        if self.debug:
            print ("====================================================================================================")
            print ("| TESTING STARTED                                                                                  |")
            print ("====================================================================================================")
            print()

        for episode in range(games):
            if self.debug:
                print ("Testing game " + str(episode + 1) + "/" + str(games) + "...")
            if setting == "1_QL_AGAINST_3_RANDOM":
                self.players = [QLPlayer(id=0, train=False, nn=self.nn, epsilon=0),
                                RandomPlayer(id=1),
                                RandomPlayer(id=2),
                                RandomPlayer(id=3)]
            elif setting == "1_QL_AGAINST_3_EXPERT":
                self.players = [QLPlayer(id=0, train=False, nn=self.nn, epsilon=0),
                                MixedStrategyPlayer(id=1),
                                MixedStrategyPlayer(id=2),
                                MixedStrategyPlayer(id=3)]
            self.player_turn = 0
            for player in self.players:
                player.state = [0.0, ] * 59
                player.state[0] = 1.00
            self.play()

            for p in range(len(self.players)):
                if self.players[p].state[58] == 1:
                    wins[p] += 1

            if self.debug:
                print()

        for w in range(len(wins)):
            wins[w] = wins[w] * 100.0 / games

        if self.debug:
            print ("Wins: " + str(wins))
            print()

        if self.debug:
            print ("====================================================================================================")
            print ("| TESTING ENDED                                                                                    |")
            print ("====================================================================================================")
            print()

        return wins

trainer = QLTrainer(2000 + 1, '01102016_nn_exp_2.txt', debug=False)

f = open("out.txt", "a")
#print ("test sys.stdout")

f.write ("Parameters:")
f.write ("================================================================================")
f.write ("QL Learning Rate: " + str(QLPlayer.learning_rate))
f.write ("QL Discount Rate: " + str(QLPlayer.discount_rate))
f.write ("NN Learning Rate: " + str(NN.learning_rate))
f.write ("NN Momentum:      " + str(NN.momentum))
f.write ("Training:         " + "4 QL Players")
f.write ("%-10s %20s %20s" % ("Episodes", "RND Win Percentage", "XPT Win Percentage"))
f.write ("================================================================================")
f.close()

trainer.train(test=True) 

