from uttt_main import UTTT
import player
import sys
from window import Window
import argparse
from player import UserPlayer

def game_loop(player1, player2, printing=False, display_window=True, display_delay=100):
    game = UTTT()
    window = Window(game) if display_window else None
    while game.winner == 0:
        if game.player_turn == 1:
            player1.make_move(game, window)
        else:
            player2.make_move(game, window)
        if window:
            window.update(game.game_grid, game.simplified_grid, game.last_move)
            # Flag to slow down game
            window.delay(display_delay)
        if printing:
            game.print_board()
            print("\nValid Moves: ", game.valid_moves)
    
    if game.winner == -1:
        print("\nIt's a Tie!")
    else:
        print("\nPlayer ", game.winner, " Won!")
    return game.winner
    
def create_players(player1, player2):
    ret = []
    for p in [player1, player2]:
        if p == 'random':
            ret.append(player.RandomPlayer())
        elif p == 'player':
            ret.append(player.UserPlayer())
        elif p == 'minimax':
            ret.append(player.MiniMaxPlayer(5, len(ret) + 1))
        elif len(p) >= 11 and p[0:11] == 'monte-carlo':
            ret.append(player.MonteCarloPlayer(int(p[11::]) if len(p[11::]) > 0 else 10))
    return ret

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    # run this file with 2 arguments to select the agent type of the first and second player
    # options are:
    #   player -> human player, will input commands to the command line on their turn
    #   random -> random AI agent. Will randomly select a legal move on its turn
    #   monte-carlo## -> monte carlo decision tree AI agent. Will run random simulations to determine its next move on its turn
    #       add number to the end to determine the number of sims to run. e.g., monte-carlo10 for 10 simulations per move
    #

    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", dest="player1", default="random")
    parser.add_argument("-p2", dest="player2", default="random")
    parser.add_argument("-window", dest="display_window", default="True", help="whether or not the game will run with a gui")
    parser.add_argument("-games", dest="game_count", default="1")
    parser.add_argument("-delay", dest="delay", default="100")

    args = parser.parse_args()

    player1, player2 = create_players(args.player1, args.player2)

    # Handle AI Multiple Games
    if not isinstance(player1, UserPlayer) and not isinstance(player2, UserPlayer):
        winner_dict = {-1: 0, 1: 0, 2: 0}
        for i in range(0, int(args.game_count)):
            winner_dict[game_loop(player1, player2, False, display_window=str2bool(args.display_window), display_delay=100)] += 1
        print(winner_dict)   
    # Human Game
    else:
        winner = game_loop(player1, player2, False, display_window=str2bool(args.display_window), display_delay=str2bool(args.display_window))
        print(f"Winner: Player {winner}")
