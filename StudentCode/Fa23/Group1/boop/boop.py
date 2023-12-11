# This Class is the overall script that will run and render the game
import sys
from optparse import OptionParser
from player import Player
from piece import PlayerID
from game import Game
from minimax_agent import MinimaxAgent, AlphaBetaAgent, BeamAgent
import evaluation
import game_gui


def get_eval(eval_name):
    """
    Helper to load an evaluation function based on arg name.
    """
    return getattr(evaluation, eval_name)


def create_agent(evaluator, agent_name, depth, width, options):
    """
    Helper to load an agent class based on arg name.
    """
    if agent_name == 'MinimaxAgent':
        return MinimaxAgent(evalFunc=evaluator, depth=depth, max_states=options.maxStates)
    elif agent_name == 'AlphaBetaAgent':
        return AlphaBetaAgent(evalFunc=evaluator, depth=depth, max_states=options.maxStates)
    elif agent_name == 'BeamAgent':
        return BeamAgent(evalFunc=evaluator, depth=depth, max_states=options.maxStates, width=width)
    else:
        raise Exception("invalid agent type: " + str(agent_name))


def run_game(args):
    """
    Runs the game, given the arguments.
    Returns winner and statistics for AI matches.
    """
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option("-a", "--ai", dest="ai", default="2", type="choice", choices=["1", "2", "none", "both"],
                      help="AI player, 1, 2, 'none', or 'both'")
    parser.add_option("-d", "--depth", dest="depth", default=1, type='int', help="Minimax depth")
    parser.add_option("", "--depth2", dest="depth2", type='int',
                       help="Depth of second agent, if both players are AI")
    parser.add_option("-m", "--maxStates", dest="maxStates", default=12_000, type='int',
                      help="Max number of states to evaluate. This is a 'soft' max "
                      "as we don't bail from iterating the actions within a state, and thus may exceed it slightly.")
    parser.add_option("", "--agent", dest="agent", default="AlphaBetaAgent", type="choice",
                      help="Agent type",
                      choices=["MinimaxAgent", "AlphaBetaAgent", "BeamAgent"])
    parser.add_option("", "--agent2", dest="agent2", type="choice",
                      help="Agent type for AI 2, if both players are AI",
                      choices=["MinimaxAgent", "AlphaBetaAgent", "BeamAgent"])
    parser.add_option("", "--eval", dest="eval", default="eval_territory", type="choice",
                      help="Evaluation function",
                      choices=['eval_piece_count', 'eval_board_bonus', 'eval_territory', 'eval_stranding'])
    parser.add_option("", "--eval2", dest="eval2", type="choice",
                       help="Evaluation function for second agent, if both",
                       choices=['eval_piece_count', 'eval_board_bonus', 'eval_territory', 'eval_stranding'])
    parser.add_option('-w', "--width", dest="width", default=5, type='int',
                      help="Beam width, when using BeamAgent")
    parser.add_option('', "--width2", dest="width2", type='int',
                      help="Beam width, when using BeamAgent")

    options, extra = parser.parse_args(args)
    if len(extra) != 0:
        print("Warning: ignoring unknown extra command args: " + str(extra))

    print("Running with options: " + str(options))

    # create game instance
    player1 = Player(PlayerID.ONE)
    player2 = Player(PlayerID.TWO)
    game = Game(player1, player2)

    # configure AI based on opts
    if options.depth <= 0:
        raise Exception("invalid depth, must be positive integer")
    
    if options.ai == "1" or options.ai == "both":
        evaluator = get_eval(options.eval)
        agent = create_agent(evaluator, options.agent, options.depth, options.width, options)
        player1.set_agent(agent)

    # for player 2, default to the normal settings for easy ai-vs-human CLI,
    # but allow overrides in case of agent-vs-agent mode, to facilitate experimentation
    if options.ai == "2" or options.ai == "both":
        eval_name = options.eval
        agent_name = options.agent
        depth = options.depth
        width = options.width
        if options.ai == "both":
            if options.eval2:
                eval_name = options.eval2
            
            if options.agent2:
                agent_name = options.agent2
        
            if options.depth2:
                depth = options.depth2

            if options.width2:
                depth = options.width2
            
        evaluator = get_eval(eval_name)
        agent = create_agent(evaluator, agent_name, depth, width, options)
        player2.set_agent(agent)

    # run the game
    if options.ai == "both":
        # run the AI-only game, no GUI
        game.play_ai_match()
        return game.state.winner, game.state.plies
    else:
        # run the GUI
        game_gui.run_gui(game)      


if __name__ == '__main__':       
    # parse CLI args
    run_game(sys.argv[1:])