# This class runs a series of experiements to compare AI agents.
from boop import run_game
from piece import PlayerID
import time

def compare_evals(e1, e2, n, maxStates):
    """
    Compare evaluation funcs, using AlphaBeta and other default opts.
    """
    args = ["--ai", "both", 
            "--agent", "AlphaBetaAgent", 
            "--eval", e1, "--eval2", e2,
            "--maxStates", str(maxStates)
    ]
    run_exp(args, n)


def compare_beam_params(w1, w2, d1, d2, n, maxStates):
    """
    Compare beam search parameters.
    """
    args = ["--ai", "both", 
            "--agent", "BeamAgent", 
            "--eval", "eval_territory",
            "--maxStates", str(maxStates),
            "--width", str(w1),
            "--width2", str(w2),
            "--depth", str(d1),
            "--depth2", str(d2)
    ]
    run_exp(args, n)


def compare_alpha_vs_beam(w, d_beam, d_alpha, n, maxStates):
    """
    Compare the best AlphaBeta vs Beam Search.
    """
    args = ["--ai", "both", 
            "--agent", "BeamAgent", 
            "--agent2", "AlphaBetaAgent",
            "--eval", "eval_territory",
            "--maxStates", str(maxStates),
            "--width", str(w),
            "--depth", str(d_beam),
            "--depth2", str(d_alpha)
    ]
    run_exp(args, n)


def compare_mm_depths(d1, d2, n, maxStates):
    """
    Compares minimax depths for same eval func and
    same maxStates cap.
    """
    args = ["--ai", "both", 
            "--agent", "AlphaBetaAgent", 
            "--eval", 'eval_territory',
            "--maxStates", str(maxStates),
            "--depth", str(d1),
            "--depth2", str(d2),
    ]
    run_exp(args, n)
   

def run_exp(args, n):
    print("Starting experiment")
    start = time.time()
    p2Wins = 0
    plies = 0
    for i in range(n):
        winner, plies = run_game(args)
        if winner == PlayerID.TWO:
            p2Wins += 1
        print(f"====> So far: p2 win percentage = {p2Wins / (i+1)}")
    end = time.time()
    p2_percentage = p2Wins / n
    avg_plies = plies / n
    duration = end - start
    print(f"====> Final p2 win percentage: {p2_percentage}, avg plies: {avg_plies}, time: {duration} sec")


if __name__ == '__main__':
    # NOTE: at low levels of maxStates, eval_piece_count and eval_board_bonus deadlock.
    # At high levels, the games will be SLOW!

    # NOTE: originally, this was setup to do multiple runs of each, but as
    # there is no randomization and the algorithms are deterministic,
    # there's no point for the head-to-heads, only for the meta statistics.
    # In other words, you could run multiple, but they'll play the same game
    # over and over, with the same results.

    # These experiments are head-to-head contests of evaluation functions
    compare_evals("eval_piece_count", "eval_board_bonus", 1, 500_000)
    #compare_evals("eval_board_bonus", "eval_territory", 1, 500_000)
    #compare_evals("eval_board_bonus", "eval_stranding", 1, 500_000)

    # See if who goes first has an advantage?
    #compare_evals("eval_territory", "eval_board_bonus", 1, 500_000)
    #compare_evals("eval_board_bonus", "eval_board_bonus", 1, 500_000)

    # This experiment compares a wider, shallower beam to a narrower but deeper one
    #compare_beam_params(10, 5, 10, 15, 1, 500_000)
    
    # This compares minimax to beam search
    #compare_alpha_vs_beam(10, 10, 2, 1, 200_000)

    # This runs more alpha-beta minimax games at lower numbers of max states,
    # to observe the effects on time/plies
    #compare_evals("eval_territory", "eval_territory", 1, 1_000_000)
    #compare_evals("eval_territory", "eval_territory", 1, 100_000)
    #compare_evals("eval_territory", "eval_territory", 1, 10_000)
    #compare_evals("eval_territory", "eval_territory", 1, 1000)

    # This allows a slightly deeper minimax in some situations, though it may
    # still be capped by maxStates
    # This one is SLOW
    #compare_mm_depths(2, 3, 1, 50_000)