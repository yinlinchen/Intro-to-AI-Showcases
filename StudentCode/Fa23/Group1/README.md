# Boop-Agent

CS5804 AI Agent to play the board game _boop._

## Dependencies

- python 3.10 or greater (for annotation support)
- pytest (for running unit tests)


## Build

It is recommended to use conda and create a conda Python 3.10 environment:

```bash
conda create --name boop python=3.10
conda install --file requirements.txt
```

Depending on your OS and Python setup, you may also need to install Tk, e.g. for Ubuntu:

`sudo apt-get install python3-tk`


## Running a Game

```bash
python boop/boop.py
```

### Options

```
Options:
  -h, --help            show this help message and exit
  -a AI, --ai=AI        AI player, 1, 2, 'none', or 'both'
  -d DEPTH, --depth=DEPTH
                        Minimax depth. Recommended is 2.
  --depth2=DEPTH2       Depth of second agent, if both players are AI
  -m MAXSTATES, --maxStates=MAXSTATES
                        Max number of states to evaluate. This is a 'soft' max
                        as we don't bail from iterating the actions within a
                        state, and thus may exceed it slightly.
  --agent=AGENT         Agent type
  --agent2=AGENT2       Agent type for AI 2, if both players are AI, e.g. --ai both
  --eval=EVAL           Evaluation function
  --eval2=EVAL2         Evaluation function for second agent, if both
  -w WIDTH, --width=WIDTH
                        Beam width, when using BeamAgent
  --width2=WIDTH2       Beam width, when using BeamAgent
```

The evaluation functions may be any of:

`'eval_piece_count', 'eval_board_bonus', 'eval_territory', 'eval_stranding'`


## Testing

To run tests, simply execute:
`pytest .`

To print all messages to stdout:
`pytest -s .`

You may need to run pytest as a module, depending
on your installation:
`python -m pytest .`


## Running Experiments

See boop/experiments.py for which experiments
to enable. Know that they are CPU intensive.

`python boop/experiments.py`


Experiments print to stdout.
You may additionally wish to tee the results into a file, for long-running experiments.
To do this, you may wish to use the -u option for Python.

`python -u boop/experiments.py | tee <outfile>`


You may also run an AI vs AI scenario via the standard options, so long
as you set `--ai both`, e.g.

`python boop/boop.py --ai both --agent BeamAgent --width 10 --depth 10 --agent2 AlphaBetaAgent --depth2 2 --eval eval_territory --maxStates 10000`


## Results

The resulting games from the AI vs AI experiment are all located
in the results directory. Each experiment displays the paramters used
at the top of the file, each turn, and the number of plies, winner, and run time
at the bottom.