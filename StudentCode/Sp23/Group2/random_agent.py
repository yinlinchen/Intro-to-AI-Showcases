from game import *

moves = [0, 1, 2, 3]


def get_move():
    return random.choice(moves)


def get_max_tile(board):
    max_tile = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] > max_tile:
                max_tile = board[i][j]
    return max_tile


def main():
    iterations = 10000
    scores = np.zeros(iterations)
    max_tiles = np.zeros(iterations)
    for i in range(iterations):
        board = new_game(4)
        board = add_two(board)
        board = add_two(board)
        lost = False
        while not lost:
            board, new_score = perform_move(board, get_move())
            scores[i] += new_score
            board = add_two(board)
            if game_state(board) == 'lose':
                print(f"Game {i} score: {scores[i]}")
                max_tiles[i] = get_max_tile(board)
                lost = True
    print(f'Scores: {scores}')
    print(f'Max tiles: {max_tiles}')
    print(f'Average score: {scores.mean()}')
    print(f'Standard Deviation of scores: {scores.std()}')
    print('How often agent reached tile:')
    tile = int(max_tiles.max())
    while tile >= 2:
        print(f'{tile}: {np.sum(max_tiles >= tile) / iterations}')
        tile = int(tile / 2)


if __name__ == "__main__":
    main()
