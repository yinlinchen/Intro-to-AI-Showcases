import numpy as np
import random

# initialize a new game
def new_game(n):
    matrix = np.zeros([n, n])
    return matrix


def get_empty_cells(mat):
    empty_cells = []
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if (mat[i][j] == 0):
                empty_cells.append((i, j))
    return empty_cells


# add 2 or 4 in the matrix
def add_two(mat):
    empty_cells = []
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if (mat[i][j] == 0):
                empty_cells.append((i, j))
    if (len(empty_cells) == 0):
        return mat

    index_pair = empty_cells[random.randint(0, len(empty_cells) - 1)]

    prob = random.random()
    if (prob >= 0.9):
        mat[index_pair[0]][index_pair[1]] = 4
    else:
        mat[index_pair[0]][index_pair[1]] = 2
    return mat


# to check state of the game
def game_state(mat):
    # if 2048 in mat:
    #    return 'win'

    for i in range(len(mat) - 1):  # intentionally reduced to check the row on the right and below
        for j in range(len(mat[0]) - 1):  # more elegant to use exceptions but most likely this will be their solution
            if mat[i][j] == mat[i + 1][j] or mat[i][j + 1] == mat[i][j]:
                return 'not over'

    for i in range(len(mat)):  # check for any zero entries
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                return 'not over'

    for k in range(len(mat) - 1):  # to check the left/right entries on the last row
        if mat[len(mat) - 1][k] == mat[len(mat) - 1][k + 1]:
            return 'not over'

    for j in range(len(mat) - 1):  # check up/down entries on last column
        if mat[j][len(mat) - 1] == mat[j + 1][len(mat) - 1]:
            return 'not over'

    return 'lose'


def reverse(mat):
    new = []
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0]) - j - 1])
    return new


def transpose(mat):
    new = []
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])

    return np.transpose(mat)


def cover_up(mat):
    new = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    done = False
    for i in range(4):
        count = 0
        for j in range(4):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return (new, done)


def merge(mat):
    done = False
    score = 0
    for i in range(4):
        for j in range(3):
            if mat[i][j] == mat[i][j + 1] and mat[i][j] != 0:
                mat[i][j] *= 2
                score += mat[i][j]
                mat[i][j + 1] = 0
                done = True
    return (mat, done, score)


# up move
def up(game):
    game = transpose(game)
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = transpose(game)
    return (game, done, temp[2])


# down move
def down(game):
    game = reverse(transpose(game))
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = transpose(reverse(game))
    return (game, done, temp[2])


# left move
def left(game):
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    return (game, done, temp[2])


# right move
def right(game):
    game = reverse(game)
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = reverse(game)
    return (game, done, temp[2])


def perform_move(board, move):
    if move == 0:
        board, _, score = up(board)
    elif move == 1:
        board, _, score = left(board)
    elif move == 2:
        board, _, score = right(board)
    elif move == 3:
        board, _, score = down(board)
    else:
        print("ILLEGAL MOVE")
        return None, None
    return board, score


def main():
    print("NEW GAME")
    game = new_game(4)
    game = add_two(game)
    game = add_two(game)
    score = 0
    lost = False
    print(game)
    print()
    while not lost:
        move = ''
        while (not move.isdigit()) or (int(move) > 3):
            move = input('Enter move (0:up, 1:left, 2:right, 3:down): ')
        game, new_score = perform_move(game, int(move))
        score += new_score
        game = add_two(game)
        print(np.array(game))
        print()
        if game_state(game) == 'lose':
            print("YOU LOSE!")
            print("SCORE: ", score)
            lost = True


if __name__ == "__main__":
    main()
