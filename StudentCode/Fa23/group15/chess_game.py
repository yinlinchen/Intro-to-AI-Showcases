import chess
import random

def print_board(board):
    print(board)

def get_move_from_user(board):
    print("Enter your move (e.g., 'e2e4'): ")
    while True:
        try:
            move = input()
            chess_move = chess.Move.from_uci(move)
            if chess_move in board.legal_moves:
                return chess_move
            else:
                print("Illegal move. Try again.")
        except ValueError:
            print("Invalid format. Try again.")

def make_ai_move(board):
    return random.choice(list(board.legal_moves))

def game_loop(play_with_ai):
    board = chess.Board()

    while not board.is_game_over():
        print_board(board)
        if play_with_ai and board.turn == chess.BLACK:
            print("AI is making a move...")
            ai_move = make_ai_move(board)
            board.push(ai_move)
        else:
            user_move = get_move_from_user(board)
            board.push(user_move)

    print("Game over")
    print("Result:", board.result())

# Ask the user if they want to play against AI
choice = input("Do you want to play against the AI? (yes/no): ").strip().lower()
play_with_ai = choice == "yes"

game_loop(play_with_ai)