import chess
import pygame
from pygame.locals import QUIT, MOUSEBUTTONDOWN
from chess_agent import ChessAgent

board_size = 600

"""
    Pygame GUI - draws the chess board and pieces
"""
def draw_board(screen, board, piece_positions):
    # Load chessboard image
    chessboard_image = pygame.image.load("./images/board.png")
    chessboard_image = pygame.transform.scale(chessboard_image, (board_size, board_size))
    screen.blit(chessboard_image, (0, 0))

    # Load and draw chess pieces
    square_size = board_size // 8
    for rank in range(8):
        for file in range(8):
            piece = board.piece_at(chess.square(file, 7 - rank))
            if piece:
                piece_color = "w" if piece.color == chess.WHITE else "b"
                piece_filename = f"./images/{piece_color}_{piece.symbol().lower()}.png"

                # Draw piece at the current position
                if piece_positions.get(piece):
                    x, y = piece_positions[piece]
                else:
                    x, y = file * square_size, rank * square_size

                piece_image = pygame.image.load(piece_filename)
                piece_image = pygame.transform.scale(piece_image, (square_size, square_size))
                screen.blit(piece_image, (x, y))
    pygame.display.flip()

"""
    Game loops through player and agent turns
"""
def game_loop(board, screen, chess_agent):
    # Player goes first
    player_turn = True
    piece_positions = {}

    while not board.is_game_over():
            # Listen for user clicks
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                # User click event
                elif event.type == MOUSEBUTTONDOWN:
                    if player_turn:
                        x, y = pygame.mouse.get_pos()
                        file = x // (board_size // 8)
                        rank = 7 - y // (board_size // 8)  # Invert the y-coordinate to match chess notation
                        square = chess.square(file, rank)

                        # Player selected a piece
                        if board.piece_at(square) and board.piece_at(square).color == board.turn:
                            selected_square = square
                        elif 'selected_square' in locals():
                            move = chess.Move(selected_square, square)
                        
                            # Consider pawn promotion move
                            if (board.piece_at(selected_square) is not None and 
                                board.piece_at(selected_square).piece_type == chess.PAWN
                                and chess.square_rank(square) in [0, 7]
                            ):
                                move = handle_promotion(board, move);

                            # Push the valid move to the board
                            if move in board.legal_moves:
                                board.push(move)
                                # Print move to console
                                print("Player Move:", move.uci())

                                # Update board
                                draw_board(screen, board, piece_positions)
                                pygame.display.flip()

                                # End of player turn
                                player_turn = False
            if not player_turn:

                # Chess agent moves
                agent_move = chess_agent.choose_move(board)

                if agent_move:
                    # Push agent move to board
                    board.push(agent_move)
                    print("AI Move:", agent_move.uci())  # Print the AI's move to the console

                    # Check?
                    if board.is_check():
                        print("Check")

                    # End of agent turn
                    player_turn = True

            # Update the GUI
            draw_board(screen, board, piece_positions)

"""
    Handles a pawn promotion move by asking for user input
"""
def handle_promotion(board, move):
    # Ask what piece to promote the pawn to
    promotion_piece = None
    while promotion_piece is None:
        promotion_piece = input("Pawn promotion: Choose piece (Q/R/B/N): ").upper()
        if promotion_piece not in ["Q", "R", "B", "N"]:
            print("Invalid choice. Please choose Q, R, B, or N.")
            promotion_piece = None

    # Convert input letter to a chess piece
    if promotion_piece == "R":
        move.promotion = chess.ROOK
    elif promotion_piece == "B":
        move.promotion = chess.BISHOP
    elif promotion_piece == "N":
        move.promotion = chess.KNIGHT
    else:
        move.promotion = chess.QUEEN
    return move

"""
    Print the end result to console
"""
def print_result(board):
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"Checkmate - {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate")
    else:
        print(f"Unknown result")


"""
    Main method initializes and runs the game
"""
def main():
    # Initialize chess game and GUI
    pygame.init()
    board = chess.Board()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Chess Game")

    # Create the ChessAgent
    chess_agent = ChessAgent()

    # Play the game
    game_loop(board, screen, chess_agent)

    # Print the result
    print_result(board)
    
    # Keep the window open until user quits
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
    pygame.quit()

if __name__ == "__main__":
    main()