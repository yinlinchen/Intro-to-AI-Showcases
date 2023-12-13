import graphics
import pygame
import chessboard
from minimax import MinimaxAgent
from chesspiece import KING_VAL
pygame.init()

ai_on = True
winner = None

running = True
if __name__ == "__main__":

    #initalize the board.
    curGameState = chessboard.Chessboard()
    curGameState.init_board()
    g = graphics.Graphics()
    minimaxAgent = MinimaxAgent()
    
    isPlayer = True
    while running:

        #determine if checkmate and assign winner.
        if curGameState.isCheckmate(isPlayer):
            winner = None
            if isPlayer:
                winner = "AI"
            else:
                winner = "Player"
            break

        #loop through pygame events to get input.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:

                #toggle AI for debugging.
                if event.key == pygame.K_LALT:
                    ai_on = not ai_on

            #detect if player clicked.
            if isPlayer and event.type == pygame.MOUSEBUTTONDOWN:

                #obtain which tile the player clicked on and get that piece
                mouse_x, mouse_y = event.pos
                col = mouse_x // g.TILE_SIZE
                row = mouse_y // g.TILE_SIZE
                selected_piece = curGameState.board[row][col]

                #be able to deselect by clicking tile again
                if (row, col) == g.selected_tile:
                    g.selected_tile = None
                    g.move_tiles = []
                
                #select a chess piece if the tile has a piece
                elif g.selected_tile == None and selected_piece != None:


                    threat_list = curGameState.isCheck(isPlayer)

                    #safely select piece if not in check and/or king.
                    if len(threat_list) == 0 or selected_piece.value == KING_VAL:
                        g.selected_tile = (row, col)
                        g.move_tiles = selected_piece.GetLegalMoves(curGameState)

                    #if king is in check, make sure only move pieces that can block.
                    elif len(threat_list) != 0 or selected_piece == KING_VAL:
                        move_tiles = selected_piece.GetLegalMoves(curGameState)

                        for move in move_tiles:
                            clone_state = curGameState.clone()
                            cloned_piece = clone_state.board[selected_piece.position[0]][selected_piece.position[1]]
                            clone_state.move_piece(cloned_piece, move)

                            if not clone_state.isCheck(isPlayer):
                                g.selected_tile = (row, col)
                                g.move_tiles.append(move)
                
                    #prevent king from moving in check positions
                    if selected_piece.value == KING_VAL:
                        move_tiles = selected_piece.GetLegalMoves(curGameState)
                        for move in move_tiles:
                            clone_state = curGameState.clone()
                            cloned_piece = clone_state.board[selected_piece.position[0]][selected_piece.position[1]]
                            clone_state.move_piece(cloned_piece, move)
                            if clone_state.isCheck(isPlayer):
                                g.move_tiles.remove(move)

                             
                #move the piece if possible.
                elif g.selected_tile != None:


                    move_piece = curGameState.board[g.selected_tile[0]][g.selected_tile[1]]

                    #deselect current piece if the attempted move is not a valid move.
                    if (row, col) not in g.move_tiles:
                        g.selected_tile = None
                        g.move_tiles = []

                    #move player and switch control to AI.
                    else:
                        curGameState.move_piece(move_piece, (row, col))
                        g.selected_tile = None
                        g.move_tiles = []
                        if ai_on:
                            isPlayer = not isPlayer
                                

            #is AI run minimax to update board
            elif not isPlayer:
                g.selected_tile = None
                g.move_tiles = []

                piece_move = minimaxAgent.get_next_move(curGameState)
                l = len(piece_move[1])
                move = piece_move[1]
                curGameState.move_piece(piece_move[0], move)
                isPlayer = not isPlayer
        
        g.draw_screen(curGameState)


    #Show the winner unitl window is closed.
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        g.draw_winner(winner)
        pygame.display.flip()

pygame.quit()
