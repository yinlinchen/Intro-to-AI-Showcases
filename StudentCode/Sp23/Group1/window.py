import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
import math

# Window size
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

# Marker size
MARKER_WIDTH = round(WINDOW_WIDTH * 0.0604838)
MARKER_HEIGHT = round(WINDOW_WIDTH * 0.0604838)

# Base
TOP_LEFT = ( round(WINDOW_WIDTH * 0.0887096), round(WINDOW_HEIGHT * 0.0812807) )
STRIDE_BOARD_X = round(WINDOW_WIDTH * 0.3)
STRIDE_BOARD_Y = round(WINDOW_HEIGHT * 0.3)
STRIDE_X = round(WINDOW_WIDTH * 0.08064512)
STRIDE_Y = round(WINDOW_WIDTH * 0.086206)

# Highlight box
transparent_surface = pygame.Surface((STRIDE_BOARD_X, STRIDE_BOARD_Y), pygame.SRCALPHA)
player_1_color = (0, 0, 255, 128)
player_2_color = (255, 0, 0, 128)
playable_grid = (255, 255, 0, 128)

transparent_cell = pygame.Surface((STRIDE_X, STRIDE_Y), pygame.SRCALPHA)
last_move_cell_color = (0, 0, 0, 128)

SURFACE = pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE

class Window:
    def __init__(self, game):
        
        pygame.init()
        self.window  = pygame.display.set_mode( ( WINDOW_WIDTH, WINDOW_HEIGHT ), SURFACE )

        bg = pygame.image.load('images/blank_board.png').convert_alpha()
        self.background_image = pygame.transform.smoothscale(bg, ( WINDOW_WIDTH, WINDOW_HEIGHT ))

        x_marker = pygame.image.load('images/X.png').convert_alpha()
        self.x_marker = pygame.transform.smoothscale(x_marker, ( MARKER_WIDTH, MARKER_HEIGHT ))

        o_marker = pygame.image.load('images/O.png').convert_alpha()
        self.o_marker = pygame.transform.smoothscale(o_marker, ( MARKER_WIDTH, MARKER_HEIGHT ))

        self.playable_grid = None

        self.update(game.game_grid, game.simplified_grid, game.last_move)
    
    def update(self, game_grid, simplified_grid, last_move):
        self.window.fill( (255, 255, 255) )
        self.window.blit( self.background_image, ( 0,0 ) )

        base_x, base_y = TOP_LEFT

        for i in range(9):
            board_base = (base_x + math.floor(i % 3) * STRIDE_BOARD_X, base_y + math.floor(i / 3) * STRIDE_BOARD_Y)
            for j in range(9):
                if game_grid[i][j] == 1:
                    self.window.blit( self.x_marker, self.map_idx_to_pos(board_base, j) )
                if game_grid[i][j] == 2:
                    self.window.blit( self.o_marker, self.map_idx_to_pos(board_base, j) )

        # Highlight last move
        if last_move is not None:
            last_move_board, last_move_cell = last_move
            board_base = (base_x - 4 + math.floor(last_move_board % 3) * STRIDE_BOARD_X, base_y - 5 + math.floor(last_move_board / 3) * STRIDE_BOARD_Y)
            transparent_cell.fill(last_move_cell_color)
            self.window.blit(transparent_cell, self.map_idx_to_pos(board_base, last_move_cell))

        #Offset
        base_x -= 20
        base_y -= 18

        # Highlight playable grid
        if last_move is not None:
            last_move_board, last_move_cell = last_move

            if simplified_grid[last_move_cell] != 0 :
                for i in range(9):
                    board_base_x, board_base_y = (base_x + math.floor(i % 3) * STRIDE_BOARD_X, base_y + math.floor(i / 3) * STRIDE_BOARD_Y)
                    if simplified_grid[i] == 0:
                        transparent_surface.fill(playable_grid)
                        self.window.blit(transparent_surface, (board_base_x, board_base_y))
            else:
                board_base_x, board_base_y = (base_x + math.floor(last_move_cell % 3) * STRIDE_BOARD_X, base_y + math.floor(last_move_cell / 3) * STRIDE_BOARD_Y)
                transparent_surface.fill(playable_grid)
                self.window.blit(transparent_surface, (board_base_x, board_base_y))
            
        # High Light won boxes
        for i in range(9):
            board_base_x, board_base_y = (base_x + math.floor(i % 3) * STRIDE_BOARD_X, base_y + math.floor(i / 3) * STRIDE_BOARD_Y)
            if simplified_grid[i] == 1: 
                transparent_surface.fill(player_1_color)
                self.window.blit(transparent_surface, (board_base_x, board_base_y))
            if simplified_grid[i] == 2: 
                transparent_surface.fill(player_2_color)
                self.window.blit(transparent_surface, (board_base_x, board_base_y))
        pygame.display.update()

    def map_idx_to_pos(self, board_base, idx):
        base_x, base_y = board_base
        return (base_x + math.floor(idx % 3) * STRIDE_X, base_y + math.floor(idx / 3) * STRIDE_Y)

    def make_move(self):
        made_move = False
        while not made_move:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    board, cell = self.map_click_to_idx(mouse_x, mouse_y)
                    if board is not None and cell is not None:
                        made_move = True
        return board, cell


    def map_click_to_idx(self, x, y):
        top_x, top_y = TOP_LEFT
        fixed_x, fixed_y = x - top_x, y - top_y
        board_x = math.floor(fixed_x / STRIDE_BOARD_X)
        board_y = math.floor(fixed_y / STRIDE_BOARD_Y)
        fixed_x, fixed_y = fixed_x - (STRIDE_BOARD_X * board_x), fixed_y - (STRIDE_BOARD_Y * board_y)
        cell_x = math.floor(fixed_x / STRIDE_X)
        cell_y = math.floor(fixed_y / STRIDE_Y)
        board = board_x + board_y * 3
        cell = cell_x + cell_y * 3
        return board, cell


    def delay(self, t):
        pygame.time.delay(t)

    def quit():
        pygame.quit()