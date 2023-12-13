import pygame
pygame.init()

PIECE_SIZE = 100

#Piece weights for minimax evaluation function.
PAWN_VAL = 100
ROOK_VAL = 500
BISHOP_VAL = 330
KNIGHT_VAL = 320
QUEEN_VAL = 900
KING_VAL = 2000

#Class to represent a chesspiece.
class Chesspiece:

    #creates a chess piece.
    def __init__(self, row, col, color, value):
        self.position = (row, col)
        self.color = color
        self.image = None
        self.value = value
        self.hasMoved = False

    #string representation for piece.
    def __repr__(self):
        return f"{self.color}:{self.value}"
    
    #returns all legal moves for a piece given a gamestate.
    def GetLegalMoves(self, gameState):
        return
    
    #clones a piece.
    def clone(self):
        return
    

#Pawn class.
class Pawn(Chesspiece):

    #creates a pawn.
    def __init__(self, row , col, color):
        super().__init__(row, col, color, PAWN_VAL)
        if color == "black":
            self.image = pygame.image.load("images/black-pawn.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))
            self.direction = 1
        else:
            self.direction = -1
            self.image = pygame.image.load("images/white-pawn.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))
    

    def clone(self):
        new_pawn = Pawn(self.position[0], self.position[1], self.color)
        new_pawn.hasMoved = self.hasMoved
        return new_pawn
    
    def GetLegalMoves(self, gameState):
        legal_moves = []
        cur_pos = self.position 

        #determine if can move 1 forward.
        move_1_row = cur_pos[0] + self.direction
        if (move_1_row >= 0 and move_1_row <= 7 and gameState.board[move_1_row][cur_pos[1]] == None):
            legal_moves.append( (move_1_row, cur_pos[1]) )
        
        #check first move 2 forward.
        move_2_row = cur_pos[0] + (2 * self.direction)
        if (not self.hasMoved and gameState.board[move_2_row][cur_pos[1]] == None and gameState.board[move_1_row][cur_pos[1]] == None):
            legal_moves.append( (move_2_row, cur_pos[1]) )
        
        #check diagonal overtake
        d1 = (cur_pos[0] + self.direction, cur_pos[1] + self.direction)
        d2 = (cur_pos[0] + self.direction, cur_pos[1] - self.direction)

        #first diagonal
        if 0 <= d1[0] <= 7 and 0 <= d1[1] <= 7 and gameState.board[d1[0]][d1[1]] != None:
            d_piece = gameState.board[d1[0]][d1[1]]
            if d_piece.color != self.color:
                legal_moves.append(d1)
        
        #second diagonal
        if 0 <= d2[0] <= 7 and 0 <= d2[1] <= 7 and gameState.board[d2[0]][d2[1]] != None:
            d_piece = gameState.board[d2[0]][d2[1]]
            if d_piece.color != self.color:
                legal_moves.append(d2) 
        return legal_moves

#Rook class.
class Rook(Chesspiece):

    #Creates rook object.
    def __init__(self, row , col, color):
        super().__init__(row, col, color, ROOK_VAL)
        if color == "black":
            self.image = pygame.image.load("images/black-rook.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))
        else:
            self.image = pygame.image.load("images/white-rook.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))
    
    def clone(self):
        new_rook = Rook(self.position[0], self.position[1], self.color)
        new_rook.hasMoved = self.hasMoved
        return new_rook
    
    def GetLegalMoves(self, gameState):
        legal_moves = []
        cur_pos = self.position

        #up direction
        for up in range(cur_pos[0] - 1, -1, -1):
            piece = gameState.board[up][cur_pos[1]]
            if piece == None:
                legal_moves.append( (up, cur_pos[1]) )
            elif piece.color != self.color:
                legal_moves.append( (up, cur_pos[1]) )
                break
            else:
                break
        
        #down direction
        for down in range(cur_pos[0] + 1, 8):
            piece = gameState.board[down][cur_pos[1]]
            if piece == None:
                legal_moves.append( (down, cur_pos[1]) )
            elif piece.color != self.color:
                legal_moves.append( (down, cur_pos[1]) )
                break
            else:
                break
        
        #left direction
        for left in range(cur_pos[1] - 1, -1, -1):
            piece = gameState.board[cur_pos[0]][left]
            if piece == None:
                legal_moves.append( (cur_pos[0], left) )
            elif piece.color != self.color:
                legal_moves.append( (cur_pos[0], left) )
                break
            else:
                break
        
        #right direction
        for right in range(cur_pos[1] + 1, 8):
            piece = gameState.board[cur_pos[0]][right]
            if piece == None:
                legal_moves.append( (cur_pos[0], right) )
            elif piece.color != self.color:
                legal_moves.append( (cur_pos[0], right) )
                break
            else:
                break
        
        return legal_moves


#Knight class.
class Knight(Chesspiece):

    #Creates Knight.
    def __init__(self, row , col, color):
        super().__init__(row, col, color, KNIGHT_VAL)
        if color == "black":
            self.image = pygame.image.load("images/black-knight.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))
        else:
            self.image = pygame.image.load("images/white-knight.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))

    def clone(self):
        new_knight = Knight(self.position[0], self.position[1], self.color)
        new_knight.hasMoved = self.hasMoved
        return new_knight
    
    def GetLegalMoves(self, gameState):
        cur_pos = self.position
        legal_moves = []

        #left hand side L's
        l1 = (cur_pos[0] - 1, cur_pos[1] - 2)
        l2 = (l1[0] + 2, l1[1])

        #upper L's
        u1 = (cur_pos[0] - 2, cur_pos[1] - 1)
        u2 = (u1[0], u1[1] + 2)

        #right side L's
        r1 = (cur_pos[0] - 1, cur_pos[1] + 2)
        r2 = (r1[0] + 2, r1[1])

        #down side L's
        d1 = (cur_pos[0] + 2, cur_pos[1] - 1)
        d2 = (d1[0], d1[1] + 2)

        #ensures 
        all_moves = [l1, l2, u1, u2, r1, r2, d1, d2]
        for move in all_moves:
            if 0 <= move[0] <= 7 and 0 <= move[1] <= 7:
                cur_piece = gameState.board[move[0]][move[1]]
                if cur_piece == None:
                    legal_moves.append(move)
                elif cur_piece.color != self.color:
                    legal_moves.append(move)

        return legal_moves

#bishop class
class Bishop(Chesspiece):


    def __init__(self, row , col, color):
        super().__init__(row, col, color, BISHOP_VAL)
        if color == "black":
            self.image = pygame.image.load("images/black-bishop.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))
        else:
            self.image = pygame.image.load("images/white-bishop.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))

    def clone(self):
        new_bishop = Bishop(self.position[0], self.position[1], self.color)
        new_bishop.hasMoved = self.hasMoved
        return new_bishop
    
    def GetLegalMoves(self, gameState):
        legal_moves = []
        cur_pos = self.position

        #upper right diagonal
        row = cur_pos[0] - 1
        col = cur_pos[1] + 1
        while row >= 0 and col <= 7:  
            cur_piece = gameState.board[row][col]
            if (cur_piece == None):
                legal_moves.append((row, col))
            elif cur_piece.color != self.color:
                legal_moves.append( (row, col) )
                break
            else:
                break
            row -= 1
            col += 1

        #upper left diagonal
        row = cur_pos[0] - 1
        col = cur_pos[1] - 1
        while row >= 0 and col >= 0:  
            cur_piece = gameState.board[row][col]
            if (cur_piece == None):
                legal_moves.append((row, col))
            elif cur_piece.color != self.color:
                legal_moves.append( (row, col) )
                break
            else:
                break
            row -= 1
            col -= 1
        
        #bottom right diagonal
        row = cur_pos[0] + 1
        col = cur_pos[1] + 1
        while row <= 7 and col <= 7:  
            cur_piece = gameState.board[row][col]
            if (cur_piece == None):
                legal_moves.append((row, col))
            elif cur_piece.color != self.color:
                legal_moves.append( (row, col) )
                break
            else:
                break
            row += 1
            col += 1
        
        #bottom left diagonal
        row = cur_pos[0] + 1
        col = cur_pos[1] - 1
        while row <= 7 and col >= 0:  
            cur_piece = gameState.board[row][col]
            if (cur_piece == None):
                legal_moves.append((row, col))
            elif cur_piece.color != self.color:
                legal_moves.append( (row, col) )
                break
            else:
                break
            row += 1
            col -= 1
        
        return legal_moves
                
#Queen class.
class Queen(Chesspiece):

    #Create Queen object.
    def __init__(self, row , col, color):
        super().__init__(row, col, color, QUEEN_VAL)
        if color == "black":
            self.image = pygame.image.load("images/black-queen.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))
        else:
            self.image = pygame.image.load("images/white-queen.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))

    def clone(self):
        new_queen = Queen(self.position[0], self.position[1], self.color)
        new_queen.hasMoved = self.hasMoved
        return new_queen
    
    def GetLegalMoves(self, gameState):

        #Combines rooks and bishops moves.
        cur_pos = self.position
        bishop_moves = Bishop(cur_pos[0], cur_pos[1], self.color).GetLegalMoves(gameState)
        rook_moves = Rook(cur_pos[0], cur_pos[1], self.color).GetLegalMoves(gameState)
        return bishop_moves + rook_moves

#King class.
class King(Chesspiece):

    #Create King object.
    def __init__(self, row , col, color):
        super().__init__(row, col, color, KING_VAL)
        self.castled = False
        if color == "black":
            self.image = pygame.image.load("images/black-king.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))
        else:
            self.image = pygame.image.load("images/white-king.png")
            self.image = pygame.transform.scale(self.image, (PIECE_SIZE, PIECE_SIZE))

    def clone(self):
        new_king = King(self.position[0], self.position[1], self.color)
        new_king.castled = self.castled
        new_king.hasMoved = self.hasMoved
        return new_king
    
    def GetLegalMoves(self, gameState):
        legal_moves = []
        #Get all tiles adjacent to King.
        for row in range(self.position[0] - 1, self.position[0] + 2):
            for col in range(self.position[1] - 1, self.position[1] + 2):
                if 0 <= row <= 7 and 0 <= col <= 7:
                    cur_piece = gameState.board[row][col]
                    if cur_piece == None or cur_piece.color != self.color:
                        legal_moves.append( (row, col) )
        
        #take in account king side castling.
        if not self.hasMoved and not self.castled:
            is_bishop = gameState.board[self.position[0]][5]
            is_knight = gameState.board[self.position[0]][6]
            is_rook = gameState.board[self.position[0]][7]
            swap_loc = (self.position[0], 6)
            if is_bishop == None and is_knight == None:
                if is_rook != None and is_rook.value == ROOK_VAL and not is_rook.hasMoved:
                    legal_moves.append(swap_loc)

        return legal_moves

        
        
