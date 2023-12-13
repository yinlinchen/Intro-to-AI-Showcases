import chesspiece

#class to reprsent the gameState, which in essence is the chessboard.
class Chessboard:

    NUM_COL = 8
    
    def __init__(self):
        self.NUM_COLUMNS = 8
        self.NUM_ROWS = 8
        self.white_pieces = []
        self.black_pieces = []
        self.board = None
        self.score = 0

    def getPawns(self, color):
        result = []
        pieces = self.white_pieces
        if color == "black":
            pieces = self.black_pieces
        
        for piece in pieces:
            if piece.value == chesspiece.PAWN_VAL:
                result.append(piece)
        
        return result
    
    #gets the appropiate king for whether it is the player or AI.
    def getKing(self, isPlayer):
        pieces = self.white_pieces
        if not isPlayer:
            pieces = self.black_pieces
        
        for piece in pieces:
            if piece.value == chesspiece.KING_VAL:
                return piece
        
        return None

    #determines if the current state is a win.
    def isWin(self):
        for piece in self.white_pieces:
            if piece.value == chesspiece.KING_VAL:
                return False
        return True
    
    #determines if the current state is a loss.
    def isLose(self):
        for piece in self.black_pieces:
            if piece.value == chesspiece.KING_VAL:
                return False
        return True
    
    def isCheckmate(self, isPlayer):

        if len(self.isCheck(isPlayer)) == 0:
            return False
        
        moves = self.GetLegalMoves(isPlayer)

        for piece, move_list in moves:
            for move in move_list:
                state_clone = self.clone()
                piece_clone = state_clone.board[piece.position[0]][piece.position[1]]
                state_clone.move_piece(piece_clone, move)
                if len(state_clone.isCheck(isPlayer)) == 0:
                    return False
        
        return True
    
    def get_piece(self, pos):

        all_pieces = self.white_pieces + self.black_pieces
        for piece in all_pieces:
            if piece.position == pos:
                return piece
        return None
    
    #determines the threat's to king for AI or player.
    def isCheck(self, isPlayer):
        piece_king = self.getKing(isPlayer)
        moves = self.GetLegalMoves( not isPlayer)
        threat_list = []
        for piece, move_list in moves:
            if piece_king.position in move_list:
                threat_list.append(piece)
            
        return threat_list
        
    #gets all legal moves for AI or player.
    def GetLegalMoves(self, isPlayer):
        legal_moves = []
        pieces = self.white_pieces
        if not isPlayer: pieces = self.black_pieces

        for piece in pieces:
            legals = piece.GetLegalMoves(self)
            legal_moves.append( (piece, legals))

        return legal_moves
    
    #generates the successor board given a piece move(a piece and a move).
    def GenerateSuccessor(self, piece_move):
        successor_state = self.clone()
        piece_pos = piece_move[0].position
        piece = successor_state.board[piece_pos[0]][piece_pos[1]]
        move = piece_move[1]
        successor_state.move_piece(piece, move)
        return successor_state

    #clones a board.
    def clone(self):
        new_state = Chessboard()
        new_board = [[None for _ in range(self.NUM_COLUMNS)] for _ in range(self.NUM_ROWS)]
        for row in range(self.NUM_ROWS):
            for column in range(self.NUM_COLUMNS):
                cur_piece = self.board[row][column]

                if (cur_piece != None):
                    clone_piece = cur_piece.clone()

                    new_board[row][column] = clone_piece
                    if (clone_piece.color == "white"):
                        new_state.white_pieces.append(clone_piece)
                    else:
                        new_state.black_pieces.append(clone_piece)

        new_state.board = new_board
        return new_state
    
    #moves a piece to a position.
    def move_piece(self, piece, position):
        if piece == None:
            return
        
        legal_moves = piece.GetLegalMoves(self)
        if position in legal_moves:

            self.board[piece.position[0]][piece.position[1]] = None

            piece.position = position

            new_position_piece = self.board[position[0]][position[1]]

            #remove piece from the list
            if (new_position_piece != None):
                if new_position_piece.color == "white":
                    self.white_pieces.remove(new_position_piece)
                else:
                    self.black_pieces.remove(new_position_piece)

            self.board[position[0]][position[1]] = piece

            #pawn moving and promotion
            if piece.value == chesspiece.PAWN_VAL:
                if piece.color == "black" and position[0] == 7:
                    black_queen = chesspiece.Queen(position[0], position[1], "black")
                    self.black_pieces.remove(piece)
                    self.black_pieces.append(black_queen)
                    self.board[position[0]][position[1]] = black_queen
                elif piece.color == "white" and position[0] == 0:
                    white_queen = chesspiece.Queen(position[0], position[1], "white")
                    self.white_pieces.remove(piece)
                    self.white_pieces.append(white_queen)
                    self.board[position[0]][position[1]] = white_queen

            #castling for the king
            if piece.value == chesspiece.KING_VAL:
                
                if not piece.hasMoved and position == (7, 6) or position == (0, 6):
                    rook = self.board[piece.position[0]][7]
                    if rook != None:
                        rook.position = (piece.position[0], 5)
                        self.board[piece.position[0]][5] = rook
                        rook.hasMoved = True
                        self.board[piece.position[0]][7] = None
                        piece.castled = True

            piece.hasMoved = True
            
    #initalizes the board
    def init_board(self):
        board = [[None for _ in range(self.NUM_COLUMNS)] for _ in range(self.NUM_ROWS)]

        #place white and black pawns
        for column in range(self.NUM_COLUMNS):
            white_pawn = chesspiece.Pawn(6, column, "white")
            black_pawn = chesspiece.Pawn(1, column, "black")
            board[6][column] = white_pawn
            board[1][column] = black_pawn
            self.white_pieces.append(white_pawn)
            self.black_pieces.append(black_pawn)

        #place rooks
        white_rook1 = chesspiece.Rook(7, 0, "white")
        white_rook2 = chesspiece.Rook(7, 7, "white")
        black_rook1 = chesspiece.Rook(0, 0, "black")
        black_rook2 = chesspiece.Rook(0, 7, "black")
        board[7][0] = white_rook1
        board[7][7] = white_rook2
        board[0][0] = black_rook1
        board[0][7] = black_rook2
        self.white_pieces.extend([white_rook1, white_rook2])
        self.black_pieces.extend([black_rook1, black_rook2])

        #place knights
        white_knight1 = chesspiece.Knight(7, 1, "white")
        white_knight2 = chesspiece.Knight(7, 6, "white")
        black_knight1 = chesspiece.Knight(0, 1, "black")
        black_knight2 = chesspiece.Knight(0, 6, "black")
        board[7][1] = white_knight1
        board[7][6] = white_knight2
        board[0][1] = black_knight1
        board[0][6] = black_knight2
        self.white_pieces.extend([white_knight1, white_knight2])
        self.black_pieces.extend([black_knight1, black_knight2])

        #place bishops
        white_bishop1 = chesspiece.Bishop(7, 2, "white")
        white_bishop2 = chesspiece.Bishop(7, 5, "white")
        black_bishop1 = chesspiece.Bishop(0, 2, "black")
        black_bishop2 = chesspiece.Bishop(0, 5, "black")
        board[7][2] = white_bishop1
        board[7][5] = white_bishop2
        board[0][2] = black_bishop1
        board[0][5] = black_bishop2
        self.white_pieces.extend([white_bishop1, white_bishop2])
        self.black_pieces.extend([black_bishop1, black_bishop2])

        #place queens
        white_queen = chesspiece.Queen(7, 3, "white")
        black_queen = chesspiece.Queen(0, 3, "black")
        board[7][3] = white_queen
        board[0][3] = black_queen
        self.white_pieces.append(white_queen)
        self.black_pieces.append(black_queen)

        #place kings
        white_king = chesspiece.King(7, 4, "white")
        black_king = chesspiece.King(0, 4, "black")
        board[7][4] = white_king
        board[0][4] = black_king
        self.white_pieces.append(white_king)
        self.black_pieces.append(black_king)

        self.board = board

    