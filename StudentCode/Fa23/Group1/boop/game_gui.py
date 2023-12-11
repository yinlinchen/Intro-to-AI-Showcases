import tkinter as tk
from board import Triple
from piece import PlayerID
from PIL import Image, ImageTk
from game import Game
from piece import Piece


class BoardGUI:
    def __init__(self, master, game: Game):
        self.master = master
        self.game = game
        self.gamestate = game.get_state()
        self.board = self.gamestate.board
        self.player1 = self.gamestate.player1
        self.player2 = self.gamestate.player2
        self.selected_label = None
        self.selected_piece: Piece = None  # Keep track of the selected piece        
        self.winner_label = tk.Label(master, text="Winner")
        self.winner_label.grid(row=self.board.rows + 10, column=0, columnspan=self.board.cols, sticky="nsew")        

        # Create buttons for each cell in the board
        self.buttons = [[tk.Button(master, text="", width=5, height=2)
                 for j in range(self.board.cols)] for i in range(self.board.rows)]

        for i in range(self.board.rows):
            for j in range(self.board.cols):
                self.buttons[i][j].bind("<Button-1>", lambda event, i=i, j=j: self.cell_clicked(i, j))


        # Configure row and column weights
        for i in range(self.board.rows):
            master.grid_rowconfigure(i, weight=1)
        for j in range(self.board.cols):
            master.grid_columnconfigure(j, weight=1)

        # Create labels for player hands
        self.player1_hand_label = tk.Label(master, text="Player 1 Hand:")
        self.player1_hand_label.grid(row=self.board.rows, column=0, columnspan=self.board.cols, sticky="nsew")

        self.player2_hand_label = tk.Label(master, text="Player 2 Hand:")
        # Increase the row parameter to avoid overlapping with the buttons grid
        self.player2_hand_label.grid(row=self.board.rows + 5, column=0, columnspan=self.board.cols, sticky="nsew")

        # Display players' hands
        self.display_player_hand(self.player1, self.board.rows + 1, 1, True)  # Pass player ID as an argument
        # Increase the row parameter to avoid overlapping with the buttons grid
        self.display_player_hand(self.player2, self.board.rows + 6, 2, False)  # Pass player ID as an argument

        # Pack the buttons into the grid
        for i in range(self.board.rows):
            for j in range(self.board.cols):
                color = "light blue"
                self.buttons[i][j].grid(row=i, column=j, sticky="nsew")
                self.buttons[i][j].configure(bg=color)

        # see if an AI goes first        
        self.master.after(1000, self.check_ai_turn)

    def cell_clicked(self, row, col):
        # Handle the click event for a specific cell
        print(f"Cell clicked: {row}, {col}")        
        # Add a Cat or Kitten      
        self.gamestate.apply_action((self.selected_piece, row, col))
        self.selected_label.configure(bg="white")
        self.refresh_board()
        if self.gamestate.has_pending_decision():
            self.handle_decisions()
            print('finished decisions')
        
        self.selected_label = None
        self.selected_piece = None      

        # Refresh the board with the updated state
        self.refresh_board()     

        if self.gamestate.is_terminal():            
            self.reset_game()   
        # Pause for a moment before checking AI turn
        self.master.after(1000, self.check_ai_turn)        

    def refresh_board(self):

        # Clear existing images on buttons
        for i in range(self.board.rows):
            for j in range(self.board.cols):
                self.buttons[i][j].configure(image=None)
        
        # Clear existing player hand buttons,
        # but don't clear the label text
        for widget in self.master.winfo_children():
            if (isinstance(widget, tk.Label) and
                not widget is self.player1_hand_label and
                not widget is self.player2_hand_label):
                widget.destroy()
        
        # Iterate over the board and update the images on buttons
        for i in range(self.board.rows):
            for j in range(self.board.cols):
                piece = self.board.grid[i][j]

                if piece is not None:
                    # Load and display an image in the cell
                    image_path = piece.getImagePath()
                    image = Image.open(image_path)
                    image = image.resize((50, 50), Image.LANCZOS)  # Adjust the size as needed

                    photo = ImageTk.PhotoImage(image)
                    self.buttons[i][j].configure(image=photo)
                    self.buttons[i][j].image = photo  # Keep a reference to prevent the image from being garbage collected
                else:
                    self.buttons[i][j].image = None
            self.master.grid_rowconfigure(i, weight=1, uniform="row_group")
        # update hands
        print('updating hands')
        print(self.gamestate.turn == PlayerID.ONE)        
        isPlayer1 = self.gamestate.turn == PlayerID.ONE
        self.display_player_hand(self.player1, self.board.rows + 1, 1, isPlayer1)
        self.display_player_hand(self.player2, self.board.rows + 6, 2, not isPlayer1)

    def display_player_hand(self, player, row, player_id, isTurn: bool):
        # Display the pieces in the player's hand
        hand = []
        for piece in player.hand.kittens:
            hand.append(piece)

        for piece in player.hand.cats:
            hand.append(piece)

        for col, piece in enumerate(hand):
            if piece is not None:               
                image_path = piece.getImagePath()                
                image = Image.open(image_path)
                image = image.resize((50, 50), Image.LANCZOS)
                # test if label present and delete if there
                photo = ImageTk.PhotoImage(image)
                label = tk.Label(self.master, image=photo)
                label.photo = photo  # Keep a reference to prevent the image from being garbage collected
                label.grid(row=row, column=col)
                if isTurn:
                    label.configure(bg='gold')
                else:
                    label.configure(bg='white')
                # Bind click event to the label
                label.bind("<Button-1>", lambda event, piece=piece, player_id=player_id, label=label: self.piece_selected(event, piece, player_id, label))


    def piece_selected(self, event, piece: Piece, player_id, label):
        if piece.get_player() == self.gamestate.turn:
            # Handle the piece selection logic here
            print(f"Piece selected: {piece} from Player {player_id}")
            
            # Remove the visual effect from the previously selected label (if any)
            if self.selected_label:
                self.selected_label.configure(bg="white")

            # Add the visual effect to the selected label
        
            label.configure(bg="red")

            # Update the selected piece
            self.selected_piece = piece
            self.selected_label = label
    
    def handle_decisions(self):
        # check AI turn at the start of this: will resolve AI decision if any and it is the AI's turn
        self.master.after(1000, self.check_ai_turn)

        # NOTE: need to check this, in the case of 2 pending decisions with one AI that must go 2nd
        if self.gamestate.has_pending_decision(): 
            if self.gamestate.current_player().pending_decision():
                decisions = self.gamestate.current_player().decisions
                self.master.after(0, self.handle_decision(decisions, self.gamestate.current_player()))
                self.master.after(1000, self.check_ai_turn)
            
            if self.gamestate.other_player().pending_decision():
                decisions = self.gamestate.other_player().decisions
                self.master.after(0, self.handle_decision(decisions, self.gamestate.other_player()))
                self.master.after(1000, self.check_ai_turn)

            if self.gamestate.must_promote:           
                # Iterate through all pieces on the board
                for i in range(self.board.rows):
                    for j in range(self.board.cols):
                        piece = self.board.grid[i][j]

                        # Check if the piece belongs to the current player
                        if piece is not None and piece.get_player() == self.gamestate.turn:
                            # Highlight the piece                     
                            self.buttons[i][j].configure(bg='green')
                self.refresh_board()
                # Wait for the player to click on a piece
                self.master.after(0, self.wait_for_piece_click)
        print('finished handle decisions')

        # check AI turn
        self.master.after(1000, self.check_ai_turn)

    def wait_for_piece_click(self):
        # Used for promotions.
        # Bind a click event to each button on the board
        for i in range(self.board.rows):
            for j in range(self.board.cols):
                self.buttons[i][j].bind("<Button-1>", lambda event, i=i, j=j: self.piece_click_handler(i, j))

    def piece_click_handler(self, row, col):
        # This method is called when a button on the board is clicked
        piece = self.board.grid[row][col]
        if piece is not None and piece.get_player() == self.gamestate.turn:
            # reset the bg colors of the pieces on the board
            for i in range(self.board.rows):
                    for j in range(self.board.cols):
                        self.buttons[i][j].bind("<Button-1>", lambda event, i=i, j=j: self.cell_clicked(i, j))
                        piece = self.board.grid[i][j]
                        if piece is not None:                                                   
                            self.buttons[i][j].configure(bg='light blue')
            # A piece belonging to the current player is clicked
            self.gamestate.resolve_promotion(row, col)
            
            
            # Refresh the board with the updated state
            self.refresh_board()
            if self.gamestate.has_pending_decision():
                self.handle_decisions()
            else:
                self.master.after(1000, self.check_ai_turn)
            print('finished piece_click_handler')

    def handle_decision(self, decisions: list[Triple], player):
        selected = []        
        for i in range(len(decisions)):            
            for (x, y) in decisions[i]:                
                self.buttons[x][y].configure(bg='green')
                self.buttons[x][y].bind("<Button-1>", lambda event, i=x, j=y: self.add_clicked(i, j, selected, decisions, player))

    def add_clicked(self, row, col, selected, decisions: list[Triple], player):
        for i in range(len(decisions)):
            if (row, col) in decisions[i]:
                if len(selected) == 0:
                    selected.append((row, col))
                    self.buttons[row][col].configure(bg='blue')
                    break
                elif len(selected) == 1:
                    if self.check_adjacy(row, col, selected):
                        selected.append((row, col))
                        self.buttons[row][col].configure(bg='blue')
                        break
                elif len(selected) == 2:
                    if self.check_path(row, col, selected):
                        selected.append((row, col))                    
                        for i in range(self.board.rows):
                            for j in range(self.board.cols):
                                self.buttons[i][j].bind("<Button-1>", lambda event, i=i, j=j: self.cell_clicked(i, j))
                                piece = self.board.grid[i][j]
                                if piece is not None:                                                   
                                    self.buttons[i][j].configure(bg='light blue')
                        self.gamestate.resolve_selection(player.id, selected)
                        self.refresh_board()
                        # always check again, as this will do an AI turn too
                        self.handle_decisions()
            

    def check_adjacy(self, row, col, selected):        
        return (
            (row == selected[0][0] and col == selected[0][1] + 1) or \
            (col== selected[0][1] and row == selected[0][0] + 1) or \
            (row == selected[0][0] + 1 and col== selected[0][1] + 1) or \
            (row == selected[0][0] - 1 and col== selected[0][1] + 1) or \
            (row == selected[0][0] - 1 and col== selected[0][1] - 1) or \
            (row == selected[0][0] + 1 and col== selected[0][1] - 1)
        )
    
    def check_path(self, row, col, selected):
        if selected[0][0] == selected[1][0] and selected[0][1] + 1 == selected[1][1]:
            return selected[1][0] == row and selected[1][1] + 1 == col
        elif selected[0][1] == selected[1][1]:
            return selected[1][1] == col and selected[1][0] + 1 == row
        elif selected[0][0] + 1 == selected[1][0] and selected[0][1] + 1 == selected[1][1]:
            return selected[1][0] + 1 == row and selected[1][1] + 1 == col
        elif selected[0][0] - 1 == selected[1][0] and selected[0][1] + 1 == selected[1][1]:
            return selected[1][0] - 1 == row and selected[1][1] + 1 == col
        elif selected[0][0] - 1 == selected[1][0] and selected[0][1] - 1 == selected[1][1]:
            return selected[1][0] - 1 == row and selected[1][1] - 1 == col
        elif selected[0][0] + 1 == selected[1][0] and selected[0][1] - 1 == selected[1][1]:
            return selected[1][0] + 1 == row and selected[1][1] - 1 == col
        else:
            return False
    
    def check_ai_turn(self):
        print(self.gamestate.board)
        self.refresh_board()
        action = self.game.update_ai_once()
        if self.gamestate.is_terminal():            
            self.reset_game()

        while action is not None:
            print(self.gamestate.board)
            self.refresh_board()  # board was changed
            action = None

            # for case of agent vs agent, or further resolution requried,
            # we can check again
            if self.gamestate.has_pending_decision():
                action = self.game.update_ai_once()  # may or may not be AI turn
        
        # otherwise, it's a human's turn
    
    def reset_game(self):
        winner = self.gamestate.winner
        popup_window = tk.Toplevel(self.master)
        popup_window.geometry("200x200")
        winner_label = tk.Label(popup_window, text="Winner!")
        winner_label.grid(row=self.board.rows + 10, column=0, columnspan=self.board.cols, sticky="nsew")

        # Check if the winner_label still exists
        if winner_label.winfo_exists():
            winner_label.config(text="")
            if winner:
                winner_label.config(text=f"Player {winner.value} wins!")

        # Pause for a moment before resetting
        winner_label.after(2000, self._reset_game())

    def _reset_game(self):
        self.refresh_board()
        self.game.reset()       
        self.gamestate = self.game.get_state()
        self.board = self.gamestate.board
        self.player1 = self.gamestate.player1
        self.player2 = self.gamestate.player2
        self.selected_label = None
        self.selected_piece: Piece = None
        self.refresh_board()    
    
def run_gui(game: Game):
    # Create the main window
    window = tk.Tk()

    # Set the initial size of the window (width x height)
    window.geometry("400x400")

    # Create an instance of BoardGUI, passing the main window and the game board
    board_gui = BoardGUI(window, game)

    if game.get_state().is_terminal():
        game.on_Terminal()
        board_gui.refresh_board()

    # Run the Tkinter event loop
    window.mainloop()
