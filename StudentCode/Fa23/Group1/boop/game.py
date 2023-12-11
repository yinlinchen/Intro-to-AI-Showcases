# This class holds the root of the game state, and the logic to generate successors
# and control valid moves.


from board import Board, off_bed, MAX_COORD, Point
from player import *
from piece import *

class GameState():
    """
    This class contains the game state.
    It can be copied and modified to evaluate successors, etc.
    """
    def __init__(self, player1: Player, player2: Player):
        """
        Constructs new GameState.
        """
        self.board = Board()
        self.turn: PlayerID = PlayerID.ONE
        self.player1 = player1
        self.player2 = player2
        self.winner = None
        self.must_promote = False
        self.plies = 0  # number of plies taken in the game 


    def get_player(self, id) -> Player:
        """
        Gets the player reference, given specified id.
        """
        if id == PlayerID.ONE:
            return self.player1
        return self.player2


    def current_player(self) -> Player:
        """
        Gets the current player reference, according to the turn.
        """
        return self.get_player(self.turn)


    def other_player(self) -> Player:
        """
        Gets the _ohter_ player reference, according to the turn.
        """
        cur = self.current_player()
        if cur == self.player1:
            return self.player2
        return self.player1
    

    def advance_turn(self):
        """
        Advances the game turn (really a ply).
        Note that we don't advance if there are pending decisions
        left to resolve.
        """
        if self.turn == PlayerID.ONE:
            self.turn = PlayerID.TWO
        else:
            self.turn = PlayerID.ONE
        
        # track plies as well, so we can determine length of game
        # both for statistics and evaluation
        self.plies += 1


    def copy(self) -> 'GameState':
        """
        Makes a new copy of the GameState.
        """
        gs = GameState(self.player1.copy(), self.player2.copy())
        gs.board = self.board.copy()
        gs.turn = self.turn
        gs.winner = self.winner
        gs.must_promote = self.must_promote
        gs.plies = self.plies
        return gs


    def get_legal_actions(self) -> list[tuple[Piece, int, int]]:
        """
        Given the current player's turn and hand, generates legal actions.
        Actions are a tuple of (Piece, x, y)
        """
        player = self.current_player()
        empties = self.board.empty_squares()
        actions = []
        if player.hand.has_kitten():
            kitten = Kitten(player.id)
            for (x,y) in empties:
                actions.append((kitten, x, y))
        
        if player.hand.has_cat():
            cat = Cat(player.id)
            for (x,y) in empties:
                actions.append((cat, x, y))
        
        return actions
    

    def get_legal_promotions(self) -> list[Point]:
        """
        Gets legal promotions: one per piece on board
        of the current player.
        """
        if not self.must_promote:
            raise Exception("get_legal_promotions called on state with no promotions")

        pieces = self.get_pieces_on_board(self.turn)
        spaces = [(x,y) for (_, x, y) in pieces]
        return spaces
    

    def get_pieces_on_board(self, id: PlayerID) -> list[tuple[Piece, int, int]]:
        """
        Gets a list of tuples of (Piece, x, y)
        for the given PlayerID.
        This is useful for both tracking promotions and for
        evaluation functions.
        """
        pieces = []
        for x in range(MAX_COORD+1):
            for y in range(MAX_COORD+1):
                piece = self.board[x][y]
                if piece is not None and piece.get_player() == id:
                    pieces.append((piece, x, y))
        return pieces
    

    def get_legal_selections(self, playerID: PlayerID) -> list[Triple]:
        """
        Gets legal selections for the specified player.
        Returns a shallow copy of the decisions for the player.
        """
        player = self.get_player(playerID)
        if not player.pending_decision():
            raise Exception("get_legal_selections called on player with no decisions")

        return player.decisions.copy()
    

    def has_pending_decision(self) -> bool:
        """
        Returns whether the current state has a pending decision that must be resolved
        before gameplay can continue.
        """
        return (self.must_promote
                or self.player1.pending_decision()
                or self.player2.pending_decision())


    def generate_successor_from_action(self, action) -> 'GameState':
        """
        Given the action, generates the successor game state.
        Note that this may setup pending decisions on the next game state.
        If there are any current pending decisions, this method will raise an Exception internally.
        These decisions should be used to generate a successor state first.
        
        Internally, this advances the turn for the next state unless the next state
        creates pending decisions.
        """
        gs = self.copy()
        gs.apply_action(action)
        return gs 


    def generate_successor_from_selection(self, playerID, selection) -> 'GameState':
        """
        Generates the successor state from the selected triple.
        NOTE: it is assumed that this will be called for the _current player's_ turn first,
        though doesn't really matter as the selections are independent of order, as long as
        we are consistent.

        If there are no more decisions, this advances the turn to the next player.
        """
        gs = self.copy()
        gs.resolve_selection(playerID, selection)
        return gs


    def generate_successor_from_promotion(self, x, y) -> 'GameState':
        """
        Generates the successor state from the selected promoted piece.

        This internally advances the turn, if there are no pending decisions for the other player.
        """
        gs = self.copy()
        gs.resolve_promotion(x, y)
        return gs
    

    def is_terminal(self) -> bool:
        """
        Returns True if we've reached a terminal state
        """
        return self.winner is not None


    def apply_move(self, playerID: PlayerID, move, log=False):
        """
        Helper to apply a move, which could be an action,
        selection, or promotion.
        Uses pattern matching to decide.
        """
        match move:
            case (piece, x, y) if isinstance(piece, Piece):
                # normal action
                if log:
                    print(f"AI action: {piece} to ({x}, {y})")
                self.apply_action(move)
            case (x, y):
                # promotion
                if log:
                    print(f"AI promotion: ({x}, {y})")
                self.resolve_promotion(x, y)
            case triple if triple is not None:
                # selection
                if log:
                    print(f"AI selection: {triple}")
                self.resolve_selection(playerID, triple)
            case _:
                if log:
                    print("Error: unknown action")


    def apply_action(self, action):
        """
        Applies an action to a game state.
        
        If an action should require a selection be made, this method tracks
        the list of possible decisions on each player, at least one of which must be applied.
        The intermediate state is noted as requiring a pending decision,
        which gives us an opportunity both to copy it again for each decision
        on the AI's turn, and to present a valid model for the UI to enable a user player's
        selection.

        This method will advance the current player at the end, UNLESS the state becomes terminal,
        or if decisions / promotions must be made.
        """
        if self.has_pending_decision():
            raise Exception("Attempted to apply action to state with pending decisions")

        (piece, x, y) = action
        if self.board[x][y] is not None:
            raise Exception("Illegal move")
        
        piece = self._play_piece(piece, x, y)
        if piece is None:
            raise Exception("Invalid piece")
 
        affected = self._apply_boops(piece, x, y)

        # Check for any resulting triples
        p1_triples, p2_triples = self._check_triples(affected)

        # if terminal, don't advance player or require a decision.
        # NOTE that this check sets the winner on the state.
        if self._check_terminal(p1_triples, p2_triples):
            # advance the plies for this turn though, as taken
            self.plies += 1
            return

        # if any overlapping sets, they require decisions, possibly per player
        p1_triples, self.player1.decisions = self._get_decisions(p1_triples)
        p2_triples, self.player2.decisions = self._get_decisions(p2_triples)

        # otherwise, promote the kittens to cats!
        self._graduate_pieces(p1_triples)
        self._graduate_pieces(p2_triples)

        # If not, then check for all 8 pieces: require decision on which to promote.
        # Just make sure they don't already have a pending decision that would
        # put cats back in hand.
        player = self.current_player()
        self.must_promote = not (player.hand.has_cat()
                             or player.hand.has_kitten()
                             or player.pending_decision())
        
        # If there are no situations to resolve, advance the turn
        if not self.has_pending_decision():
            self.advance_turn()


    def resolve_selection(self, playerID, selection):
        """
        Applies a selection to an intermediate game state which requires
        a tie-breaking decision.

        Selection itself is list of 3 xy coords [(x1, y1), (x2, y2), (x3, y3)]
        of which 3 pieces to promote, in the event that an action leads to multiple
        sets of 3 being available in the same turn.

        This will advance the turn, if the pending decisions for all players have been resolved.
        """
        if not self.has_pending_decision():
            raise Exception("Attempted to apply selection to state with no pending decisions")
        
        player = self.get_player(playerID)
        if not player.pending_decision():
            raise Exception("Attempted to apply selection to player with no pending decisions")
        
        # NOTE: be sure _not_ to modify the contents of player's decisions,
        # but reassign as needed.
        remaining = []
        for (x, y) in selection:
            # promote the selected pieces
            self._graduate_piece(x, y)


        # remove resolved decisions (likely only 1 in there)
        for decision in player.decisions:
            found = False
            for pt in decision:
                if pt in selection:
                    found = True
                    break
            
            if not found:
                # no piece of the selection was part of this (extremely rare if possible),
                # thus still must resolve
                remaining.append(decision)

        player.decisions = remaining
        
        if not self.has_pending_decision():
            self.advance_turn()


    def resolve_promotion(self, x, y):
        """
        Resolves pending promotion by selecting a piece.
        This is required when a player has all 8 pieces on the board at the end of their turn,
        but at least one is a kitten.

        This can only occur for the current player.

        This will advance the turn, if no pending decisions remain.
        """
        if not self.must_promote:
            raise Exception("Attempted to apply promotion to state with no pending promotion")
        
        self._graduate_piece(x, y)
        self.must_promote = False
        if not self.has_pending_decision():
            self.advance_turn()


    def _play_piece(self, piece, x, y) -> Piece:
        """
        Helper to actually play the piece of an action.
        """
        # Make sure we remove a piece of correct type from Player's hand
        if type(piece) is Kitten:
            piece = self.current_player().hand.get_kitten()
        else:
            piece = self.current_player().hand.get_cat()

        self.board[x][y] = piece
        return piece


    def _apply_boops(self, piece, x, y) -> list[tuple[Piece, int, int]]:
        """
        Give the piece moving to a space, apply boops to surrounding squares.
        """
        surrounding = Board.surrounding_squares(x, y)
        affected = [(piece, x, y)]
        for (sx, sy) in surrounding:
            sq_piece = self.board[sx][sy]
            if sq_piece is None:
                continue

            if not piece.can_boop(sq_piece):
                continue

            # calc where piece will land
            (bx, by) = Board.boop_vector(x, y, sx, sy)
            next_x = sx + bx
            next_y = sy + by

            # if booped off the bed, return to correct player's hand
            if off_bed(next_x, next_y):
                player = self.get_player(sq_piece.get_player())
                player.hand.add_piece(sq_piece)
                self.board[sx][sy] = None

            # otherwise: check if new space is clear
            elif self.board[next_x][next_y] is None:
                # move the piece!
                self.board[next_x][next_y] = sq_piece
                self.board[sx][sy] = None
                affected.append((sq_piece, next_x, next_y))

        return affected


    def _check_triples(self, affected_pieces) -> tuple[list[Triple], list[Triple]]:
        """
        Given the affected pieces from the last action, checks for new sets of triples.
        By nature of booping, these can't be adjacent, so we don't need to hunt for extra efficiency
        in that regard.
        """
        p1_triples = []
        p2_triples = []

        for (piece, x, y) in affected_pieces:
            triples = self._check_piece(piece, x, y)
            if piece.get_player() == PlayerID.ONE:
                p1_triples.extend(triples)
            else:
                p2_triples.extend(triples)

        return p1_triples, p2_triples

    
    def _check_piece(self, piece, x, y) -> list[Triple]:
        """
        Checks a piece to see if part of any triples.
        """
        triples = []
        pairs = Board.get_completion_points(x, y)
        for pair in pairs:
            if self._is_match(piece, pair[0], pair[1]):
                triple = [(x, y), pair[0], pair[1]]
                triples.append(triple)

        return triples
    

    def _get_decisions(self, triples) -> tuple[list[Triple], list[Triple]]:
        """
        Gets triples that require a user-decision to resolve.
        """
        decisions = []
        if len(triples) > 1:
            decisions = Board.get_overlapping(triples)
            triples = [trip for trip in triples if trip not in decisions]
        
        return triples, decisions
    

    def _graduate_pieces(self, triples):
        """
        For a given triples, graduates pieces (if applicable)
        and returns to the player's hand.

        NOTE: graduate is a no-op for existing Cats, keeping this simple
        """
        for triple in triples:
            for (x, y) in triple:
               self._graduate_piece(x, y)


    def _graduate_piece(self, x, y):
        """
        Graduates a single pieces(if applicable)
        and returns to the player's hand.
        """
        piece = self.board[x][y]
        piece = piece.graduate()
        self.get_player(piece.player).hand.add_piece(piece)
        self.board[x][y] = None


    def _check_terminal(self, p1_triples, p2_triples) -> bool:
        """
        Checks if this is a terminal GameState.
        There are two options for this:
        - A player has 3 cats in a row.
        - A player has all 8 _cats_ on the board.
        """
        # TODO: can there be a tie if both have triple cats?
        # Need to consult rules, current player may matter here!
        for triple in p1_triples:
            cats = self.board.num_cats(triple)
            if cats == 3:
                self.winner = PlayerID.ONE
                return True

        for triple in p2_triples:
            cats = self.board.num_cats(triple)
            if cats == 3:
                self.winner = PlayerID.TWO
                return True

        # check the other win condition: if all 8 pieces on the board are cats!
        player = self.current_player()
        if player.hand.has_cat() or player.hand.has_kitten():
            return False

        for col in self.board:
            for piece in col:    
                if (piece is not None
                    and piece.get_player() == self.turn
                    and type(piece) is Kitten):
                    return False

        # player has all pieces on the board, and all are Cats! They win!
        self.winner = player.id
        return True


    def _is_match(self, piece, pos1, pos2) -> bool:
        """
        Helper to check that the pieces at pos1 and pos2 match
        the player of piece.
        """
        (x1, y1) = pos1
        (x2, y2) = pos2

        # ensure positions are valid
        if off_bed(x1, y1) or off_bed(x2, y2):
            return False
        
        piece_2 = self.board[x1][y1]
        piece_3 = self.board[x2][y2]

        return (piece_2 is not None
            and piece_3 is not None
            and piece.same_player(piece_2)
            and piece.same_player(piece_3))
        


class Game():
    def __init__(self, player1: Player, player2: Player):
        """
        Constructs a game with two players, which may
        be human or AI.
        """
        self.player1 = player1
        self.player2 = player2
        self.state = GameState(player1, player2)


    def reset(self):
        """
        Resets the game with a new GameState
        and Players.
        """
        p1 = Player(PlayerID.ONE)
        p2 = Player(PlayerID.TWO)
        p1.agent = self.player1.agent
        p2.agent = self.player2.agent

        self.player1 = p1
        self.player2 = p2
        self.state = GameState(self.player1, self.player2)


    def get_state(self) -> GameState:
        """
        Returns the GameState for this game.
        """
        return self.state
    

    def update_ai_once(self):
        """
        Updates an AI agent's turn, if applicable.
        """
        current = self.state.current_player()
        other = self.state.other_player()

        # Who's turn is it?
        # This is a little tricky as resolution is independent
        # of the actual game turn.
        is_currents_turn = self.state.turn == current.id

        # In the case of resolution, prefer the current player first.
        # First, check to resolve any pending decisions:
        if self.state.has_pending_decision():
            # It is the current player's turn if they have a decision,
            # or if they must pick a piece to promote
            is_currents_turn = (current.pending_decision()
                or self.state.must_promote)
        
        player_to_move = current if is_currents_turn else other
        
        if player_to_move.is_agent():
            action, val = player_to_move.agent.getAction(self.state, player_to_move.id)

            # Update the state
            self.state.apply_move(player_to_move.id, action, True)

            # return the action for the UI to display, if desired
            return action
        
        return None
    

    def play_ai_match(self):
        """
        This plays an automated match between 2 ai agents.
        """
        while not self.state.is_terminal():
            self.update_ai_once()
            print(self.state.board)

        print(f"Winner is: {self.state.winner} in {self.state.plies} plies")
            

            
        
                
                


