# This Class will represent the board in the boop game
from piece import Piece, Cat, Kitten, PlayerID

# Extent of the Board / bed
MAX_COORD = 5

# Type aliases to make annotations clearer
Point = tuple[int,int]
Triple = list[Point] # used in cases to denote 3 points
Pair = list[Point] # used in cases to denote 2 points


def off_bed(x: int, y: int) -> bool:
    """
    Returns whether the coord is off the bed.
    """
    return x < 0 or x > MAX_COORD or y < 0 or y > MAX_COORD


def on_bed(x: int, y: int) -> bool:
    """
    For convenience, inverse of off_bed
    """
    return not off_bed(x, y)


# Static cache of the tripling coords for fast-access:
# this is a mapping of an (x, y) coord
# to a list of pairs of the adjacent coords that
# make complete and valid triples.
TRIPLES: dict[Point, list[Pair]] = {}
for x in range(MAX_COORD+1):
    for y in range(MAX_COORD+1):
        pt = (x,y)
        pairs = [
            # row
            [(x-2, y), (x-1, y)],
            [(x-1, y), (x+1, y)],
            [(x+1, y), (x+2, y)],

            # col
            [(x, y-2), (x, y-1)],
            [(x, y-1), (x, y+1)],
            [(x, y+1), (x, y+2)],

            # diagonals
            [(x-2, y-2), (x-1, y-1)],
            [(x-1, y-1), (x+1, y+1)],
            [(x+1, y+1), (x+2, y+2)],

            [(x-2, y+2), (x-1, y+1)],
            [(x-1, y+1), (x+1, y-1)],
            [(x+1, y-1), (x+2, y-2)]
        ]
        pairs = [pair for pair in pairs if on_bed(pair[0][0], pair[0][1])
                 and on_bed(pair[1][0], pair[1][1])]
        TRIPLES[pt] = pairs


# Static cache of the L-coords for fast-access:
# this is a mapping of an (x, y) coord
# to a list of pairs of the adjacent coords that
# make L-shapes, used in evaluation functions.
L_COMPLETIONS: dict[Point, list[Pair]] = {}
for x in range(MAX_COORD+1):
    for y in range(MAX_COORD+1):
        pt = (x,y)
        pairs = [
            # left
            [(x-2, y), (x-2, y+2)],
            [(x-2, y), (x-2, y-2)],

            # right
            [(x+2, y), (x+2, y+2)],
            [(x+2, y), (x+2, y-2)],

            # middle up
            [(x, y+2), (x-2, y+2)],
            [(x, y+2), (x+2, y+2)],

            # middle down
            [(x, y-2), (x-2, y-2)],
            [(x, y-2), (x+2, y-2)]
        ]
        pairs = [pair for pair in pairs if on_bed(pair[0][0], pair[0][1])
                 and on_bed(pair[1][0], pair[1][1])]
        L_COMPLETIONS[pt] = pairs



class Board():
    """
    This is the 6x6 boop game board.
    By convention, we'll consider (0,0) to be the bottom left of the board,
    and access the grid by x then y, e.g. (1, 5) = grid[1][5].
    """

    def __init__(self):
        """
        Initializes Board with empty grid.
        NOTE: can't set as [[None]*(MAX_COORD+1)]*(MAX_COORD+1) or you get
        shallow copies of the same list per row, and setting one square sets an entire row!
        """
        self.grid: list[list[Piece]] = []
        for x in range(MAX_COORD+1):
            col = [None]*(MAX_COORD+1)
            self.grid.append(col)

        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

    
    def __eq__(self, other: object) -> bool:
        """
        Equality override for board comparison.
        This lets us write tests more easily against expected
        GameStates.
        """
        for x in range(MAX_COORD+1):
            for y in range(MAX_COORD+1):
                piece = self.grid[x][y]
                other_piece = other.grid[x][y]
                if not piece == other_piece:
                    return False

        return True


    def __getitem__(self, i) -> list[Piece]:
        """
        Provides index access to a square on the board.
        """
        return self.grid[i]
    

    def __setitem__(self, i, item):
        """
        Sets a square on the board.
        """
        self.grid[i] = item


    def __repr__(self):
        """
        Nicer representation of the grid.
        NOTE: to match the UI, now treating 0,0 as the upper left here, and drawing
        in terms of rows and cols as opposed to x and y, though internally either
        representation is equivalent.
        """
        s = " |------|------|------|------|------|------|\n"
        for row in range(self.rows):
            s += f"{row}|"
            for col in range(self.cols):
                piece = self.grid[row][col]
                if piece is None:
                    s += "      |"
                    continue
                player = 1 if piece.get_player() == PlayerID.ONE else 2
                if type(piece) is Kitten:
                    s += f" K({player}) |"
                else:
                    s += f" C({player}) |"
            s += "\n |------|------|------|------|------|------|\n"
        s += "    0      1      2      3      4      5    \n"
        return s


    @classmethod
    def boop_vector(cls, x: int, y: int, bx: int, by: int) -> Point:
        """
        Computes the boop vector for a piece at (bx, by)
        booped by a new piece at (x, y).
        """
        boop_vector = (bx - x, by - y)  # this will be 0 or +- 1 in either direction
        return boop_vector
    

    @classmethod
    def get_completion_points(cls, x: int, y: int) -> list[Pair]:
        """
        Returns lists of pairs of points that complement this
        point to form a complete triple.
        """
        return TRIPLES[(x,y)]
    

    @classmethod
    def get_L_compeltion_points(cls, x: int, y: int) -> list[Pair]:
        """
        Returns lists of pairs of points that complement this
        point to form an L shape.

        The first point of the Pair is always the corner.

        This is not used during the game, but can be used to evaluate
        strong positions in our heuristic evaluation functions.
        """
        return L_COMPLETIONS[(x, y)]
    

    @classmethod
    def get_overlapping(cls, triples: list[Triple]) -> list[Triple]:
        """
        Given lists of 3 pts, finds and returns triples with overlaps groups, if any.
        """
        # fast path (it is rare to have multiple, if any)
        if len(triples) <= 1:
            return []
        
        overlaps = set()
        for i in range(len(triples)):
            triple = triples[i]
            for j in range(i+1, len(triples)):
                other = triples[j]
                if Board.overlaps(triple, other):
                    overlaps.add(i)
                    overlaps.add(j)
        
        # convert to the actual triples
        return [triples[k] for k in overlaps]


    @classmethod
    def overlaps(cls, trip1: Triple, trip2: Triple) -> bool:
        """
        Returns True if these triples have an overlap, False otherwise.
        """
        for pt in trip1:
            if pt in trip2:
                return True
        
        return False
    

    @classmethod
    def surrounding_squares(cls, x: int, y: int) -> list[Point]:
        """
        Gets the squares adjacent to (x, y).
        Returns a list of up to 8 squares.
        This omits (x, y) itself, as well as any coordinates
        that would go off the board.
        """
        squares = []
        minX = max(x - 1, 0)
        maxX = min(x + 1, MAX_COORD)
        minY = max(y - 1, 0)
        maxY = min(y + 1, MAX_COORD)
        for nx in range(minX, maxX+1):
            for ny in range(minY, maxY+1):
                if not (nx == x and ny == y):
                    squares.append((nx, ny))
        return squares

    
    def empty_squares(self) -> list[Point]:
        """
        Returns list of empty (x,y) pairs on the board.
        """
        empties = []
        for x in range(MAX_COORD+1):
            for y in range (MAX_COORD+1):
                if self.grid[x][y] is None:
                    empties.append((x,y))
        return empties
    

    def copy(self) -> 'Board':
        """
        Makes a _shallow_ copy of the current game state / board.
        """
        board = Board()
        for i in range(len(self.grid)):
            for j in range (len(self.grid[i])):
                board[i][j] = self.grid[i][j]

        return board
    

    def num_cats(self, triple: Triple) -> int:
        """
        Returns number of cats in a triple of coords.
        """
        return sum(1 if type(self.grid[x][y]) is Cat else 0 for (x,y) in triple)
    

