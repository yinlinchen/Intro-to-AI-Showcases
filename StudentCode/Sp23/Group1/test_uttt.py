import unittest
from uttt_main import UTTT

class TestUTTT(unittest.TestCase):
    def test_get_valid_moves(self):
        game = UTTT()
        self.assertEqual(len(game.valid_moves), 81)
        game.make_move(4,1)
        print()
        # game.print_board()

        self.assertEqual(len(game.valid_moves), 9)
        game.make_move(1,2)
        print()
        # game.print_board()

        self.assertEqual(len(game.valid_moves), 9)
        game.make_move(2,4)
        print()
        # game.print_board()

    def test_get_valid_moves_with_won_board(self):
        game = UTTT()
        self.assertTrue(game.make_move(4,0))
        self.assertTrue(game.make_move(0,4))
        self.assertTrue(game.make_move(4,1))
        self.assertTrue(game.make_move(1,4))
        self.assertTrue(game.make_move(4,2))
        self.assertTrue(game.make_move(2,4))
        # print(game.valid_moves)
        # game.print_board()
        self.assertFalse(game.make_move(4,5))

    def test_make_move(self):
        game = UTTT()
        self.assertTrue(game.make_move(0, 0))
        self.assertFalse(game.make_move(1,1))

    def test_generate_next_uttt(self):
        game = UTTT()
        next_game = game.generate_next_uttt(4, 4)
        self.assertEqual(game.game_grid[4][4], 0)
        self.assertEqual(next_game.game_grid[4][4], 1)
        self.assertEqual(game.player_turn, 1)
        self.assertEqual(next_game.player_turn, 2)

    def test_check_board_win(self):
        game = UTTT()
        game.game_grid[0][0] = 1
        game.game_grid[0][1] = 1
        game.game_grid[0][2] = 1
        self.assertTrue(game.check_board_win(0, 1))
        self.assertFalse(game.check_board_win(0, 2))

    def test_fully_won_game(self):
        game = UTTT()
        self.assertTrue(game.make_move(4,0))
        self.assertTrue(game.make_move(0,4))
        self.assertTrue(game.make_move(4,1))
        self.assertTrue(game.make_move(1,4))
        self.assertTrue(game.make_move(4,2))
        self.assertTrue(game.make_move(2,4))
        self.assertTrue(game.make_move(1,2))
        self.assertTrue(game.make_move(2,1))
        self.assertTrue(game.make_move(1,5))
        self.assertTrue(game.make_move(5,1))
        self.assertTrue(game.make_move(1,8))
        self.assertTrue(game.make_move(8,1))
        self.assertTrue(game.make_move(7,2))
        self.assertTrue(game.make_move(2,7))
        self.assertTrue(game.make_move(7,5))
        self.assertTrue(game.make_move(5,7))
        self.assertTrue(game.make_move(7,8))
        self.assertEqual(game.winner, 1)
        # game.print_board()

    def test_end_game_tie_game(self):
        game = UTTT()
        game.game_grid = [[0,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1]]
        
        game.make_move(0,0)
        game.print_board()
        self.assertEqual(game.winner, -1)

    def test_end_game_won_game(self):
        game = UTTT()
        game.game_grid = [[0,1,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1]]
        
        game.make_move(0,0)
        self.assertEqual(game.winner, 1)

    def test_end_game_not_won_game(self):
        game = UTTT()
        game.game_grid = [[0,0,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1],
                          [1,2,1,1,2,2,2,1,1]]
        
        game.make_move(1,0)
        self.assertEqual(game.winner, 0)
        


if __name__ == '__main__':
    unittest.main()