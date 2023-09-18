"""
Snake Eater
Made with PyGame
"""

import pygame, sys, time, random
# from searchAgent import SearchAgent
# from qlearningAgents import QLearningAgent
import snakeGameData
import AStarAlgorithm
import qlearningAgents
import util


# Difficulty settings
# Easy      ->  10
# Medium    ->  25
# Hard      ->  40
# Harder    ->  60
# Impossible->  120
difficulty = 25 


frame_x = snakeGameData.frame_size_x
frame_y = snakeGameData.frame_size_y

# Checks for errors encountered
check_errors = pygame.init()
# pygame.init() example output -> (6, 0)
# second number in tuple gives number of errors
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')

# Initialise game window
pygame.display.set_caption('Snake Eater')
game_window = pygame.display.set_mode((frame_x, frame_y))


# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)


# FPS (frames per second) controller
fps_controller = pygame.time.Clock()

change_to = snakeGameData.direction
buttons = []

# Game Over
def game_over():
    my_font = pygame.font.SysFont('times new roman', 90)
    game_over_surface = my_font.render('YOU DIED', True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (frame_x/2, frame_y/4)
    game_window.fill(black)
    game_window.blit(game_over_surface, game_over_rect)
    show_score(0, red, 'times', 20)
    pygame.display.flip()

    while(True):
        for event in pygame.event.get():
            if event.key == ord('r'):
                reset_game()
                #play_game()
            if event.key == ord('q'):
                quit_game()

def reset_game():
    snakeGameData.snake_pos = [100, 50]
    snakeGameData.snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]

    snakeGameData.food_pos = [random.randrange(1, (frame_x//10)) * 10, random.randrange(1, (frame_y//10)) * 10]
    snakeGameData.food_spawn = True

def quit_game():
    pygame.quit()
    sys.exit()

# Score
def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(snakeGameData.score), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (frame_x/10, 15)
    else:
        score_rect.midtop = (frame_x/2, frame_y/1.25)
    game_window.blit(score_surface, score_rect)
    # pygame.display.flip()

def button(text, color, font, size):
    button_font = pygame.font.SysFont(font, size)
    button_surface = button_font.render(text, True, color)
    button_rect = button_surface.get_rect()
    buttons.append((button_surface, button_rect)) #really stupid may change up later to make a button class
    


def process_buttons():
    mousePos = pygame.mouse.get_pos()
    return NotImplemented

'''
def play_game():
    #not entirely sure why I have to do this, made the game happy when I did do it
    direction = 'RIGHT'
    change_to = 'RIGHT'
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game()
            
            #Call the agent here
            # Whenever a key is pressed down
            elif event.type == pygame.KEYDOWN:
                # W -> Up; S -> Down; A -> Left; D -> Right
                if event.key == pygame.K_UP or event.key == ord('w'):
                    change_to = 'UP'
                if event.key == pygame.K_DOWN or event.key == ord('s'):
                    change_to = 'DOWN'
                if event.key == pygame.K_LEFT or event.key == ord('a'):
                    change_to = 'LEFT'
                if event.key == pygame.K_RIGHT or event.key == ord('d'):
                    change_to = 'RIGHT'
                # Esc -> Create event to quit the game
                if event.key == pygame.K_ESCAPE:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
        

        # Making sure the snake cannot move in the opposite direction instantaneously
        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        # Moving the snake
        if direction == 'UP':
            snakeGameData.snake_pos[1] -= 10
        if direction == 'DOWN':
            snakeGameData.snake_pos[1] += 10
        if direction == 'LEFT':
            snakeGameData.snake_pos[0] -= 10
        if direction == 'RIGHT':
            snakeGameData.snake_pos[0] += 10

        # Snake body growing mechanism
        snakeGameData.snake_body.insert(0, list(snakeGameData.snake_pos))
        if snakeGameData.snake_pos[0] == snakeGameData.food_pos[0] and snakeGameData.snake_pos[1] == snakeGameData.food_pos[1]:
            snakeGameData.score += 1
            snakeGameData.food_spawn = False
        else:
            snakeGameData.snake_body.pop()

        # Spawning food on the screen
        if not snakeGameData.food_spawn:
            snakeGameData.food_pos = [random.randrange(1, (frame_x//10)) * 10, random.randrange(1, (frame_y//10)) * 10]
        snakeGameData.food_spawn = True

        # GFX
        game_window.fill(black)
        for pos in snakeGameData.snake_body:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        pygame.draw.rect(game_window, white, pygame.Rect(snakeGameData.food_pos[0], snakeGameData.food_pos[1], 10, 10))

        # Game Over conditions
        # Getting out of bounds
        if snakeGameData.snake_pos[0] < 0 or snakeGameData.snake_pos[0] > frame_x-10:
            game_over()
        if snakeGameData.snake_pos[1] < 0 or snakeGameData.snake_pos[1] > frame_y-10:
            game_over()
        # Touching the snake body
        for block in snakeGameData.snake_body[1:]:
            if snakeGameData.snake_pos[0] == block[0] and snakeGameData.snake_pos[1] == block[1]:
                game_over()

        show_score(1, white, 'consolas', 20)
        # Refresh game screen
        pygame.display.update()
        # Refresh rate
        fps_controller.tick(difficulty) #why should this be in the while loop???
'''







def play_game_ai(agent: qlearningAgents.QLearningAgent):
    #not entirely sure why I have to do this, made the game happy when I did do it
    change_to = 'RIGHT'
    moves = agent.getAction(agent.get_state)
    i = 0
    while True:
        if agent.gameOverState(snakeGameData.snake_pos, snakeGameData.snake_body):
            print("Game over! Score: " + snakeGameData.score )
        
        #if(i < len(moves)):
        #    change_to = moves[i]
        #i+=1
        moves = agent.getAction(agent.get_state)
        change_to = moves


        # Making sure the snake cannot move in the opposite direction instantaneously
        if change_to == 'UP' and snakeGameData.direction != 'DOWN':
            snakeGameData.direction = 'UP'
        if change_to == 'DOWN' and snakeGameData.direction != 'UP':
            snakeGameData.direction = 'DOWN'
        if change_to == 'LEFT' and snakeGameData.direction != 'RIGHT':
            snakeGameData.direction = 'LEFT'
        if change_to == 'RIGHT' and snakeGameData.direction != 'LEFT':
            snakeGameData.direction = 'RIGHT'

        # Moving the snake
        if snakeGameData.direction == 'UP':
            snakeGameData.snake_pos[1] -= 10
        if snakeGameData.direction == 'DOWN':
            snakeGameData.snake_pos[1] += 10
        if snakeGameData.direction == 'LEFT':
            snakeGameData.snake_pos[0] -= 10
        if snakeGameData.direction == 'RIGHT':
            snakeGameData.snake_pos[0] += 10

        # Snake body growing mechanism
        snakeGameData.snake_body.insert(0, list(snakeGameData.snake_pos))
        if snakeGameData.snake_pos[0] == snakeGameData.food_pos[0] and snakeGameData.snake_pos[1] == snakeGameData.food_pos[1]:
            snakeGameData.score += 1
            snakeGameData.food_spawn = False
        else:
            snakeGameData.snake_body.pop()

        # Spawning food on the screen
        if not snakeGameData.food_spawn:
            snakeGameData.food_pos = [random.randrange(1, (frame_x//10)) * 10, random.randrange(1, (frame_y//10)) * 10]
            # moves = agent.getAction(agent.get_state, qlearningAgents)
            moves = qlearningAgents.SnakeGameQAgent.getAction()
            i = 0
        snakeGameData.food_spawn = True

        # GFX
        game_window.fill(black)
        for pos in snakeGameData.snake_body:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        pygame.draw.rect(game_window, white, pygame.Rect(snakeGameData.food_pos[0], snakeGameData.food_pos[1], 10, 10))

        # Game Over conditions
        # Getting out of bounds
        if snakeGameData.snake_pos[0] < 0 or snakeGameData.snake_pos[0] > frame_x-10:
            #game_over()
            return
        if snakeGameData.snake_pos[1] < 0 or snakeGameData.snake_pos[1] > frame_y-10:
            #game_over()
            return
        # Touching the snake body
        for block in snakeGameData.snake_body[1:]:
            if snakeGameData.snake_pos[0] == block[0] and snakeGameData.snake_pos[1] == block[1]:
                #game_over()
                return

        show_score(1, white, 'consolas', 20)
        # Refresh game screen
        pygame.display.update()
        # Refresh rate
        fps_controller.tick(difficulty) #why should this be in the while loop???


thing = qlearningAgents.QLearningAgent()
#may change, default thing at the moment that just starts the game
play_game_ai(thing)

#def test_driver

class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # inspects the starting state
    """

    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        util.raiseNotDefined()
