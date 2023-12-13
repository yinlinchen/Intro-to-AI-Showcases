# Blackjack AI
# Authored by Joseph Westerlund, Parsa Nikpour, and Ryan Clarke

import numpy as np
import random

class BlackjackGame:
    def __init__(self):
        # Initialize the game with an empty deck, player hand, dealer hand, and game state.
        self.deck = []
        self.player_hand = []
        self.dealer_hand = []
        self.game_over = False
        self.player_money = 1000  # Starting money for the player
        self.split_hand = None # A separate hand for the split action, if used

    def build_deck(self):
        # Creates and shuffles a standard deck of 52 playing cards
        suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
        values = list(range(2, 11)) + ["Jack", "Queen", "King", "Ace"]
        self.deck = [(value, suit) for suit in suits for value in values]
        random.shuffle(self.deck)

    def deal_card(self, hand):
        # Deals a card to a specified hand, and rebuilds the deck if empty
        if self.deck:
            card = self.deck.pop()
            hand.append(card)
            return card
        else:
            self.build_deck()
            return self.deal_card(hand)

    def total_hand(self, hand):
        # Calculates the total value of a hand, with special handling for Aces
        total = 0
        ace_count = 0
        for card in hand:
            if isinstance(card[0], int):
                total += card[0]
            elif card[0] in ["Jack", "Queen", "King"]:
                total += 10
            else:
                ace_count += 1
                total += 11

        # Adjusts total if Aces are present and total exceeds 21
        while total > 21 and ace_count:
            total -= 10
            ace_count -= 1

        return total

    def start_game(self):
        # Initializes and starts a new game round
        self.build_deck()
        self.player_hand = [self.deal_card(self.player_hand), self.deal_card(self.player_hand)]
        self.dealer_hand = [self.deal_card(self.dealer_hand), self.deal_card(self.dealer_hand)]

    def available_actions(self):
        # Determines the available actions for the player based on game rules
        actions = ["hit", "stand"]
        if self.can_split():
            actions.append("split")
        if len(self.player_hand) == 2:  # Double down is only available on the first move
            actions.append("double")
        return actions

    def can_split(self):
        # Checks if the player's hand can be split (same card value)
        return len(self.player_hand) == 2 and self.player_hand[0][0] == self.player_hand[1][0]
        
    def player_move(self, move):
        # Executes the player's chosen move
        if move == "hit":
            self.deal_card(self.player_hand)
        elif move == "stand":
            self.game_over = True
        elif move == "double":
            self.player_money -= 100  # Assumes a fixed bet amount for doubling
            self.deal_card(self.player_hand)
            self.game_over = True
        elif move == "split" and self.can_split():
            # Handles the split action
            self.split_hand = [self.player_hand.pop()]
            self.deal_card(self.player_hand)
            self.deal_card(self.split_hand)

    def check_winner(self):
        # Determines the winner of the round
        player_total = self.total_hand(self.player_hand)
        split_total = self.total_hand(self.split_hand) if self.split_hand else None
        dealer_total = self.total_hand(self.dealer_hand)

        # Stores results for the player and split hand, if any
        result = {}
        result['player'] = self.determine_outcome(player_total, dealer_total)
        if split_total is not None:
            result['split'] = self.determine_outcome(split_total, dealer_total)
        return result

    def determine_outcome(self, player_total, dealer_total):
        # Determines the outcome of the game based on totals
        if player_total > 21:
            return "lose"
        elif dealer_total > 21 or player_total > dealer_total:
            return "win"
        elif player_total < dealer_total:
            return "lose"
        else:
            return "tie"

    def play_round(self, move):
        # Manages the gameplay for one round
        if not self.split_hand or self.total_hand(self.split_hand) > 21:
            self.player_move(move)

        # Checks if the game is over and if so, allows the dealer to make a move
        if (self.game_over or self.total_hand(self.player_hand) > 21) and (not self.split_hand or self.total_hand(self.split_hand) > 21):
            self.dealer_move()
            self.game_over = True

        return {
            "player_hand": self.player_hand,
            "split_hand": self.split_hand,
            "dealer_hand": self.dealer_hand,
            "result": self.check_winner() if self.game_over else None,
            "game_over": self.game_over
        }

    def dealer_move(self):
        # Automatically plays for the dealer based on standard rules
        while self.total_hand(self.dealer_hand) < 17:
            self.deal_card(self.dealer_hand)

class QLearningAgent:
    def __init__(self, alpha=0.02, gamma=1.0, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1):
        # Initializes the Q-learning agent with learning parameters
        self.q_table = dict()  # Q-table to store the state-action values
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Rate of decay for epsilon
        self.epsilon_min = epsilon_min  # Minimum value for epsilon

    def choose_action(self, state):
        # Chooses an action (exploration or exploitation) based on epsilon
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(["hit", "stand"])
        else:
            return self.best_action(state)
        
    def update_epsilon(self):
        # Updates epsilon for the exploration-exploitation balance
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def best_action(self, state):
        # Chooses the best action for a given state from the Q-table
        self.q_table.setdefault(state, {"hit": 0, "stand": 0})
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_table(self, old_state, action, reward, new_state):
        # Updates the Q-table based on the agent's experience
        # Ensure state-action pairs are initialized in the Q-table
        if old_state not in self.q_table:
            self.q_table[old_state] = {'hit': 0, 'stand': 0}
        if new_state not in self.q_table:
            self.q_table[new_state] = {'hit': 0, 'stand': 0}

        # Q-learning update formula
        old_value = self.q_table[old_state][action]
        future_rewards = max(self.q_table[new_state].values())
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * future_rewards)
        self.q_table[old_state][action] = new_value

def train_agent(episodes=100000):
    # Trains the Q-learning agent over a specified number of episodes
    agent = QLearningAgent()
    win_count = 0
    lose_count = 0
    tie_count = 0

    for episode in range(episodes):
        game = BlackjackGame()
        game.start_game()

        while not game.game_over:
            state = (game.total_hand(game.player_hand), game.dealer_hand[1][0], "hit" in game.available_actions())
            action = agent.choose_action(state)
            result = game.play_round(action)
            new_state = (game.total_hand(game.player_hand), game.dealer_hand[1][0], "hit" in game.available_actions())
            
            # Assign rewards based on game outcome
            if result['game_over']:
                if result['result']['player'] == 'win':
                    reward = 2
                    win_count += 1
                elif result['result']['player'] == 'lose':
                    reward = -1
                    lose_count += 1
                else:  # tie
                    reward = 0
                    tie_count += 1
            else:
                reward = 0  # No immediate reward for continuing the game
            
            agent.update_q_table(state, action, reward, new_state)

        agent.update_epsilon()
        # Print results every 100 episodes for progress tracking
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Wins: {win_count}, Losses: {lose_count}, Ties: {tie_count}")

    return agent

def display_game_state(game):
    # Displays the current state of the Blackjack game
    print("\nPlayer's Hand: ", game.player_hand)
    print("Dealer's Showing: ", game.dealer_hand[0])
    print("Player's Total: ", game.total_hand(game.player_hand))

def user_action(agent, state):
    while True:
        print("Agent recommends:", agent.best_action(state))
        action = input("Choose an action (hit/stand/double/split) or see agent's recommendation (agent): ").lower()
        if action in ["hit", "stand", "double", "split"]:
            return action
        elif action == "agent":
            print("Agent's action:", agent.best_action(state))
        else:
            print("Invalid action. Please choose again.")

def user_action_2(agent, state):
    while True:
        print("Agent recommends:", agent.best_action(state))
        action = input("Choose an action (hit/stand) or see agent's recommendation (agent): ").lower()
        if action in ["hit", "stand"]:
            return action
        elif action == "agent":
            print("Agent's action:", agent.best_action(state))
        else:
            print("Invalid action. Please choose again.")

def play_blackjack(agent):
    game = BlackjackGame()
    game.start_game()

    while True:
        display_game_state(game)
        state = (game.total_hand(game.player_hand), game.dealer_hand[1][0], "hit" in game.available_actions())

        if game.can_split():
            action = user_action(agent, state)
        else:
            action = user_action_2(agent, state)

        result = game.play_round(action)
        display_game_state(game)

        if result['game_over']:
            print("Game Over! Result: ", result['result'])
            break

# Train the Q-learning agent
trained_agent = train_agent()

# Run the interactive Blackjack game
play_blackjack(trained_agent)