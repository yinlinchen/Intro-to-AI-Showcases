import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os

# AGENT CLASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Agent:
    def __init__(self, x, y, environment):
        self.x = x
        self.y = y
        self.environment = environment
        self.satiated = False
        self.energy = 10000 # starting energy when agent is spawned
        self.speed = 1
        self.size = 10
        self.movement_cost = self.speed * (self.size ** 3)# 1/speed to get step cost, * speed^2 as biological limitation, add'l energy for add'l speed
        self.food_reward = 13000 # reward collected for consuming food
        self.stationary_penalty = 1000 # penalty for remaining stationary

    def manhattan(self, food):
        return (abs(self.x - food[0]) + abs(self.y - food[1]))

    # Movement is completely random
    def move_randomly(self):
        self.x += random.randint(-1, 1)
        self.y += random.randint(-1, 1)

    # take a step length towards closest food
    # this world has no walls, so pathfinding would be unnecessarily expensive
    # we can just move the exact manhattan distance
    def shortest_path_step(self, pos):
        spaces_to_move = self.speed
        pos_x, pos_y = pos
        agent_x = self.x
        agent_y = self.y
        y_delta = pos_y - agent_y
        x_delta = pos_x - agent_x
        consumed = y_delta == 0 and x_delta == 0

        while spaces_to_move > 0 and not consumed:
            if abs(y_delta) > abs(x_delta):
                # move down
                if y_delta < 0:
                    self.y = agent_y - 1
                # move up
                else:
                    self.y = agent_y + 1
            else:
                # move left
                if x_delta < 0:
                    self.x = agent_x - 1
                # move right
                else:
                    self.x = agent_x + 1
            # decrement number of moves in sequence
            spaces_to_move -= 1
            # consume energy for movement
            self.energy -= self.movement_cost
            # reset vars
            agent_x = self.x
            agent_y = self.y
            y_delta = pos_y - agent_y
            x_delta = pos_x - agent_x
            consumed = y_delta == 0 and x_delta == 0
    
    # Movement to closest food object
    def move_to_food(self):
        closest = None
        dist = float('inf')

        # get closest food
        for i in range(len(self.environment.food_grid)):
            distance_to_food = self.manhattan(self.environment.food_grid[i])
            if distance_to_food < dist:
                closest = self.environment.food_grid[i]
                dist = distance_to_food

        # determine whether the food is worth consuming
        if dist < (self.energy * self.speed):
            self.shortest_path_step(closest)
            self.eat()   
        else:
            self.energy -= self.stationary_penalty
            self.satiated = True

    def eat(self):
        if ((self.x, self.y) in self.environment.food_grid):
            self.environment.food_grid.remove((self.x, self.y))
            self.energy += self.food_reward
            

class Predator(Agent):
    def __init__(self, x, y, environment):
        super().__init__(x, y, environment)
        self.speed = 2
        self.size = 15
        self.movement_cost = self.speed * (self.size ** 4)
        self.hunt_reward = 100
        self.stationary_penalty = 15000

    def hunt(self, environment):
        if ((self.x, self.y) in environment.agents):
            environment.agents.remove((self.x, self.y))
            self.energy += self.hunt_reward
            print('eat!')

# class Prey(Agent):
#     def __init__(self, x, y, environment):
#         super().__init__(x, y, environment)
#         self.speed = 1
#         self.size = 5
#         self.movement_cost = self.speed * (self.size ** 3)
#         self.food_reward = 13000
#         self.stationary_penalty = 1000

# ENVIRONMENT CLASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Environment:
    def __init__(self, width, height, num_agents, num_food):
        self.width = width
        self.height = height
        self.num_food = num_food
        self.agents = [Agent(random.randint(0, width), random.randint(0, height), self) for _ in range(num_agents)] # 10 agents to start
        self.agent_counts = [len(self.agents)]
        self.food_grid = []
        self.remaining_food = []
        self.avg_energy = []
        self.avg_speed = []
        self.avg_size = []
        self.positions = [[] for _ in range(num_agents)]
        self.food_positions = []
        self.speed_dist = []
        self.size_dist = []
        
        self.predators = [Predator(random.randint(0, width), random.randint(0, height), self) for _ in range(int(num_agents*0.1))]
        self.pred_counts = [len(self.predators)]
        # self.preys = [Prey(random.randint(0, width), random.randint(0, height), self) for _ in range(num_agents)]

    def populate_food(self):
        self.food_grid = [(random.randint(0, self.width), random.randint(0, self.height)) for _ in range(self.num_food)]
        self.food_positions.append(self.food_grid)
        return self.food_positions

    def kill_the_weak(self):
        self.agents = [agent for agent in self.agents if agent.energy > 0.0]
        self.predators = [predator for predator in self.predators if predator.energy > 0.0]

    def calculate_energy_avg(self):
        energy = [agent.energy for agent in self.agents]
        avg = sum(energy) / len(energy)
        return avg
    
    def calculate_speed_avg(self):
        speed = [agent.speed for agent in self.agents]
        avg = sum(speed) / len(speed)
        return avg
    
    def calculate_size_avg(self):
        size = [agent.size for agent in self.agents]
        avg = sum(size) / len(size)
        return avg

    # function that simulates one generation
    def step(self):
        self.populate_food()
        all_agents_satiated = False
        
        # move all agents one step at a time until all are satiated
        while not all_agents_satiated:
            satiation = []
            for i, agent in enumerate(self.agents):
                agent.move_to_food()
                [a.hunt(self) for a in self.predators if (a.x, a.y) == (agent.x, agent.y)]
                satiation.append(agent.satiated)
                # print(i, len(self.positions), len(self.agents))
                if len(self.positions) <= i:
                    self.positions.append([])
                self.positions[i].append((agent.x, agent.y))
            all_agents_satiated = all(satiation)
        
        # remove agents with energy < 0
        self.kill_the_weak()

        # write metrics to lists for examiniation
        self.remaining_food.append(len(self.food_grid))
        if len(self.agents) > 0:
            self.avg_energy.append(self.calculate_energy_avg())
            self.avg_speed.append(self.calculate_speed_avg())
            self.avg_size.append(self.calculate_size_avg())
        else:
            self.avg_energy.append(0)
            self.avg_speed.append(0)
            self.avg_size.append(0)
            
        if len(self.agents) > 0:
            self.speed_dist.append([agent.speed for agent in self.agents])
            self.size_dist.append([agent.size for agent in self.agents])
        else:
            self.speed_dist.append([0])
            self.size_dist.append([0])

        # REALLOCATION OF SURVIVING AGENTS AND REPRODUCTION
        num_survivors = len(self.agents)
        new_agents = []
        # random mutation occurs with the following odds
        speed_boost = {0 : 10, 1 : 2, 2 : 1}
        size_boost = {0.85:1, 1:8, 1.15:1}

        for _ in range(num_survivors):
            # all surviving agents persist the next generation
            parent = random.choice(self.agents)
            parent.x = random.randint(0, self.width)
            parent.y = random.randint(0, self.height)
            parent_speed = parent.speed
            parent_size = parent.size
            new_parent = Agent(parent.x, parent.y, self)
            # all surviving agents will replicate
            new_parent.speed = parent_speed
            new_parent.size = parent_size
            new_agents.append(new_parent)
            # speed of child randomly mutates
            speed = parent_speed + random.choices(list(speed_boost.keys()), weights=list(speed_boost.values()), k=1)[0]
            size = parent_size * (random.choices(list(size_boost.keys()), weights=list(size_boost.values()), k=1)[0])
            if size <= 0.1:
                size = 0.1
            child_agent = Agent(random.randint(0, self.width), random.randint(0, self.height), self)
            child_agent.speed = speed
            child_agent.size = size
            new_agents.append(child_agent)
        
        num_survivors_p = int(len(self.predators)/2)
        new_predators = []
        speed_boost = {0 : 10, 1 : 2, 2 : -1}
        size_boost = {0.9:1, 1:8, 1.10:1}
        
        for _ in range(num_survivors_p):
            p_parent = random.choice(self.predators)
            p_parent.x = random.randint(0, self.width)
            p_parent.y = random.randint(0, self.height)
            p_parent_speed = parent.speed
            p_parent_size = parent.size
            p_new_parent = Predator(parent.x, parent.y, self)
            p_new_parent.speed = p_parent_speed
            p_new_parent.size = p_parent_size
            new_predators.append(p_new_parent)
            p_speed = p_parent_speed + random.choices(list(speed_boost.keys()), weights=list(speed_boost.values()), k=1)[0]
            p_size = p_parent_size * (random.choices(list(size_boost.keys()), weights=list(size_boost.values()), k=1)[0])
            if p_size <= 0.1:
                p_size = 0.1
            p_child_agent = Predator(random.randint(0, self.width), random.randint(0, self.height), self)
            p_child_agent.speed = p_speed
            p_child_agent.size = p_size
            new_predators.append(p_child_agent)
    
        self.agents = new_agents
        self.predators = new_predators

        # Add current number of agents to list
        self.agent_counts.append(len(self.agents))
        self.pred_counts.append(len(self.predators))
        
        # All agents movement plotted
        #mypath = os.path.dirname(os.path.abspath(__file__)) + '/'
        #if len(self.agents) > 0:
            # plot the movement of all agents at each time step
        #    fig, ax = plt.subplots()
        #   for i in range(len(self.positions)):
        #        x_values = [x[0] for x in self.positions[i]]
        #        y_values = [y[1] for y in self.positions[i]]
        #        ax.plot(x_values, y_values)
            
        #    fig.savefig(mypath + 'agents_movement.png')
        
        return self.agents, self.food_grid, self.positions

    # Function that generates animation in gif format
    def animate_generation(self, env, iterations, num_agents):
            
        plt.switch_backend('Agg') # don't show plot
        fig, (ax1, ax2) = plt.subplots(2, 1)

        # ax1 for agents and food locations in scatter plot
        ax1.set_xlim(0, self.width)
        ax1.set_ylim(0, self.height)
        ax1.set_title('Agents location')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # ax2 for number of agents over time (generations)
        ax2.set_title('Number of Agents')
        ax2.set_xlabel('Generations')
        ax2.set_ylabel('Number of Agents')
        ax2.set_xlim(0, iterations)
        ax2.set_ylim(0, int(num_agents*1.5))
        ax2.set_xticks(np.arange(0, iterations, 1))
        ax2.set_yticks(np.arange(0, int(num_agents*1.5), int(num_agents*0.2)))
        
        fig.set_figheight(8)
        fig.set_figwidth(5)
        fig.tight_layout()
        
        # initialise each plot
        scat_agents = ax1.scatter([a.x for a in self.agents], [a.y for a in self.agents], c='k', s=30, marker='*', label='Agent')
        scat_food = ax1.scatter([f[0] for f in self.food_grid], [f[1] for f in self.food_grid], c='r', s=30, marker='x', label='Food')
        line_agents, = ax2.plot([], [], c='k') # initialise empty line_agents

        num_agents_list = [num_agents]

        def update(frame_number):
            agent, food, pos = env.step()
            
            scat_agents.set_offsets(np.c_[[a.x for a in agent], [a.y for a in agent]])
            scat_food.set_offsets(np.c_[[f[0] for f in food], [f[1] for f in food]])
            
            num_agents = len(self.agents)
            
            # update the title of scatter plot
            ax1.set_title('Agents location - Generation #%i (No. of Agents: %s)' %(frame_number, num_agents))
            
            # monitor the number of agents at each generation
            num_agents_list.append(num_agents)
            line_agents.set_data(range(len(num_agents_list)), num_agents_list)
            
            return scat_agents, line_agents
        
        # Legend for each figure
        ax1.legend(loc='lower center', bbox_to_anchor = (0.5, -0.3), ncol=2)
        plt.subplots_adjust(wspace=0, hspace=0.4)
        
        # interval = delay between frames in milliseconds, e.g., 500 = 0.5 sec for each frame
        # blit = optimise drawing
        ani = animation.FuncAnimation(fig=fig,
                                        func=update,
                                        frames=iterations,
                                        interval=500,
                                        blit=False,
                                        repeat=True,
                                        )
        mypath = os.path.dirname(os.path.abspath(__file__)) + '/'
        # writergif = animation.PillowWriter(fps=30)
        # ani.save(mypath + "animation.gif", writer=writergif)
        ani.save(mypath + 'animation.gif', writer='pillow')
    
    def animate_agent(self):
        mypath = os.path.dirname(os.path.abspath(__file__)) + '/'
        figs = []
        for i in range(len(self.positions)):
            figs.append(plt.figure())
            
            def update(frame_number):
                plt.clf()
                plt.xlim([0,self.width])
                plt.ylim([0,self.height])
                plt.title(f"Time Step: {frame_number}")
                x_values = [x[0] for x in self.positions[i][:frame_number]]
                y_values = [y[1] for y in self.positions[i][:frame_number]]
                plt.plot(x_values, y_values)
                
                # food_x_values = [x[0] for x in self.food_positions[frame_number]]
                # food_y_values = [y[1] for y in self.food_positions[frame_number]]
                # plt.scatter(food_x_values, food_y_values)
                
            ani = animation.FuncAnimation(fig=figs[i],
                                      func=update,
                                      frames=len(self.positions[0]),
                                      interval=100,
                                      repeat=True)
            
            ani.save(mypath + f'individual_agents_animation/animation_agent_#{i}.gif', writer='pillow')

# DRIVER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def simulate(x, y, iterations, num_agents, num_food) :
    env = Environment(x, y, num_agents, num_food)
    #env.animate_generation(env, iterations, num_agents)
    #env.animate_agent(env, num_agents)
    for i in range(iterations) :
        print("Iteration Number " + str(i+1))
        print("Total Population: " + str(env.agent_counts[i]))
        print("Predator Population: " + str(env.pred_counts[i]))
        # print("Speed Distribution: " + str(env.speed_dist))
        env.step()
        
        if i == iterations - 1 :
            # print(env.agent_counts)
            return(env.agent_counts, env.pred_counts, env.avg_energy, env.avg_speed, env.avg_size, env.speed_dist, env.size_dist)