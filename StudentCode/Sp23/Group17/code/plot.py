import matplotlib.pyplot as plt
from predator_extension import simulate
import os

# # -----------------------input for simulation:
# env_x: x dimension of environment
# env_y: y dimension of environment
# generation: number of total generations to run
# num_agents: number of agents at start of simulation
# num_food: number of food items for agents to consume at each generation
env_x = 50
env_y = 50
generation = 20
num_agents = 50
num_food = 50

# run simulation
population, pred_population, food, avg_speed, avg_size, speed_dist, size_dist = simulate(env_x, env_y, generation, num_agents, num_food)

# plot
fig, axs = plt.subplots(5, figsize=(8, 8), constrained_layout=True)
fig.suptitle("Agent Population Speed, and Size Over Time")
axs[0].plot(population, 'k', label="Agents")
axs[0].plot(pred_population, 'r', label="Predators")
axs[0].set(ylabel="Population of Agents", xlabel="Generations")
axs[0].legend()
axs[0].xaxis.set_ticks(range(0, generation+1))
axs[1].plot(avg_speed, 'k')
axs[1].set(ylabel="Average Agent Speed", xlabel="Generations")
axs[0].xaxis.set_ticks(range(0, generation+1))
axs[2].plot(avg_size, 'k')
axs[2].set(ylabel="Average Agent Size", xlabel="Generations")
axs[2].xaxis.set_ticks(range(0, generation+1))
# histogram for population and speed distribution

cnts, values, bars = axs[3].hist(speed_dist[-1], bins=range(0,int(max(speed_dist[-1]))*2), edgecolor='k', align='left')
if len(cnts) == 0:
    cnts = 1
axs[3].set(ylabel="Frequency", xlabel="Speed")
axs[3].set_xlim(0, max(speed_dist[-1])+1)
axs[3].set_ylim(0, max(cnts)+1)
axs[3].xaxis.set_ticks(range(0, int(max(speed_dist[-1]))+2))
cmap = plt.cm.viridis
for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
    bar.set_facecolor(cmap(cnt/cnts.max()))

cnts, values, bars = axs[4].hist(size_dist[-1], bins=range(0,int(max(size_dist[-1]))*2), edgecolor='k', align='left')
axs[4].set(ylabel="Frequency", xlabel="Size")
axs[4].set_xlim(0, max(size_dist[-1])+1)
axs[4].set_ylim(0, max(cnts)+1)
axs[4].xaxis.set_ticks(range(0, int(max(size_dist[-1]))+2))
cmap = plt.cm.plasma
for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
    bar.set_facecolor(cmap(cnt/cnts.max()))

mypath = os.path.dirname(os.path.abspath(__file__)) + '/'
fig.savefig(mypath + "iteration_plot.png")
plt.show()