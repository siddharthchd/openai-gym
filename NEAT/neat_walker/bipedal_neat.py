import neat
import gym
import pickle
import numpy as np
import visualize
import os
import multiprocessing
import pickle

cpu_count = multiprocessing.cpu_count()
env = gym.make('BipedalWalker-v3')

def environment(net):

    total_reward = 0
    obs = env.reset()
    obs = obs[0 : len(obs)]
    done = False
    t = 0

    while not done:

        t += 1

        action = net.activate(obs)

        env.render()

        observation, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print('Ending episode at time step {} with fitness {}'.format(t, total_reward))
            break

    if type(total_reward) is type(None):
        print('Total reward is None. Fail!')

    return total_reward

def eval_genomes(genome, config):

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    rewards = np.array(environment(net))

    return np.average(rewards)

def run(config_file):

    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_file)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))

    parallelEval = neat.ParallelEvaluator(cpu_count, eval_genomes)
    print('Evaluating on {} CPUs'.format(cpu_count))

    winner = population.run(parallelEval.evaluate, 1000)

    print('\n Best Genome : \n {}'.format(winner))
    print('\n\n Output : ')

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog = False, view = True)
    visualize.plot_species(stats, view = True)

    environment(winner_net)

if __name__ == '__main__':

	
    config_file = 'neat_config_walker'
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, config_file)
    

    run(config_file)
