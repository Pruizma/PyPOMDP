import argparse
import os
import json
import multiprocessing
from pomdp_runner import PomdpRunner
from util import RunnerParams
from logger import Logger as log
import statistics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Solve pomdp')
    parser.add_argument('config', type=str, help='The file name of algorithm configuration (without JSON extension)')
    parser.add_argument('--env', type=str, default='Tiger-2D.POMDP', help='The name of environment\'s config file')
    parser.add_argument('--budget', type=float, default=float('inf'), help='The total action budget (defeault to inf)')
    parser.add_argument('--snapshot', type=bool, default=False, help='Whether to snapshot the belief tree after each episode')
    parser.add_argument('--logfile', type=str, default=None, help='Logfile path')
    parser.add_argument('--random_prior', type=bool, default=False,
                        help='Whether or not to use a randomly generated distribution as prior belief, default to False')
    parser.add_argument('--max_play', type=int, default=1000, help='Maximum number of play steps')
    #New argument
    parser.add_argument('--benchmark', type=int, default=0, help='Sets the simulation to benchmark type, if present, must be followed of either "1" or "True"')

    args = vars(parser.parse_args())
    params = RunnerParams(**args)

    with open(params.algo_config) as algo_config:
        algo_params = json.load(algo_config)
        runner = PomdpRunner(params)
        #If interactive simulation is selected, the system will simulate only 1 simulation
        if params.benchmark == 0:
            runner.run(**algo_params)
        else:
            # If benchmark simulation is selected, the system will simulate provided simulations
            for i in range(params.benchmark):
                runner.runBench(**algo_params)

             #Showing final results
            log.info('\n'.join([
                '+'*20,
                'Results after ending {} simulations:'.format(params.benchmark),
                '='*20,
                'Total steps: {}'.format(runner.steps),
                'Final reward: {}'.format(runner.fReward),
                '='*5 + ' Analysing results ' + '='*5,
                'Average steps: {}'.format(runner.steps / params.benchmark),
                'Average reward: {}'.format(runner.fReward/params.benchmark),
                'Standard deviation of the steps: {}'.format(statistics.stdev(runner.step_list)),
                'Standard deviation of the reward: {}'.format(statistics.stdev(runner.fReward_list)),
                '=' * 20
            ]))