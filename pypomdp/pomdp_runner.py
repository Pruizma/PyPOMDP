import os

from models import RockSampleModel, Model
from solvers import POMCP, PBVI
from parsers import PomdpParser, GraphViz
from logger import Logger as log

class PomdpRunner:

    #Creating variables to compute total steps and reward
    steps = 0
    fReward = 0
    # Creating variables to compute stdev of steps and reward
    step_list = []
    fReward_list = []

    def __init__(self, params):
        self.params = params
        if params.logfile is not None:
            log.new(params.logfile)

    def create_model(self, env_configs):
        """
        Builder method for creating model (i,e, agent's environment) instance
        :param env_configs: the complete encapsulation of environment's dynamics
        :return: concrete model
        """
        MODELS = {
            'RockSample': RockSampleModel,
        }
        return MODELS.get(env_configs['model_name'], Model)(env_configs)

    def create_solver(self, algo, model):
        """
        Builder method for creating solver instance
        :param algo: algorithm name
        :param model: model instance, e.g, TigerModel or RockSampleModel
        :return: concrete solver
        """
        SOLVERS = {
            'pbvi': PBVI,
            'pomcp': POMCP,
        }
        return SOLVERS.get(algo)(model)

    def snapshot_tree(self, visualiser, tree, filename):
        visualiser.update(tree.root)
        visualiser.render('./dev/snapshots/{}'.format(filename))  # TODO: parametrise the dev folder path


    def run(self, algo, T, **kwargs):
        visualiser = GraphViz(description='tmp')
        params, pomdp = self.params, None
        total_rewards, budget = 0, params.budget
        environment = params.env_config
        benchmark = params.benchmark

        log.info('~~~ initialising ~~~')
        with PomdpParser(params.env_config) as ctx:
            # creates model and solver
            model = self.create_model(ctx.copy_env())
            pomdp = self.create_solver(algo, model)

            # supply additional algo params
            belief = ctx.random_beliefs() if params.random_prior else ctx.generate_beliefs()

            if algo == 'pbvi':
                belief_points = ctx.generate_belief_points(kwargs['stepsize'])
                pomdp.add_configs(belief_points)
            elif algo == 'pomcp':
                pomdp.add_configs(budget, belief, **kwargs)

        # have fun!
        log.info('''
        ++++++++++++++++++++++
            Starting State:  {}
            Starting Budget:  {}
            Init Belief: {}
            Time Horizon: {}
            Max Play: {}
        ++++++++++++++++++++++'''.format(model.curr_state, budget, belief, T, params.max_play))

        for i in range(params.max_play):
            # plan, take action and receive environment feedbacks
            pomdp.solve(T)
            action = pomdp.get_action(belief)
            new_state, obs, reward, cost = pomdp.take_action(action)

            if params.snapshot and isinstance(pomdp, POMCP):
                # takes snapshot of belief tree before it gets updated
                self.snapshot_tree(visualiser, pomdp.tree, '{}.gv'.format(i))
            
            # update states
            belief = pomdp.update_belief(belief, action, obs)
            total_rewards += reward
            budget -= cost

            # Printing the details for every step of the interactive simulation
            # log.info('\n'.join([
            #     'Taking action: {}'.format(action),
            #     'Observation: {}'.format(obs),
            #     'Reward: {}'.format(reward),
            #     'Budget: {}'.format(budget),
            #     'New state: {}'.format(new_state),
            #     'New Belief: {}'.format(belief),
            #     '=' * 20
            # ]))

            # Tiger problem ----------------------------------------------------------------
            # When the open action is selected, the tiger problem will end, either the person scapes or is eaten by the tiger, so it has to stop.
            if "Tiger-2D.POMDP" in environment:
                    if "open" in action:
                        log.info('\n'.join([
                            'Taking action: {}'.format(action),
                            'Observation: {}'.format(obs),
                            'Reward: {}'.format(reward),
                            '=' * 20
                        ]))
                        break;
                    log.info('\n'.join([
                        'Taking action: {}'.format(action),
                        'Observation: {}'.format(obs),
                        'Reward: {}'.format(reward),
                        'New Belief: {}'.format(belief),
                        '=' * 20
                    ]))
            # Web ads problem ----------------------------------------------------------------
            # When the adv action is selected, the web ad problem will end, either the person gets a tie or a skate advertisement so it has to stop.
            if "Web.POMDP" in environment:
                if params.benchmark == 0:
                     if "adv" in action:
                        log.info('\n'.join([
                            'Taking action: {}'.format(action),
                            'Observation: {}'.format(obs),
                            'Reward: {}'.format(reward),
                             '=' * 20
                        ]))
                        break;
                     log.info('\n'.join([
                        'Taking action: {}'.format(action),
                        'Observation: {}'.format(obs),
                        'Reward: {}'.format(reward),
                        'New Belief: {}'.format(belief),
                        '=' * 20
                    ]))

            # Landing problem ----------------------------------------------------------------
            # When the arrive action is selected, the landing problem will end, either the tripulation finds a treasure or they die horribly to the creatures in the landing.
            if "Landing.POMDP" in environment:
                if "arrive" in action:
                    log.info('\n'.join([
                        'Taking action: {}'.format(action),
                        'Observation: {}'.format(obs),
                        'Reward: {}'.format(reward),
                        '=' * 20
                    ]))
                    break;
                log.info('\n'.join([
                    'Taking action: {}'.format(action),
                    'Observation: {}'.format(obs),
                    'Reward: {}'.format(reward),
                    'New Belief: {}'.format(belief),
                    '=' * 20
                ]))

            #Tag problem ----------------------------------------------------------------
            # When the status is tagger, the robot s will catch robot t, the tag problem will end so it has to stop.
            if "Tag.POMDP" in environment:
                if params.benchmark == 0:
                    if "tagged" in model.curr_state:
                        log.info('\n'.join([
                            'Taking action: {}'.format(action),
                            'Observation: {}'.format(obs),
                            'Reward: {}'.format(reward),
                            '=' * 20
                        ]))
                        break;
                    log.info('\n'.join([
                        'Taking action: {}'.format(action),
                        'Observation: {}'.format(obs),
                        'Reward: {}'.format(reward),
                        'New state: {}'.format(new_state),
                        #'New Belief: {}'.format(belief),
                        '=' * 20
                    ]))

        # Printing the total steps and reward when the loop ends.
        if params.benchmark == 0:
            log.info('Simulation ended after {} steps. Total reward = {}'.format(i + 1, total_rewards))

        return pomdp

    def runBench(self, algo, T, **kwargs):
        visualiser = GraphViz(description='tmp')
        params, pomdp = self.params, None
        total_rewards, budget = 0, params.budget
        environment = params.env_config
        benchmark = params.benchmark


        log.info('~~~ Initialising simulation ~~~')
        with PomdpParser(params.env_config) as ctx:
            # creates model and solver
            model = self.create_model(ctx.copy_env())
            pomdp = self.create_solver(algo, model)

            # supply additional algo params
            belief = ctx.random_beliefs() if params.random_prior else ctx.generate_beliefs()

            if algo == 'pbvi':
                belief_points = ctx.generate_belief_points(kwargs['stepsize'])
                pomdp.add_configs(belief_points)
            elif algo == 'pomcp':
                pomdp.add_configs(budget, belief, **kwargs)

        # have fun!
        log.info('''
           ++++++++++++++++++++++
               Starting State:  {}
               Starting Budget:  {}
               Init Belief: {}
               Time Horizon: {}
               Max Play: {}
           ++++++++++++++++++++++'''.format(model.curr_state, budget, belief, T, params.max_play))

        for i in range(params.max_play):
            # plan, take action and receive environment feedbacks
            pomdp.solve(T)
            action = pomdp.get_action(belief)
            new_state, obs, reward, cost = pomdp.take_action(action)

            if params.snapshot and isinstance(pomdp, POMCP):
                # takes snapshot of belief tree before it gets updated
                self.snapshot_tree(visualiser, pomdp.tree, '{}.gv'.format(i))

            # update states
            belief = pomdp.update_belief(belief, action, obs)
            total_rewards += reward
            budget -= cost

            #Computing final results when a problem stops
            if "open" in action or "tagged" in model.curr_state or "adv" in action or "arrive" in action:
                log.info('Ended simulation after {} steps. Total reward = {}'.format(i + 1, total_rewards))
                self.step_list.append(i+1)
                self.fReward_list.append(total_rewards)
                self.steps += i+1
                self.fReward += total_rewards

                break;

            # Printing the details for every step of the interactive simulation
            # log.info('\n'.join([
            #   'Taking action: {}'.format(action),
            #   'Observation: {}'.format(obs),
            #   'Reward: {}'.format(reward),
            #   'Budget: {}'.format(budget),
            #   'New state: {}'.format(new_state),
            #   'New Belief: {}'.format(belief),
            #   '=' * 20
            # ]))

            if budget <= 0:
                log.info('Budget spent.')

        return pomdp