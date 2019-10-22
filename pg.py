from graph_nets import graphs
from graph_nets import utils_np

from game_model import Game
from game_network import DenseGraphTransform, GameNetwork, get_tensors
from graph_model import GraphModel

import math
import os
from typing import Type

import numpy as np
import tensorflow as tf


def ppo(session: tf.Session,
        model: GraphModel,
        run_id: str,
        game_class: Type[Game],
        batch_size: int,
        n_batches: int,
        callback_interval=1,
        callback_vars=(),
        callback=None,
        early_stopping_condition=None,
        gamma=0.99,
        lam=0.95,
        policy_update_iters=100,
        policy_kl_target=0.01,
        value_update_iters=100,
        policy_lr=0.01,
        value_lr=0.01,
        reg_param=0.0001,
        epsilon=0.2,
        output_dir="pg-runs"):
    """Run policy optimization.

    params:
    model: model that maps the game to the network structure
    run_id: string to keep track of this run
    game_class: provides functions for instantiating games, state transition, etc
    max_timesteps: int, run for this many total in-game actions across all games
    batch_size: int, number of transitions to sample from memory and optimize over at each step
    memory_buffer: int, maximum size of memory to sample transitions from (forgets the oldest after this point)
    update_target_interval: int, copy learner's weights to the target network after this many timesteps
    pretrain_games: int, run this many games to fill memory before beginning training
    callback_interval: int, call callback function every n timesteps if callback is given
    callback_vars: list, gives the names and therefore number of scalars returned by callback function
    callback: arbitrary function that takes this network as a parameter, default None
    """
    training_network = GameNetwork('learner', model, reg_param)
    value_estimator = ValueEstimator(model, training_network, "value-estimator",
                                     value_update_iters, value_lr, gamma, lam)

    def choose(current_state, possible_actions):
        distribution = training_network.apply_policy(session, current_state, possible_actions)
        return np.random.choice(len(distribution), p=distribution)
    # Set up output and metrics
    out_dir = "{}/{}".format(output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    training_network.save_metadata(out_dir, model.game.object_vars.get_nominal()[0])
    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(out_dir, session.graph)
    sample_weights = tf.placeholder(tf.float32, shape=(None,), name="weights")
    logp_old_ph = tf.placeholder(tf.float32, shape=(None,), name="logp_old")
    ratio = tf.exp(training_network.log_p_chosen - logp_old_ph)
    clipped_ratio = tf.minimum(1+epsilon, tf.maximum(1-epsilon, ratio))
    ppo_loss = -tf.reduce_mean(tf.math.minimum(sample_weights * ratio, sample_weights * clipped_ratio))
    tf.summary.scalar('ppo_loss', ppo_loss)
    ppo_step_op = tf.train.AdamOptimizer(policy_lr).minimize(ppo_loss)
    approx_kl = tf.reduce_mean(logp_old_ph - training_network.log_p_chosen)
#    pg_loss = training_network.get_policy_gradient_loss(sample_weights)
#    pg_step_op = tf.train.AdamOptimizer(policy_lr).minimize(pg_loss)
    # Train for the specified amount of time
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    print(f"Training for {n_batches} total updates...")
    callback_history = []
    for epoch in range(n_batches):
        game: Game = game_class.new_game()
        # Collect on-policy data until we have a sufficient batch of examples
        states = []
        actions = []
        action_ids = []
        ind_rewards = []
        episodes = []
        episode_length = 0
        n_steps = 0
        while n_steps < batch_size:
            # Take an action, get a reward and new state, and add the transition to the current trajectory
            states.append(game.get_state())
            actions.append(game.get_actions())
            action_ids.append(choose(states[-1], actions[-1]))
            action = actions[-1][action_ids[-1]]
            ind_rewards.append(game.act(action))
            episode_length += 1
            # If the new state is terminal, record the begin/end indices and reset the game
            if game.get_state().is_terminal():
                n_previous = n_steps
                n_steps += episode_length
                episode_length = 0
                episodes.append((n_previous, n_steps))
                game = game_class.new_game()
        # Compute discounted state graph tuples for each trajectory
        state_graphs = [model.state_graph(states[i], actions[i]) for i in range(n_steps)]
        state_tuples = utils_np.data_dicts_to_graphs_tuple(state_graphs)
        n_objects = [states[i].n_objects for i in range(len(states))]
        # Compute value estimates and advantages based on current value estimator (critic)
        values, advantages = value_estimator.evaluate(session, state_tuples, ind_rewards, episodes)
        if math.isnan(values[0]):
            print("ERROR: NaN value estimates!")
            break
        # And get the old log probabilities of the chosen actions
        action_node_ids = [states[i].n_objects + action_ids[i] for i in range(len(states))]
        logp_old = session.run(training_network.log_p_chosen, {
            get_tensors(training_network.input_graphs): get_tensors(state_tuples),
            training_network.true_action: action_node_ids,
            training_network.n_objects: n_objects})
        # Update the policy wrt PPO loss, given previous probability distribution and estimated advantages
        summaries = tf.summary.merge_all(scope=training_network.scope)
        results = {}
        for i in range(policy_update_iters):
            results = session.run(
                {'pg_loss': ppo_loss,
                 'kl': approx_kl,
                 'policy': training_network.policy,
                 'p_chosen': training_network.p_chosen,
                 'logp_chosen': training_network.log_p_chosen,
                 'summaries': summaries,
                 'step_op': ppo_step_op},
                {get_tensors(training_network.input_graphs): get_tensors(state_tuples),
                 training_network.true_action: action_node_ids,
                 sample_weights: advantages,
                 logp_old_ph: logp_old,
                 training_network.n_objects: n_objects})
            if i == policy_update_iters-1 or i % 10 == 0 or results['kl'] > 1.5 * policy_kl_target:
                print(f"\tPPO loss[{i}]: {results['pg_loss']}, KL divergence: {results['kl']}")
#                      f"logp(chosen): {results['logp_chosen']}, logp(chosen|old): {logp_old}"
            if i == policy_update_iters-1 or results['kl'] > 1.5 * policy_kl_target:
                print(f"\tStopping policy update after {i} iterations, KL divergence={results['kl']}")
                break
        # And update the critic according to the discounted rewards
        rewards = value_estimator.discounted_returns(ind_rewards, episodes)
        value_estimator.optimize(session, state_tuples, rewards)
        # Record results for this iteration
        file_writer.add_summary(results['summaries'], epoch)
        saver.save(session, out_dir + "/model.ckpt", epoch)
        additional = ""
        avg_reward = sum(ind_rewards) / len(episodes)
        file_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="avg.reward", simple_value=avg_reward)]), epoch)
        if callback is not None and (epoch % callback_interval == 0 or epoch == n_batches-1):
            x = callback(training_network)
            callback_history.append(x)
            with tf.variable_scope("callback"):
                for i in range(len(callback_vars)):
                    name = callback_vars[i]
                    file_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=x[i])]), epoch)
            additional = ", callback={}".format(x)
        print(f"epoch {epoch}: loss={results['pg_loss']}, avg. reward={avg_reward}{additional}\n"
              f"\trewards={ind_rewards}\n\tadvantage={advantages}")
        early_stop = False if early_stopping_condition is None else early_stopping_condition(callback_history)
        if epoch == n_batches-1 or early_stop:
            print('Examples:')
            node_base = 0
            for i in range(len(actions)):
                n_objects = states[i].n_objects
                n_actions = len(actions[i])
                print(f'\t{states[i]}; {[str(opt) for opt in actions[i]]}:')
                print(f'\t\t\tp == {results["policy"][node_base+n_objects:node_base+n_objects+n_actions]}')
                print(f'\t\tactual choice == {action_ids[i]}')
                print(f'\t\tp_choice == {results["p_chosen"][i]}')
                print(f'\t\test. value == {values[i]}')
                print(f'\t\t--> reward == {ind_rewards[i]}; reward-to-go == {rewards[i]}; advantage == {advantages[i]}')
                node_base += n_objects + n_actions
            print(f'loss(example batch) == {results["pg_loss"]}')
        if early_stop:
            break
    return training_network, 0.0


class ValueEstimator:
    """Defines a value function approximator that can take as input the internal state representation
    from a graph network."""
    def __init__(self, model, training_network, scope, value_update_iters, lr, gamma=0.99, lam=0.95):
        self.training_network = training_network
        self.value_update_iters = value_update_iters
        self.gamma = gamma
        self.lam = lam
        with tf.variable_scope(scope):
            inner_graph: graphs.GraphsTuple = self.training_network.intermediate_graphs
            inner_graph_frozen: graphs.GraphsTuple = inner_graph.map(tf.stop_gradient)
            value_function = DenseGraphTransform(model.action_dimension, 1, 1,
                                                 global_activation=None,
                                                 name="value-function",
                                                 regularizer=training_network.regularizer)
            self.value_estimate = tf.squeeze(value_function(inner_graph_frozen).globals)
            self.value_target_ph = tf.placeholder(tf.float32, shape=(None,))
            self.value_loss = tf.losses.mean_squared_error(self.value_target_ph, self.value_estimate)
            self.step_op = tf.train.AdamOptimizer(lr).minimize(self.value_loss)
        self.latest_internal_state = []

    def discounted_returns(self, ind_rewards, episodes):
        """Compute discounted rewards across multiple trajectories.
        Args:
            ind_rewards: Flat list of individual rewards per timestep across all episodes in the batch
            episodes: List of tuples identifying (begin index, end index) for each episode
        Returns:
            A list the same length of ind_rewards where:
                rewards[t] == sum{l:0 ... # of steps remaining in t's episode}(gamma^l * ind_rewards[t+l])
        """
        rewards = []
        for begin, end in episodes:
            for i in range(begin, end):
                rewards.append(sum((ind_rewards[i+t]*(self.gamma**t) for t in range(end-i))))
        return rewards

    def evaluate(self, session, state_tuples, ind_rewards, episodes):
        values, internal_state = session.run(
            [self.value_estimate, self.training_network.intermediate_graphs],
            {self.training_network.input_graphs: state_tuples})
        if math.isnan(values[0]):
            print("ERROR: NaN value estimates!")
            print(f"\tinternal state = {internal_state[0]}")
            advantages = [v for v in values]
        else:
            advantages = []
            for begin, end in episodes:
                delta = [ind_rewards[i] + (self.gamma*values[i+1]) - values[i] for i in range(begin, end-1)]
                delta.append(ind_rewards[end-1] - values[end-1])
                for t in range(end-begin):
                    avg_advantage = 0.0
                    for l in range(end-begin-t):
                        avg_advantage += ((self.gamma * self.lam) ** l) * delta[t+l]
                    advantages.append(avg_advantage)
        return values, advantages

    def optimize(self, session, state_tuples, rewards):
        value_loss_current = 0.0
        for i in range(self.value_update_iters):
            value_loss_current, _ = session.run([self.value_loss, self.step_op], {
                self.training_network.input_graphs: state_tuples,
                self.value_target_ph: rewards
            })
            if i == self.value_update_iters-1 or i % 20 == 0:
                print(f"\tvalue loss[{i}]: {value_loss_current}")
        return value_loss_current
