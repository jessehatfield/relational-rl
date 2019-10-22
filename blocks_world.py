import random
import time
from math import log10, floor

from game_model import GameModel, GameState, ActionProperties, GameAction, Game
from graph_model import GraphModel
from pg import ppo

import numpy as np
import tensorflow as tf


n_blocks = 4
max_moves = 20
block_names = [chr(i+65) for i in range(n_blocks)]


blocks_world = GameModel(
    global_enums={'filler': [0]},
    object_enums={'label': block_names},
    action_enums={'move': {'on_top_of': ActionProperties(n_nodes=2),
                           'to_table': ActionProperties(n_nodes=1)}},
    edge_enums={'position': ['above', 'below'], 'role': ['from', 'to']})


class BlocksWorldGame(Game):
    """Basic blocks world game"""

    def __init__(self):
        self._stacks = []
        shuffled = [name for name in block_names]
        random.shuffle(shuffled)
        for name in shuffled:
            maximum = len(self._stacks)
            i = random.randrange(maximum+1)
            if i == maximum:
                self._stacks.append([])
            self._stacks[i].append(name)
#        for name in shuffled:
#            self._stacks.append([name])
        self._goal = sorted(block_names)
        self._moves = 0
        self._solved = False
        self._timeout = False
        self._check()

    def _check(self):
        if len(self._stacks) == 1 and self._goal == self._stacks[0]:
            self._solved = True
        elif self._moves == max_moves:
            self._timeout = True
        return self._solved or self._timeout

    def get_state(self):
        nodes = []
        edges = []
        for stack in self._stacks:
            size = len(stack)
            for i in range(size-1):
                block_id = len(nodes)
                nodes.append({'label': stack[i]})
                edges.append((block_id, block_id+1, {'position': 'below'}))
                edges.append((block_id+1, block_id, {'position': 'above'}))
            nodes.append({'label': stack[size-1]})
        return GameState(blocks_world, {'filler': 0}, nodes, edges, terminal=self._solved or self._timeout)

    def get_actions(self):
        """Any block at the top of the stack may be moved on top of another stack, or to a new stack. Blocks on their
        own may be moved to the top of a stack."""
        if self._solved or self._timeout:
            return []
        else:
            actions = []
            top_block_ids = []
            available_block_ids = []
            n = 0
            for stack in self._stacks:
                n += len(stack)
                if len(stack) > 1:
                    top_block_ids.append(n-1)
                available_block_ids.append(n-1)
            for top_block in top_block_ids:
                actions.append(GameAction(blocks_world, 'move', 'to_table', [top_block]))
            for top_block in available_block_ids:
                for top_block_2 in available_block_ids:
                    if not top_block == top_block_2:
                        actions.append(GameAction(blocks_world, 'move', 'on_top_of',
                                                  [top_block, top_block_2],
                                                  [{'role': 'from'}, {'role': 'to'}]))
            return actions

    def act(self, action: GameAction):
        self._moves += 1
        move_from = action.node_ids[0]
        move_to = action.node_ids[1] if action.option == 'on_top_of' else None
        move_from_stack = -1
        move_to_stack = -1
        n = 0
        for stack_id in range(len(self._stacks)):
            n += len(self._stacks[stack_id])
            if n-1 == move_from:
                move_from_stack = stack_id
            elif move_to is not None and n-1 == move_to:
                move_to_stack = stack_id
        if move_from_stack == -1:
            raise Exception("Failed to find move-from ID -- state {}, action {}".format(self.get_state(), action))
        if move_to_stack == -1 and action.option == 'on_top_of':
                raise Exception("Failed to find move-to ID -- state {}, action {}".format(self.get_state(), action))
        block = self._stacks[move_from_stack].pop()
        if move_to_stack == -1:
            self._stacks.append([block])
        else:
            self._stacks[move_to_stack].append(block)
        if len(self._stacks[move_from_stack]) == 0:
            self._stacks.pop(move_from_stack)
        self._check()
        if self._solved:
            return 1.0
        elif self._timeout:
            return -0.1
        else:
            return 0.0

    @staticmethod
    def new_game():
        game = BlocksWorldGame()
        while game._solved:
            game = BlocksWorldGame()
        return game


max_timesteps = 64000
learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]
reg_params = [1e-3]
batch_size = 128
test_interval = 10


if __name__ == "__main__":

    def test(network, n_games=200, greedy=False):
        completed = 0
        total_moves = 0
        for i in range(n_games):
            game = BlocksWorldGame()
            n_moves = 0
            reward = 0
            while len(game.get_actions()) > 0:
                actions = game.get_actions()
                distribution = network.apply_policy(session, game.get_state(), actions)
                if greedy:
                    a_i = np.argmax(distribution)
                else:
                    a_i = np.random.choice(len(distribution), p=distribution)
                reward = game.act(actions[a_i])
                n_moves += 1
            if reward > 0:
                completed += 1
                total_moves += n_moves
        avg_moves = float('nan') if completed == 0 else total_moves / float(completed)
        strategy = "greedy" if greedy else "policy"
        print(f"Completion rate ({strategy}): {completed} / {n_games} "
              f"({100.0*completed/n_games}%), avg. moves: {avg_moves}")
        return float(completed) / n_games, avg_moves

    def test_both(network, n_games=200):
        p_greedy, moves_greedy = test(network, n_games, greedy=True)
        p_policy, moves_policy = test(network, n_games, greedy=False)
        return p_greedy, moves_greedy, p_policy, moves_policy

    def stop_early(history):
        threshold = 0.95
        n = 5
        for i in range(1, n+1):
            if len(history) < i:
                return False
            if history[-i][2] < threshold:
                return False
        return True

    for reg_param in reg_params:
        graph_model = GraphModel(blocks_world,
                                 object_embedding_dimensions={'label': 4},
                                 action_embedding_dimensions={'move': 4},
                                 edge_embedding_dimensions={'position': 4, 'role': 4},
                                 global_embedding_dimensions={'filler': 1},
                                 hidden_node_dimension=8,
                                 hidden_edge_dimension=8,
                                 hidden_global_dimension=4,
                                 state_dimension=2, global_output_size=1,
                                 node_output_size=0,
                                 hidden_node_output=[4])
        for learning_rate in learning_rates:
            print(f"Experiment: reg_param={reg_param}, learning_rate={learning_rate}")
            session = tf.Session()
            lr_magnitude = -int(floor(log10(learning_rate)))
            lr_multiplier = int(learning_rate * (10**lr_magnitude))
            rp_magnitude = -int(floor(log10(reg_param)))
            rp_multiplier = int(reg_param * (10**rp_magnitude))
            run_id = "ppo_{}__lr={}e-{}__reg={}e-{}".format(
                time.time(), lr_multiplier, lr_magnitude, rp_multiplier, rp_magnitude)
            print(f"\t{run_id}")
            try:
                learner, avg_reward = ppo(
                    session,
                    graph_model,
                    run_id,
                    BlocksWorldGame,
                    batch_size,
                    int(max_timesteps / batch_size),
                    policy_lr=learning_rate,
                    value_lr=0.001,
                    reg_param=reg_param,
                    callback_interval=test_interval,
                    callback=test_both,
                    early_stopping_condition=stop_early,
                    callback_vars=['p_finish_greedy', 'avg_steps_greedy', 'p_finish_policy', 'avg_steps_policy'],
                    gamma=0.99,
                    lam=0.95,
                    output_dir='bw-vpg-4'
                )
            except tf.errors.InvalidArgumentError as e:
                print(e)
            session.close()
            tf.reset_default_graph()
