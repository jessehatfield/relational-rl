import numpy as np

from game_model import GameModel, GameState, GameAction


class GraphModel(object):
    def __init__(self, game: GameModel,
                 object_embedding_dimensions,
                 action_embedding_dimensions,
                 edge_embedding_dimensions,
                 global_embedding_dimensions,
                 hidden_node_dimension=4,
                 hidden_edge_dimension=4,
                 hidden_global_dimension=4,
                 state_dimension=16, global_output_size=2, node_output_size=1, action_dimension=4,
                 output_value=True,
                 hidden_node_output=[]):
        self.game = game
        self.object_embedding_dimensions = object_embedding_dimensions
        self.action_embedding_dimensions = action_embedding_dimensions
        self.edge_embedding_dimensions = edge_embedding_dimensions
        self.global_embedding_dimensions = global_embedding_dimensions
        self.state_dimension = state_dimension
        self.global_output_size = global_output_size
        self.node_output_size = node_output_size
        self.action_dimension = action_dimension
        self.output_value = output_value
        self.n_hidden_node_output = len(hidden_node_output)
        self.hidden_node_output = hidden_node_output
        self.hidden_node_dimension = hidden_node_dimension
        self.hidden_edge_dimension = hidden_edge_dimension
        self.hidden_global_dimension = hidden_global_dimension

    def get_object_input_size(self):
        return len(self.game.node_vars.get_nominal())

    def get_action_input_size(self):
        return 1

    def get_node_input_size(self):
        return self.get_object_input_size() + self.get_action_input_size()

    def state_graph(self, state: GameState, actions=[]):
        """Build a graph whose nodes represent game objects, and optionally legal actions connected
        to the objects they interact with."""
        nodes = []
        edges = []
        senders = []
        receivers = []
        global_vars = []
        var_names = self.game.object_vars.get_nominal()
        edge_var_names = self.game.edge_vars.get_nominal()
        global_var_names = self.game.global_vars.get_nominal()
        object_vector_length = len(var_names)
        action_vector_length = 1
        edge_vector_length = len(edge_var_names)
        for i in range(state.n_objects):
            ids = []
            for var_name in var_names:
                ids.append(state.get_node_value_id(i, var_name))
            # Pad with 0s for the portion of the vector relating to action properties
            for j in range(action_vector_length):
                ids.append(0)
            node_arr = np.array(ids, dtype=np.float32)
            nodes.append(node_arr)
        for i in range(state.n_edges):
            edge_arr = []
            sender, receiver, props = state.get_edge(i)
            for var_name in edge_var_names:
                edge_arr.append(state.get_edge_value_id(i, var_name, 0.0))
            edges.append(edge_arr)
            senders.append(sender)
            receivers.append(receiver)
        # Add extra nodes for any actions and extra edges if they relate to objects
        for i in range(len(actions)):
            action = actions[i]
            action_definition = [action.get_combination_id()]
            action_node_arr = np.zeros(object_vector_length + action_vector_length, dtype=np.float32)
            nodes.append(action_node_arr)
            for j in range(action_vector_length):
                action_node_arr[object_vector_length+j] = action_definition[j]
            for j in range(action.n_node_objects):
                action_edge_arr = []
                for var_name in edge_var_names:
                    action_edge_arr.append(action.get_target_var_id(j, var_name, 0.0))
                edges.append(action_edge_arr)
                receivers.append(action.get_target_node(j))
                senders.append(i+state.n_objects)  # actions are appended to object nodes: action[0] == node[n objects]
        # Add globals
        for var_name in global_var_names:
            global_vars.append(state.get_global_value_id(var_name))
        g = {
            "nodes": np.array(nodes, dtype=np.float32).reshape((-1, object_vector_length + action_vector_length)),
            "edges": np.array(edges, dtype=np.float32) .reshape((-1, edge_vector_length)),
            "receivers": np.array(receivers, dtype=np.float32).reshape((-1,)),
            "senders": np.array(senders, dtype=np.float32).reshape((-1,)),
            "globals": np.array(global_vars, dtype=np.float32).reshape((-1)),
            "n_node": len(nodes),
            "n_edge": len(edges)
        }
        return g

    def action_graph(self, state: GameState, action: GameAction):
        """Build a graph representing an action: mirrors the state structure except that node values refer to their
        role in the action rather than the current state."""
        node_activation = np.zeros((state.n_objects, 0), dtype=np.float32)
        for node_id in action.nodes:
            node_activation[node_id, 0] = 1.0
        g = {
            "nodes": node_activation,
            "n_node": len(node_activation),
            "n_edge": 0,
        }
        return g

    def placeholder_graph(self, with_action=True):
        """Build a placeholder graph whose actual node values are arbitrary"""
        if with_action:
            return self.state_graph(self.game.build_default_state(), actions=[self.game.build_default_action([0])])
        else:
            return self.state_graph(self.game.build_default_state())

    def placeholder_action_graph(self):
        """Build a placeholder action graph whose actual node values are arbitrary"""
        return self.action_graph(self.game.build_default_state(), self.game.build_default_action())

    def global_state_vector(self, state: GameState):
        """"Get a global state vector from the game state object"""
        all_one_hot = np.zeros(self.get_global_input_size(), dtype=np.float32)
        i = 0
        for var_name in self.game.global_vars.get_nominal():
            k = self.game.global_vars.get_num_values(var_name)
            all_one_hot[i + state.get_global_value_id(var_name) - 1] = 1.0
            i += k
        return all_one_hot

    def global_action_vector(self, action: GameAction):
        """"Get a vector describing the overall choice of action"""
        one_hot = np.zeros(self.game.actions.n_combinations, dtype=np.float32)
        i = self.game.actions.get_combination_id(action.action_type, action.option)
        one_hot[i] = 1.0
        return one_hot

    def get_global_input_size(self):
        """"Get the length of the complete global state vector."""
        n = 0
        for var_name in self.game.global_vars.get_nominal():
            n += self.game.global_vars.get_num_values(var_name)
        return n

    def get_global_action_size(self):
        """"Get the length of the complete global action vector."""
        return self.game.actions.n_combinations

    def has_global_input(self):
        return self.get_global_input_size() > 0

    def get_global_output_size(self):
        """Get the length of the global output vector."""
        return self.global_output_size

    def sample_state(self, np_random):
        g = self.game.sample_state(np_random)
        return g

    def prepare_data(self, data, start_index, n_samples):
        batched_data = data[start_index:start_index+n_samples]
        batched_states = [x[0] for x in batched_data]
        batched_actions = [x[1] for x in batched_data]
        batched_targets = [x[2] for x in batched_data]
        batched_graphs = [self.state_graph(batched_states[i], batched_actions[i]) for i in range(len(batched_data))]
        batched_globals = [self.global_state_vector(state) for state in batched_states]
        return batched_graphs, batched_globals, batched_targets
