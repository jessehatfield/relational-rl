from abc import ABC, abstractmethod
from typing import List


class VarModel(object):
    def __init__(self, enums):
        self._nominal_names = sorted([var_name for var_name in enums.keys()])
        self._nominal_values = {var_name: enums[var_name] for var_name in self._nominal_names}
        self._nominal_indices = {}
        for var_name in self._nominal_names:
            self._nominal_indices[var_name] = {}
            values = self._nominal_values[var_name]
            for i in range(len(values)):
                self._nominal_indices[var_name][values[i]] = i+1

    def get_nominal(self):
        return [x for x in self._nominal_names]

    def get_nominal_values(self, name):
        if name not in self._nominal_values:
            raise ValueError(f"Levels not found for variable'{name}'")
        values = self._nominal_values[name]
        return [x for x in values]

    def get_num_values(self, name):
        if name not in self._nominal_values:
            raise ValueError(f"Levels not found for variable'{name}'")
        return len(self._nominal_values[name])

    def get_nominal_id(self, var, value, default=None):
        if value is None and default is not None:
            return default
        if var not in self._nominal_indices:
            raise ValueError(f"Value index not found for variable'{var}'")
        index = self._nominal_indices[var]
        if value not in index:
            keys = [key for key in index]
            raise ValueError(f"Value number not found for value {var}['{value}'] (known options: {keys})")
        return index[value]

    def get_default_values(self):
        values = {name: self._nominal_values[name][0] for name in self._nominal_names}
        return values

    def sample_values(self, random):
        values = {}
        for name in self._nominal_names:
            i = random.randint(self.get_num_values(name))
            values[name] = self._nominal_values[name][i]
        return values


class ActionProperties(object):
    def __init__(self, n_nodes=0, n_edges=0, ordered_nodes=True, ordered_edges=True, edges={}):
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.ordered_nodes = ordered_nodes
        self.ordered_edges = ordered_edges
        self.edges = edges


class ActionModel(object):
    def __init__(self, actions):
        self._action_types = sorted([var_name for var_name in actions.keys()])
        self._options = {a_type: sorted([opt for opt in actions[a_type].keys()]) for a_type in self._action_types}
        self.actions = actions
        self._action_indices = {}
        self._flattened_indices = {}
        self._flattened_options = []
        for action_name in self._action_types:
            self._action_indices[action_name] = {}
            options = self._options[action_name]
            for i in range(len(options)):
                self._action_indices[action_name][options[i]] = i+1
                self._flattened_indices[(action_name, options[i])] = len(self._flattened_options)+1
                self._flattened_options.append((action_name, options[i]))
        self.n_combinations = len(self._flattened_indices)

    def get_action_types(self):
        return [x for x in self._action_types]

    def get_options(self, action_type):
        return [x for x in self._options[action_type]]

    def get_properties(self, action_type, option):
        return self.actions[action_type][option]

    def get_option_id(self, action_type, option):
        if action_type not in self._action_indices:
            raise ValueError(f"Value index not found for action type '{action_type}'")
        index = self._action_indices[action_type]
        if option not in index:
            keys = [key for key in index]
            raise ValueError(f"Value number not found for option {action_type}['{option}'] (known options: {keys})")
        return index[option]

    def get_combination_id(self, action_type, option):
        if (action_type, option) not in self._flattened_indices:
            keys = [key for key in self._flattened_indices]
            raise ValueError(f"Index not found for combination <{action_type}, {option}> (known action pairs: {keys})")
        return self._flattened_indices[(action_type, option)]


class GameModel(object):
    def __init__(self, global_enums={}, object_enums={}, edge_enums={}, action_enums={}):
        self.global_vars = VarModel(global_enums)
        self.object_vars = VarModel(object_enums)
        self.edge_vars = VarModel(edge_enums)
        self.action_vars = VarModel({k: [j for j in action_enums[k]] for k in action_enums})
        self.actions = ActionModel(action_enums)

    def build_default_state(self, n=1):
        global_var_values = self.global_vars.get_default_values()
        nodes = []
        for i in range(n):
            node_var_values = self.object_vars.get_default_values()
            nodes.append(node_var_values)
        return GameState(self, global_var_values, nodes)

    def build_default_action(self, nodes=[]):
        action_type = self.actions.get_action_types()[0]
        option = self.actions.get_options(action_type)[0]
        return GameAction(self, action_type, option, nodes)

    def sample_state(self, random, l=10):
        global_var_values = self.global_vars.sample_values(random)
        n = random.poisson(l, 1)[0]
        nodes = []
        for i in range(n):
            node_var_values = self.object_vars.sample_values(random)
            nodes.append(node_var_values)
        return GameState(self, global_var_values, nodes)


class GameState(object):
    def __init__(self, game: GameModel, global_var_values, nodes, edges=[], terminal=False):
        self.game = game
        self._global_var_values = global_var_values
        self._nodes = nodes
        self._edges = edges
        self.n_objects = len(nodes)
        self.n_edges = len(edges)
        self._terminal = terminal
        self._edges_from = {}
        self._edges_to = {}
        self._edges_from_to = {}
        for source, dest, props in edges:
            if source not in self._edges_from.keys():
                self._edges_from[source] = []
            if dest not in self._edges_to.keys():
                self._edges_to[dest] = []
            if (source, dest) not in self._edges_from_to.keys():
                self._edges_from_to[(source, dest)] = []
            self._edges_from[source].append((dest, props))
            self._edges_to[dest].append((source, props))
            self._edges_from_to[(source, dest)].append(props)

    def get_global_value(self, var, optional=False):
        if not optional and var not in self._global_var_values:
            raise ValueError("No value found for variable '{}'".format(var))
        return self._global_var_values.get(var, None)

    def get_global_value_id(self, var, default=None):
        value = self.get_global_value(var, default is not None)
        return self.game.global_vars.get_nominal_id(var, value, default)

    def get_node_value(self, i, var, optional=False):
        if i >= self.n_objects:
            raise IndexError(f"Object {i} requested, but only {self.n_objects} objects known.")
        node = self._nodes[i]
        if not optional and var not in node:
            var_names = node.keys()
            raise KeyError(f"Object {i} has no value for key '{var}' (known variables: {var_names})")
        return node.get(var, None)

    def get_node_value_id(self, i, var, default=None):
        value = self.get_node_value(i, var, default is not None)
        return self.game.object_vars.get_nominal_id(var, value, default)

    def get_edge_value(self, i, var, optional=False):
        if i >= self.n_edges:
            raise IndexError(f"Edge {i} requested, but only {self.n_edges} edges known.")
        src, dst, edge_props = self._edges[i]
        if not optional and var not in edge_props:
            var_names = edge_props.keys()
            raise KeyError(f"Edge {i} (from {src} to {dst}) has no value for key '{var}'"
                           f"(known variables: {var_names})")
        return edge_props.get(var, None)

    def get_edge_value_id(self, i, var, default=None):
        value = self.get_edge_value(i, var, default is not None)
        return self.game.edge_vars.get_nominal_id(var, value, default)

    def get_edge(self, i):
        if i >= self.n_edges:
            raise IndexError(f"Edge {i} requested, but only {self.n_edges} edges known.")
        src, dst, edge_props = self._edges[i]
        return src, dst, edge_props

    def is_terminal(self):
        return self._terminal

    def __str__(self):
        global_str = str(self._global_var_values)
        nodes_str = str(self._nodes)
        if len(self._edges) == 0:
            return "GameState<global={}, nodes=[{}]{}>".format(global_str, nodes_str,
                                                               " (terminal)" if self._terminal else "")
        edges_str = str(self._edges)
        return "GameState<global={}, nodes=[{}], edges=[{}]{}>".format(global_str, nodes_str, edges_str,
                                                                       " (terminal)" if self._terminal else "")


class GameAction(object):
    def __init__(self, game: GameModel, action_type, option, node_ids=[], edge_props=[]):
        self.game = game
        self.action_type = action_type
        self.option = option
        self.node_ids = node_ids
        self.n_node_objects = len(node_ids)
        if len(edge_props) == 0 and len(node_ids) > 0:
            self.edge_props = [{} for _ in node_ids]
        else:
            self.edge_props = edge_props

    def get_option_id(self):
        return self.game.actions.get_option_id(self.action_type, self.option)

    def get_combination_id(self):
        return self.game.actions.get_combination_id(self.action_type, self.option)

    def get_target_node(self, i):
        if i >= len(self.node_ids):
            raise IndexError(f"Target {i} requested, but only {len(self.node_ids)} target objects for this action.")
        return self.node_ids[i]

    def get_target_var(self, i, var, optional=False):
        if i >= len(self.node_ids):
            raise IndexError(f"Target {i} requested, but only {len(self.node_ids)} target objects for this action.")
        props = self.edge_props[i]
        if not optional and var not in props:
            var_names = props.keys()
            raise KeyError(f"Action-edge {i} has no value for key '{var}' (known variables: {var_names})")
        return props.get(var, None)

    def get_target_var_id(self, i, var, default=None):
        value = self.get_target_var(i, var, default is not None)
        return self.game.edge_vars.get_nominal_id(var, value, default)

    def __str__(self):
        return "GameAction<{}: {} ({})>".format(self.action_type, self.option, self.node_ids)


class Game(ABC):
    @staticmethod
    @abstractmethod
    def new_game():
        """Create a new instance of the game"""
        pass

    @abstractmethod
    def get_state(self) -> GameState:
        """Get the current GameState"""
        pass

    @abstractmethod
    def get_actions(self) -> List[GameAction]:
        """Get the list of currently legal GameAction objects, or an empty list if in a terminal state"""
        pass

    @abstractmethod
    def act(self, action: GameAction) -> float:
        """Process an action, advance the game state accordingly, and return a numeric reward (can be 0)"""
        pass


if __name__ == "__main__":
    mtg = GameModel(global_enums={'phase': ['UNTAP', 'UPKEEP', 'DRAW', 'MAIN1'], 'turn_controller': ['player', 'opponent']},
                    object_enums={'card': ['Island', 'Forest', 'Mountain', 'Plains', 'Swamp'],
                                'zone': ['player_hand', 'player_battlefield', 'opponent_battlefield']},
                    action_enums={'mulligan': {'keep': ActionProperties(), 'mulligan': ActionProperties()},
                                  'priority': {'pass': ActionProperties(), 'play': ActionProperties(n_nodes=1)}})
    print(mtg)
    print(mtg.global_vars._nominal_names)
    print(mtg.global_vars._nominal_values)
    print(mtg.global_vars._nominal_indices)
    print(mtg.object_vars._nominal_names)
    print(mtg.object_vars._nominal_values)
    print(mtg.object_vars._nominal_indices)
    print(mtg.action_vars._nominal_names)
    print(mtg.action_vars._nominal_values)
    print(mtg.action_vars._nominal_indices)
    print()
    state = GameState(mtg, {'phase': 'MAIN1', 'turn_controller': 'player'},
                      [{'card': 'Mountain', 'zone': 'player_battlefield'},
                       {'card': 'Plains', 'zone': 'opponent_battlefield'}])
    print(state)
    print(state._global_var_values)
    print(state.get_global_value('phase'))
    print(state.get_global_value_id('phase'))
    print(state.get_global_value('turn_controller'))
    print(state.get_global_value_id('turn_controller'))
    print(state._nodes)
    print(state.get_node_value(0, 'card'))
    print(state.get_node_value_id(0, 'card'))
    print(state.get_node_value(1, 'card'))
    print(state.get_node_value_id(1, 'card'))
    action = GameAction(mtg, 'priority', 'play', [0])
    print(action)
    print()

    state = mtg.build_default_state(3)
    print(state)
    print(state._global_var_values)
    print(state._nodes)
    print()
