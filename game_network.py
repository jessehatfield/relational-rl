from graph_nets import blocks
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

from graph_model import GraphModel

from typing import Any

import sonnet as snt
import tensorflow as tf


class GraphEncoder(snt.AbstractModule):

    def __init__(self, model, regularizers=None, name=None):
        super(GraphEncoder, self).__init__(name=name)
        self.model = model
        self.regularizers = {} if regularizers is None else regularizers

        with self._enter_variable_scope():
            # Separately embed different categorical variables as dense vectors
            self.object_embedding_layers = self._build_embedding_layers(
                self.model.game.object_vars, self.model.object_embedding_dimensions, "objects")
            self.action_embedding_layers = self._build_embedding_layers(
                self.model.game.action_vars, self.model.action_embedding_dimensions, "actions")
            self.edge_embedding_layers = self._build_embedding_layers(
                self.model.game.edge_vars, self.model.edge_embedding_dimensions, "edges")
            self.global_embedding_layers = self._build_embedding_layers(
                self.model.game.global_vars, self.model.global_embedding_dimensions, "globals")

    def _build_embedding_layers(self, var_model, dimension_map, prefix):
        embedding_layers = []
        for var_name in var_model.get_nominal():
            embedding = snt.Embed(vocab_size=var_model.get_num_values(var_name)+1,
                                  embed_dim=dimension_map[var_name],
                                  regularizers=self.regularizers,
                                  # TODO: densify_gradients?
                                  name=f"{prefix}-{var_name}")
            embedding_layers.append(embedding)
        return embedding_layers

    def _get_edge_encoder(self):
        edge_vars = self.model.game.edge_vars.get_nominal()
        indices = []
        for i in range(len(edge_vars)):
            var_index = snt.SliceByDim(dims=[1], begin=[i], size=[1], name=f"indices-{edge_vars[i]}")
            indices.append(var_index)

        def edge_encoder(edge_input):
            state_layers = []
            n_local_features = 0
            for j in range(len(edge_vars)):
                embedding_layer = self.edge_embedding_layers[j]
                var_indices = indices[j](edge_input)
                var_state = tf.squeeze(embedding_layer(tf.to_int32(var_indices)), axis=[1],
                                       name=f"squeeze_embedding-{edge_vars[j]}")
                n_local_features += embedding_layer.embed_dim
                state_layers.append(var_state)
                tf.summary.histogram("indices-" + embedding_layer.module_name, var_indices)
                self._summarize_embedding(embedding_layer)
            return tf.concat(state_layers, 1, name="embed_edges")

        return edge_encoder

    def _get_global_encoder(self):
        global_vars = self.model.game.global_vars.get_nominal()
        indices = []
        for i in range(len(global_vars)):
            var_index = snt.SliceByDim(dims=[1], begin=[i], size=[1], name=f"indices-{global_vars[i]}")
            indices.append(var_index)

        def global_encoder(global_input):
            state_layers = []
            n_local_features = 0
            for j in range(len(global_vars)):
                embedding_layer = self.global_embedding_layers[j]
                var_indices = indices[j](global_input)
                var_state = tf.squeeze(embedding_layer(tf.to_int32(var_indices)), axis=[1],
                                       name=f"squeeze_embedding-{global_vars[j]}")
                n_local_features += embedding_layer.embed_dim
                state_layers.append(var_state)
                tf.summary.histogram("indices-" + embedding_layer.module_name, var_indices)
                self._summarize_embedding(embedding_layer)
            return tf.concat(state_layers, 1, name="embed_globals")

        return global_encoder

    def _get_node_encoder(self):
        # Embed both kinds of nodes: environment objects and legal actions
        object_vars = self.model.game.object_vars.get_nominal()
        action_vars = self.model.game.action_vars.get_nominal()
        object_indices = []
        action_indices = []
        for i in range(len(object_vars)):
            var_index = snt.SliceByDim(dims=[1], begin=[i], size=[1], name=f"indices-{object_vars[i]}")
            object_indices.append(var_index)
        for i in range(len(action_vars)):
            var_index = snt.SliceByDim(dims=[1], begin=[i+len(object_vars)], size=[1], name=f"indices-{action_vars[i]}")
            action_indices.append(var_index)

        def node_encoder(node_input):
            state_layers = []
            for j in range(len(object_vars)):
                embedding_layer = self.object_embedding_layers[j]
                var_indices = object_indices[j](node_input)
                var_state = tf.squeeze(embedding_layer(tf.to_int32(var_indices)), axis=[1],
                                       name=f"squeeze_embedding-{object_vars[j]}")
                state_layers.append(var_state)
                tf.summary.histogram("indices-" + embedding_layer.module_name, var_indices)
                self._summarize_embedding(embedding_layer)
            for j in range(len(action_vars)):
                embedding_layer = self.action_embedding_layers[j]
                var_indices = action_indices[j](node_input)
                var_state = tf.squeeze(embedding_layer(tf.to_int32(var_indices)), axis=[1],
                                       name=f"squeeze_embedding-{action_vars[j]}")
                state_layers.append(var_state)
                tf.summary.histogram("indices-" + embedding_layer.module_name, var_indices)
                self._summarize_embedding(embedding_layer)
            result = tf.concat(state_layers, 1, name="embed_nodes")
            return result
        return node_encoder

    def _build(self, graphs):
        with self._enter_variable_scope():
            # Convert categorical variables to dense embedding vectors, within each node and edge,
            # without sending any messages
            self.independent_encoder = modules.GraphIndependent(edge_model_fn=self._get_edge_encoder,
                                                                node_model_fn=self._get_node_encoder,
                                                                global_model_fn=self._get_global_encoder,
                                                                name="graph_component_encoder")
        return self.independent_encoder(graphs)

    @staticmethod
    def _summarize_embedding(embedding_layer):
        name = f"embeddings-{embedding_layer.module_name}"
        tf.summary.histogram(name, embedding_layer.embeddings)
        tensor_shape = [1, embedding_layer.vocab_size, embedding_layer.embed_dim, 1]
        embedding_tensor = tf.reshape(embedding_layer.embeddings, tensor_shape)
        tf.summary.image(name, embedding_tensor)


class DenseGraphTransform(snt.AbstractModule):
    """
    Applies a single dense linear layer each to update all components of the graph: First, update
    each edge using a dense layer taking the edge value and two incident nodes' values as input,
    then update each node using a dense layer taking in its aggregated incident edges as input,
    then update the globals using a dense layer taking the aggregated edges and aggregated nodes as
    input. As a result, each node can incorporate information from immediate neighbors only -- if
    more hops are needed, chain multiple modules together.
    """
    def __init__(self, edge_dimension: int, node_dimension: int, global_dimension: int,
                 edge_activation: Any=tf.nn.leaky_relu,
                 node_activation: Any=tf.nn.leaky_relu,
                 global_activation: Any=tf.nn.leaky_relu,
                 reducer=blocks.unsorted_segment_max_or_zero,
                 regularizer=None, name=None):
        super(DenseGraphTransform, self).__init__(name=name)
        self.reg_linear = {"w": regularizer, "b": regularizer} if regularizer is not None else {}
        self.edge_dimension = edge_dimension
        self.node_dimension = node_dimension
        self.global_dimension = global_dimension
        self.edge_activation = edge_activation
        self.node_activation = node_activation
        self.global_activation = global_activation
        self.reducer = reducer

    def _build(self, graphs):
        with self._enter_variable_scope():
            self.graph_transformer = modules.GraphNetwork(
                edge_model_fn=self._build_linear(self.edge_dimension, self.edge_activation, "edge-dense"),
                node_model_fn=self._build_linear(self.node_dimension, self.node_activation, "node-dense"),
                global_model_fn=self._build_linear(self.global_dimension, self.global_activation, "global-dense"),
                node_block_opt={"use_sent_edges": True},
                reducer=self.reducer,
                name="graph_transformer")
        return self.graph_transformer(graphs)

    def _build_linear(self, dimension, activation, name):
        def get_module():
            dense = snt.Linear(dimension, name=name, regularizers=self.reg_linear)

            def apply_linear(inputs):
                transformed = dense(inputs)
                tf.summary.histogram("weights", dense.w)
                tf.summary.histogram("biases", dense.b)
                if activation is None:
                    activated = transformed
                else:
                    activated = activation(transformed)
                tf.summary.histogram("activations", activated)
                return activated
            return apply_linear
        return get_module


class GameNetwork(object):

    def __init__(self, scope: str, model: GraphModel, reg_param):
        # Process parameters
        self.scope = scope
        self.model = model
        self.n_out = self.model.get_global_output_size()

        # Configure regularization
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=reg_param)
        self.reg_linear = {"w": self.regularizer, "b": self.regularizer}
#        self.reg_embed = {}
        self.reg_embed = {"embeddings": self.regularizer}

        # Set up input tensors
        with tf.variable_scope(self.scope + "/state_input"):
            self.input_graphs = utils_tf.placeholders_from_data_dicts([self.model.placeholder_graph()],
                                                                      force_dynamic_num_graphs=True,
                                                                      name="local_state")

        with tf.variable_scope(self.scope + "/ground_truth"):
            # Reinforcement learning inputs
            self.true_action = tf.placeholder(tf.int32, shape=(None,), name="action")
            self.n_objects = tf.placeholder(tf.int32, shape=(None,), name="n_objects")
            self.target_q = tf.placeholder(tf.float32, shape=(None,), name="target_q")
            self.target_value = tf.placeholder(tf.float32, shape=(None,), name="target_value")

        with tf.variable_scope(self.scope):
            # Separately embed different categorical variables as dense vectors
            self.encoder_module = GraphEncoder(model, name="encoder", regularizers=self.reg_embed)
            self.embedded_graphs = self.encoder_module(self.input_graphs)

            # Apply an intermediate  transformation to pass information between neighboring nodes
            self.intermediate_graphs = DenseGraphTransform(model.hidden_edge_dimension,
                                                           model.hidden_node_dimension,
                                                           model.hidden_global_dimension,
                                                           name="intermediate",
                                                           regularizer=self.regularizer)(self.embedded_graphs)

            # Then apply a final transformation to produce a global output and node-level evaluations
            self.output_graphs = DenseGraphTransform(model.action_dimension,
                                                     1,
                                                     1,
                                                     node_activation=None,
                                                     global_activation=None,
                                                     name="output",
                                                     regularizer=self.regularizer)(self.intermediate_graphs)

        with tf.variable_scope(self.scope + "/outputs"):
            # If given a true action, get the corresponding output
            self.graph_indices = tf.math.cumsum(self.output_graphs.n_node, exclusive=True, name="starting_node_index")
            self.true_indices = self.graph_indices + self.true_action
            self.chosen_node_outputs = tf.reshape(tf.gather(
                self.output_graphs.nodes, self.true_indices, name="chosen_action_outputs"), [-1])

            # In case we need a policy output, build the following tensors:
            # 1) a learned stochastic policy for all possible actions,
            # 2) the individual probability of the chosen action
            # 3) the log of that individual probability."""
            # First, get each node's index
            node_indices = tf.range(tf.shape(self.output_graphs.nodes)[0])
            # Then, get the index of each graphs' first action
            first_action_indices = self.graph_indices + self.n_objects
            # broadcast action indices to nodes and compare to node indices
            first_action_broadcast = blocks.broadcast_globals_to_nodes(self.output_graphs.replace(
                globals=tf.reshape(first_action_indices, [-1, 1])))
            action_mask = tf.greater_equal(node_indices, tf.reshape(first_action_broadcast, [-1]))
            # Zero out the objects and apply softmax to the actions (treat action-nodes as logits)
            exp_or_zero = self.output_graphs.replace(
                nodes=tf.where(action_mask,
                               tf.math.exp(self.output_graphs.nodes),
                               tf.zeros_like(self.output_graphs.nodes)))
            # Sum the node values so that the global for each graph is the softmax denominator
            sum_nodes = blocks.GlobalBlock(lambda: tf.identity, use_edges=False, use_globals=False)
            softmax_graph = sum_nodes(exp_or_zero)

            # Then divide each node's value by that denominator, or set to 1 where denominator is 0
            def node_value_to_prob(node_inputs):
                p = tf.div_no_nan(node_inputs[:, 0], node_inputs[:, 1])
                return tf.where(p > 0, p, tf.ones_like(p))
            policy_graph = blocks.NodeBlock(lambda: node_value_to_prob,
                                            use_received_edges=False,
                                            use_sent_edges=False)(softmax_graph)
            self.policy = policy_graph.nodes
            self.p_chosen = tf.gather(self.policy, self.true_indices, name="p_true_action")
            self.log_p_chosen = tf.log(self.p_chosen, name="logp_true_action")

        # Configure metrics for training and display
        self.TRAIN_METRIC_OPS = self.scope + "/TRAIN_METRIC_OPS"
        self.VAL_METRIC_OPS = self.scope + "/VAL_METRIC_OPS"
        self.reg_term = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar('reg_loss', self.reg_term)

    def get_q_loss(self, target_q):
        with tf.variable_scope(self.scope + "/ql_metrics"):
            # Q-learning loss
            q_loss = tf.losses.mean_squared_error(target_q, self.chosen_node_outputs, scope="q_loss")
            q_loss_reg = q_loss + self.reg_term
            tf.summary.scalar('q_loss', q_loss)
            tf.summary.scalar('q_loss_reg', q_loss_reg)
        return q_loss

    def get_policy_gradient_loss(self, weights):
        """Build a tensor representing the loss term for policy gradient, given a tensor
        representing the weight of each data point (e.g. reward, advantage, reward-to-go,etc)."""
        with tf.variable_scope(self.scope + "/pg_metrics"):
            pg_loss = -tf.reduce_sum(self.log_p_chosen * weights)
            pg_loss_reg = pg_loss + self.reg_term
            tf.summary.scalar('pg_loss', pg_loss)
            tf.summary.scalar('pg_loss_reg', pg_loss_reg)
        return pg_loss

    def get_value_loss(self, predicted_value):
        """Build a tensor representing the loss term associated with a global state value"""
        with tf.variable_scope("value_metrics"):
            value_loss = tf.losses.mean_squared_error(self.target_value, tf.reshape(predicted_value, [-1]))
            tf.summary.scalar('value_loss', value_loss)
        return value_loss

    def compute_q_batch(self, session, states, actions):
        """Compute q-values for a batch of states and matching lists of legal actions."""
        state_graphs = [self.model.state_graph(states[i], actions[i]) for i in range(len(states))]
        state_tuples = utils_np.data_dicts_to_graphs_tuple(state_graphs)
        batch_dict = {get_tensors(self.input_graphs): get_tensors(state_tuples)}
        node_outputs = session.run(self.output_graphs.nodes, feed_dict=batch_dict).reshape((-1,))
        # node outputs will all be concatenated together, so we need to build a q-vector for each state
        q_vectors = []
        state_start = 0
        for i in range(len(states)):
            n_nodes = state_graphs[i]['n_node']
            n_objects = n_nodes - len(actions[i])
            q_vectors.append(node_outputs[state_start+n_objects:state_start+n_nodes])
            state_start += n_nodes
        # For each input state, skip the outputs corresponding to objects, returning only action q values
        return q_vectors

    def compute_q(self, session, state, actions):
        """Compute q for a single state and list of actions, as well as the best action.

        Uses the direct output values of the action nodes, as well as the index and action which
        maximize that value. Does not convert outputs to probabilities."""
        q = self.compute_q_batch(session, [state], [actions])[0]
        best_q = 0
        best_index = 0
        for i in range(len(q)):
            if i == 0 or q[i] > best_q:
                best_q = q[i]
                best_index = i
        return q, best_index, actions[best_index]

    def apply_policy(self, session, state, actions):
        """Get a stochastic policy output derived from node-level outputs. Applies softmax to
        node-level outputs representing actions, or a uniform distribution if values are out
        of bounds. Returns the probability distribution."""
        batched_graphs, batched_globals, batched_targets = self.model.prepare_data([(state, actions, None)], 0, 1)
        batched_tuples = utils_np.data_dicts_to_graphs_tuple(batched_graphs)
        batch_dict = {get_tensors(self.input_graphs): get_tensors(batched_tuples),
                      self.n_objects: [state.n_objects]}
        values = session.run({
            "policy": self.policy,
            "logits": self.output_graphs.nodes
        }, feed_dict=batch_dict)
        nodes = values["policy"].reshape((-1,))
        distribution = nodes[state.n_objects:]
        if sum(distribution) > 1.01 or sum(distribution) < 0.99:
            distribution = [1.0 / len(actions)] * len(actions)
        return distribution

    @staticmethod
    def load_model(session, save_dir):
        saver = tf.train.Saver()
        saver.restore(session, save_dir)

    def save_metadata(self, out_dir, var_name, meta_filename='metadata.tsv', config_filename='projector_config.pbtxt'):
        # Save metadata for visualizing embeddings
        with open(out_dir + '/' + meta_filename, 'w') as out:
            out.write('<none>\n')
            for value in self.model.game.object_vars.get_nominal_values(var_name):
                out.write(f'{value}\n')
        i = self.model.game.object_vars.get_nominal().index(var_name)
        embedding_layer = self.encoder_module.object_embedding_layers[i]
        embedding_name = embedding_layer.embeddings.name
        with open(out_dir + '/' + config_filename, 'w') as out:
            out.write('embeddings {\n')
            out.write('  tensor_name: "{}"\n'.format(embedding_name))
            out.write('  metadata_path: "{}"\n'.format(meta_filename))
            out.write('}\n')


def get_tensors(graph_tuples):
    """Get the tuple of tensors needed for any GraphTuple by any network operations"""
    graph_tuples = utils_tf.make_runnable_in_session(graph_tuples)
    return graph_tuples.nodes, graph_tuples.n_node, \
        graph_tuples.edges, graph_tuples.n_edge, \
        graph_tuples.senders, graph_tuples.receivers, \
        graph_tuples.globals
