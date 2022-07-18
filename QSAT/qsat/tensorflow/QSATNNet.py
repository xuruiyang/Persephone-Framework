import sys
sys.path.append('..')
from utils import *

import tensorflow as tf
import numpy as np

def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden

class QSATNNet():

    def __init__(self, game, args):
        # game params
        self.action_size = game.getActionSize()
        self.args = args
        self.placeholders = {}
        self.num_edge_types = 4
        self.weights={}
        self.ops = {}

        # Renaming functions 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense

        # Params
        self.params={
            'num_epochs': 10,
            'patience': 25,
            'learning_rate': 0.01,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,
            'hidden_size': 128,
            'num_timesteps': 10,
            'use_graph': True,
            'tie_fwd_bkwd': True,
            'task_ids': [0,1],
            'random_seed': 0,
            'train_file': 'molecules_train.json',
            'valid_file': 'molecules_valid.json',
            'batch_size': 256,
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': {},
            'use_edge_bias': True,
            'edge_weight_dropout_keep_prob': 1
        }

        # Neural Net
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.make_model()
            self.make_train_step()
            self.train_writer = tf.summary.FileWriter( './logs/1/train ', self.graph)
            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            self.sess.run(init_op)

    def prepare_specific_graph_model(self) -> None:
        h_dim = self.params['hidden_size']
        # inputs
        self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='edge_weight_dropout_keep_prob')
        self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32,
                                                                          [None, None, self.params['hidden_size']],
                                                                          name='node_features')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, ())
        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32,
                                                               [None, self.num_edge_types, None, None])     # [b, e, v, v]
        self.__adjacency_matrix = tf.transpose(self.placeholders['adjacency_matrix'], [1, 0, 2, 3])         # [e, b, v, v]

        # weights
        self.weights['edge_weights'] = tf.Variable(glorot_init([self.num_edge_types, h_dim, h_dim]))
        if self.params['use_edge_bias']:
            self.weights['edge_biases'] = tf.Variable(np.zeros([self.num_edge_types, 1, h_dim]).astype(np.float32))
        with tf.variable_scope("gru_scope"):
            cell = tf.contrib.rnn.GRUCell(h_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 state_keep_prob=self.placeholders['graph_state_keep_prob'])
            self.weights['node_gru'] = cell

    def gated_regression0(self, last_h, regression_gate, regression_transform, f_dim=2):
        # last_h: [b x v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis = 2)        # [b, v, 2h]
        gate_input = tf.reshape(gate_input, [-1, 2 * self.params["hidden_size"]])                           # [b*v, 2h]
        last_h = tf.reshape(last_h, [-1, self.params["hidden_size"]])                                       # [b*v, h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)           # [b*v, 1]
        if f_dim == 2:
            gated_outputs = tf.reshape(gated_outputs, [-1, self.placeholders['num_vertices'],2])                  # [b, v,2]
        else:
            gated_outputs = tf.nn.tanh(regression_gate(gate_input)) * regression_transform(last_h) 
            gated_outputs = tf.reshape(gated_outputs, [-1, self.placeholders['num_vertices']])                  # [b, v]
        output = tf.reduce_sum(gated_outputs, axis = 1)                                                         # b or [b,2] 
        return output

    def gated_regression(self, last_h, regression_gate, regression_transform, f_dim=2):
        # last_h: [b x v x h]
        gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis = 2)        # [b, v, 2h]
        gate_input = tf.reshape(gate_input, [-1, 2 * self.params["hidden_size"]])                           # [b*v, 2h]
        last_h = tf.reshape(last_h, [-1, self.params["hidden_size"]])                                       # [b*v, h]
        gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)           # [b*v, 1]
        if f_dim == 2:
            gated_outputs = tf.reshape(gated_outputs, [-1, self.placeholders['num_vertices'],2])                  # [b, v,2]
        else: 
            gated_outputs = tf.reshape(gated_outputs, [-1, self.placeholders['num_vertices']])                  # [b, v]
        output = tf.reduce_sum(gated_outputs, axis = 1)                                                         # b or [b,2] 
        return output

    def compute_final_node_representations(self) -> tf.Tensor:
        v = self.placeholders['num_vertices']
        h_dim = self.params['hidden_size']
        h = self.placeholders['initial_node_representation']                                                # [b, v, h]
        h = tf.reshape(h, [-1, h_dim])

        with tf.variable_scope("gru_scope") as scope:
            for i in range(self.params['num_timesteps']):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                for edge_type in range(self.num_edge_types):
                    m = tf.matmul(h, tf.nn.dropout(self.weights['edge_weights'][edge_type],
                                                   keep_prob=self.placeholders['edge_weight_dropout_keep_prob'])) # [b*v, h]
                    m = tf.reshape(m, [-1, v, h_dim])                                                       # [b, v, h]
                    if self.params['use_edge_bias']:
                        m += self.weights['edge_biases'][edge_type]                                         # [b, v, h]
                    if edge_type == 0:
                        acts = tf.matmul(self.__adjacency_matrix[edge_type], m)
                    else:
                        acts += tf.matmul(self.__adjacency_matrix[edge_type], m)
                acts = tf.reshape(acts, [-1, h_dim])                                                        # [b*v, h]

                h = self.weights['node_gru'](acts, h)[1]                                                    # [b*v, h]
            last_h = tf.reshape(h, [-1, v, h_dim])
        return last_h

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        self.grads = grads_and_vars
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        # self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        self.ops['train_step'] = optimizer.minimize(self.ops['loss'])
        # Initialize newly-introduced variables:
        self.sess.run(tf.local_variables_initializer())

    def make_train_step0(self):
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        self.grads = optimizer.compute_gradients(self.ops['loss'])
        self.ops['train_step'] = optimizer.minimize(self.ops['loss'])

    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None, 2],        # [t,b,2]
                                                            name='target_values')
        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')

        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            # This does the actual graph work:
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['initial_node_representation'])

        self.ops['losses'] = []
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                with tf.variable_scope("regression_gate"):
                    # I use this hack because PI and V has different dimension!
                    self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 2-task_id, [],
                                                                           self.placeholders['out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 2-task_id, [],
                                                                                self.placeholders['out_layer_dropout_keep_prob'])
                computed_values = self.gated_regression(self.ops['final_node_representations'],
                                                        self.weights['regression_gate_task%i' % task_id],
                                                        self.weights['regression_transform_task%i' % task_id], 2-task_id)

                # use different loss for different task
                if task_id==0:      # Pi
                    pi = computed_values
                    prob = tf.nn.softmax(pi)

                    # it is crucial to get loss on raw pi instead of the sofmax value!
                    self.loss_pi = tf.losses.softmax_cross_entropy(self.placeholders['target_values'][internal_id,:,:], pi)
                    # self.loss_pi = tf.losses.mean_squared_error(self.placeholders['target_values'][internal_id,:,:], pi)
                    self.ops['losses'].append(self.loss_pi)
                elif task_id==1:    # V
                    vv = computed_values
                    v = tf.nn.tanh(vv)    # v in [-1,1]
                    # v = computed_values
                    self.loss_v = tf.losses.mean_squared_error(tf.reshape(tf.slice(self.placeholders['target_values'][internal_id,:,:],[0,0],[1,1]),[-1]), v)
                    self.ops['losses'].append(self.loss_v)
                else:
                    assert False

        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])
        self.ops['prob'] = prob
        self.ops['v'] = v
        self.ops['vv'] = vv
        self.ops['pi'] = pi
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.ops['loss'])
            tf.summary.histogram("prob", prob)
            tf.summary.histogram("v", v)
        self.merged = tf.summary.merge_all()
