import tensorflow.compat.v1 as tf

from Networks.base_network import BaseNetwork

tf.disable_v2_behavior()


class PPONetworkActor(BaseNetwork):

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states, clip_param, num_actions, epsilon_greedy=False, new_simulation=False):

        # Variables
        super().__init__(simulation, rnn_dim, rnn_cell, my_scope, internal_states, action_dim=1, new_simulation=new_simulation)

        #            ----------        Non-Reflected       ---------            #

        # TODO: Differences from SB: no activation function? Logits inputted to probability distribution.
        #  Difference in sampling - uses randomorm to select, with no softmax involved.
        #  With SB version, problem seems to be fixed when action probabiltiy goes down.
        #  Problem that initial dropff in CS usage due to cost, should use reward scaffolding here.

        self.action_values = tf.layers.dense(self.rnn_output, num_actions, activation=tf.nn.sigmoid,
                                             kernel_initializer=tf.orthogonal_initializer,
                                             name=my_scope + '_action_values', trainable=True)

        #            ----------        Reflected       ---------            #

        self.action_values_ref = tf.layers.dense(self.rnn_output_ref, num_actions, activation=tf.nn.sigmoid,
                                                 kernel_initializer=tf.orthogonal_initializer,
                                                 name=my_scope + '_action_values', trainable=True, reuse=True)
        self.action_values_ref = tf.concat([self.action_values_ref[0:, :][:, :1],
                                            self.action_values_ref[0:, :][:, 2:3],
                                            self.action_values_ref[0:, :][:, 1:2],
                                            self.action_values_ref[0:, :][:, 3:4],
                                            self.action_values_ref[0:, :][:, 5:6],
                                            self.action_values_ref[0:, :][:, 4:5],
                                            self.action_values_ref[0:, :][:, 6:7],
                                            self.action_values_ref[0:, :][:, 8:9],
                                            self.action_values_ref[0:, :][:, 7:8],
                                            self.action_values_ref[0:, :][:, 9:]], axis=1)

        #            ----------        Combined       ---------            #

        # Combined Actor impulse output
        self.action_values_combined = tf.math.divide(tf.add(self.action_values, self.action_values_ref), 2.0,
                                                     name="Action_values_combined")
        self.action_probabilities = tf.nn.softmax(self.action_values_combined)  # BS.TL x An

        # TODO: remove the softmax activation (so that it outputs logits) and then feed those logits into
        #  tf.distributions.Categorical(logits=logits). Although mathematically equivalent, this can be more robust
        #  numerically

        if epsilon_greedy:
            # Epsilon-greedy
            self.action_output = tf.argmax(self.action_values_combined, 1)  # BS.TL
            self.action_output = tf.reshape(self.action_output, [self.batch_size*self.trainLength, 1])  # BS.TL x 1
        else:
            # Probabilistic.
            self.action_output = tf.random.categorical(tf.log(self.action_probabilities), 1)  # BS.TL x 1

        self.batch_indexes = tf.reshape(tf.range(0, self.batch_size*self.trainLength, 1, dtype=tf.int64), [self.batch_size*self.trainLength, 1])  # BS.TL x 1
        self.action_indices = tf.concat([self.batch_indexes, self.action_output], 1)  # BS.TL x 2
        self.chosen_action_probability = tf.gather_nd(self.action_probabilities, indices=self.action_indices)  # BS.TL

        #            ----------        Loss functions       ---------            #

        # Placeholders
        self.action_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='action_placeholder')

        self.new_log_prob_action = self.chosen_action_probability  # TODO: Check format - may not be log
        # TODO: Move chosen action probability here.
        self.old_log_prob_action_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='old_log_prob_angle')

        self.scaled_advantage_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='scaled_advantage')

        # COMBINED LOSS

        self.action_ratio = tf.exp(self.new_log_prob_action - self.old_log_prob_action_placeholder)
        self.action_surrogate_loss_1 = tf.math.multiply(self.action_ratio, self.scaled_advantage_placeholder)
        self.action_surrogate_loss_2 = tf.math.multiply(
            tf.clip_by_value(self.action_ratio, 1 - clip_param, 1 + clip_param), self.scaled_advantage_placeholder)
        self.action_loss = -tf.reduce_mean(tf.minimum(self.action_surrogate_loss_1, self.action_surrogate_loss_2))

        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name='actor_optimizer_impulse').minimize(
            self.action_loss)

    @staticmethod
    def bounded_output(x, lower, upper):
        scale = upper - lower
        return scale * tf.nn.sigmoid(x) + lower

