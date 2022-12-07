import tensorflow.compat.v1 as tf

from Networks.base_network import BaseNetwork

tf.disable_v2_behavior()


class PPONetworkCritic(BaseNetwork):

    def __init__(self, simulation, rnn_dim, rnn_cell, my_scope, internal_states, outputs_per_step, new_simulation=True):
        super().__init__(simulation, rnn_dim, rnn_cell, my_scope, internal_states, outputs_per_step, new_simulation=new_simulation)

        #            ----------        Non-Reflected       ---------            #

        self.Value = tf.layers.dense(self.rnn_output, 1, None,
                                     kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_Value',
                                     trainable=True)

        #            ----------        Reflected       ---------            #

        self.Value_ref = tf.layers.dense(self.rnn_output_ref, 1, None,
                                         kernel_initializer=tf.orthogonal_initializer, name=my_scope + '_Value',
                                         reuse=True, trainable=True)
        #            ----------        Combined       ---------            #

        self.Value_output = tf.math.divide(tf.math.add(self.Value, self.Value_ref), 2.0, name="value_output")

        #            ----------        Loss functions       ---------            #

        # Placeholders
        self.returns_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='returns')

        self.critic_loss = tf.reduce_mean(
            tf.squared_difference(tf.squeeze(self.Value_output), self.returns_placeholder))

        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name='actor_optimizer_impulse').minimize(
            self.critic_loss)


