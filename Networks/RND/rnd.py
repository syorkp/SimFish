import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from Networks.base_network_2 import BaseNetwork2


class RandomNetworkDistiller(BaseNetwork2):

    def __init__(self, simulation, my_scope, internal_states, predictor, new_simulation=True):
        super().__init__(simulation, my_scope, internal_states, action_dim=2, new_simulation=new_simulation)

        self.stream_1, self.stream_2 = tf.split(self.output, 2, 1)
        self.stream_1_ref, self.stream_2_ref = tf.split(self.output_ref, 2, 1)

        self.output_scalar_1 = tf.layers.dense(self.stream_1, 1, activation=tf.nn.sigmoid,
                                               kernel_initializer=tf.orthogonal_initializer,
                                               name=my_scope + '_stream_1', trainable=True)
        self.output_scalar_1_ref = tf.layers.dense(self.stream_1_ref, 1, activation=tf.nn.sigmoid,
                                                   kernel_initializer=tf.orthogonal_initializer,
                                                   name=my_scope + '_stream_1', trainable=True, reuse=True)

        self.output_scalar_2 = tf.layers.dense(self.stream_2, 1, activation=tf.nn.tanh,
                                               kernel_initializer=tf.orthogonal_initializer,
                                               name=my_scope + '_stream_2', trainable=True)
        self.output_scalar_2_ref = tf.layers.dense(self.stream_2_ref, 1, activation=tf.nn.tanh,
                                                   kernel_initializer=tf.orthogonal_initializer,
                                                   name=my_scope + '_stream_2', trainable=True, reuse=True)

        self.output_1_combined = tf.divide(tf.add(self.output_scalar_1, self.output_scalar_1_ref), 2)
        self.output_2_combined = tf.divide(tf.subtract(self.output_scalar_2, self.output_scalar_2_ref), 2)

        self.rdn_output = tf.concat([self.output_1_combined, self.output_2_combined], axis=1)

        if predictor:
            self.max_gradient_norm = 0.5

            self.target_outputs = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="output_target")
            self.target_output_1, self.target_output_2 = tf.split(self.target_outputs, 2, axis=1)

            self.loss = tf.squared_difference(tf.squeeze(self.output_1_combined), tf.squeeze(self.target_output_1)) + \
                        tf.squared_difference(tf.squeeze(self.output_2_combined), tf.squeeze(self.target_output_2))
            self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

            self.model_params = tf.trainable_variables()
            self.model_gradients = tf.gradients(self.loss, self.model_params)
            self.model_gradients, _grad_norm = tf.clip_by_global_norm(self.model_gradients, self.max_gradient_norm)
            self.model_gradients = list(zip(self.model_gradients, self.model_params))

            self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-5)
            self.train = self.trainer.apply_gradients(self.model_gradients)
