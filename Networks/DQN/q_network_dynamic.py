# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

from Networks.dynamic_base_network import DynamicBaseNetwork

tf.disable_v2_behavior()


class QNetworkDynamic(DynamicBaseNetwork):

    def __init__(self, simulation, my_scope, internal_states, internal_state_names, num_actions,
                 base_network_layers=None, modular_network_layers=None, ops=None, connectivity=None,
                 reflected=None, reuse_eyes=False):
        super().__init__(simulation, my_scope, internal_states, internal_state_names, action_dim=1, num_actions=num_actions,
                         base_network_layers=base_network_layers, modular_network_layers=modular_network_layers, ops=ops,
                         connectivity=connectivity, reflected=reflected, algorithm="dqn", reuse_eyes=reuse_eyes)

        # Shared
        self.AW = tf.Variable(tf.random_normal([self.processing_network_output.shape[1] // 2, num_actions]), name=my_scope + "aw")
        self.VW = tf.Variable(tf.random_normal([self.processing_network_output.shape[1] // 2, 1]), name=my_scope + "vw")
        self.Temp = tf.placeholder(shape=None, dtype=tf.float32)
        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
        self.actions_one_hot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)
        self.exp_keep = tf.placeholder(shape=None, dtype=tf.float32)

        # Main stream
        self.streamA, self.streamV = tf.split(self.processing_network_output, 2, 1)
        self.Value = tf.matmul(self.streamV, self.VW)
        self.Advantage = tf.matmul(self.streamA, self.AW)

        # Reflected stream
        self.streamA_ref, self.streamV_ref = tf.split(self.processing_network_output_ref, 2, 1)
        self.Value_ref = tf.matmul(self.streamV_ref, self.VW)
        self.Advantage_ref = tf.matmul(self.streamA_ref, self.AW)

        # Swapping rows in advantage - Note that this is specific to the current action space and order
        if num_actions == 10:
            self.Advantage_ref = tf.concat([self.Advantage_ref[0:, :][:, :1],
                                            self.Advantage_ref[0:, :][:, 2:3],
                                            self.Advantage_ref[0:, :][:, 1:2],
                                            self.Advantage_ref[0:, :][:, 3:4],
                                            self.Advantage_ref[0:, :][:, 5:6],
                                            self.Advantage_ref[0:, :][:, 4:5],
                                            self.Advantage_ref[0:, :][:, 6:7],
                                            self.Advantage_ref[0:, :][:, 8:9],
                                            self.Advantage_ref[0:, :][:, 7:8],
                                            self.Advantage_ref[0:, :][:, 9:]], axis=1)
        elif num_actions == 12:
            self.Advantage_ref = tf.concat([self.Advantage_ref[0:, :][:, :1],
                                            self.Advantage_ref[0:, :][:, 2:3],
                                            self.Advantage_ref[0:, :][:, 1:2],
                                            self.Advantage_ref[0:, :][:, 3:4],
                                            self.Advantage_ref[0:, :][:, 5:6],
                                            self.Advantage_ref[0:, :][:, 4:5],
                                            self.Advantage_ref[0:, :][:, 6:7],
                                            self.Advantage_ref[0:, :][:, 8:9],
                                            self.Advantage_ref[0:, :][:, 7:8],
                                            self.Advantage_ref[0:, :][:, 9:10],
                                            self.Advantage_ref[0:, :][:, 11:],
                                            self.Advantage_ref[0:, :][:, 10:11],
                                            ], axis=1)
        else:
            print(f"Q-Network not set up for {num_actions} actions")
            self.Advantage_ref = tf.concat([self.Advantage_ref[0:, :][:, :1],
                                            self.Advantage_ref[0:, :][:, 2:3],
                                            self.Advantage_ref[0:, :][:, 1:2],
                                            self.Advantage_ref[0:, :][:, 3:4],
                                            self.Advantage_ref[0:, :][:, 5:6],
                                            self.Advantage_ref[0:, :][:, 4:5],
                                            self.Advantage_ref[0:, :][:, 6:7],
                                            self.Advantage_ref[0:, :][:, 8:9],
                                            self.Advantage_ref[0:, :][:, 7:8],
                                            self.Advantage_ref[0:, :][:, 9:]], axis=1)

        #                ------------ Integrating Normal and Reflected ------------                   #

        self.Value_final = tf.divide(tf.add(self.Value, self.Value_ref), 2)
        self.Advantage_final = tf.divide(tf.add(self.Advantage, self.Advantage_ref), 2)

        self.salience = tf.gradients(self.Advantage_final, self.observation)
        # Then combine them together to get our final Q-values.
        self.Q_out = self.Value_final + tf.subtract(self.Advantage_final,
                                                    tf.reduce_mean(self.Advantage_final, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Q_out, 1)
        self.Q_dist = tf.nn.softmax(self.Q_out / self.Temp)
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_one_hot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)

        # In order to only propagate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        self.maskA = tf.zeros([self.batch_size, self.train_length // 2])
        self.maskB = tf.ones([self.batch_size, self.train_length // 2])
        self.mask = tf.concat([self.maskA, self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.updateModel = self.trainer.minimize(self.loss)
