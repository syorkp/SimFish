import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

from Analysis.Neural.CNN.MEI.graphs_for_mei import MEICore, MEIReadout, Trainer, TrainerExtended

tf.logging.set_verbosity(tf.logging.ERROR)


def produce_mei(model_name):

    # Initial, random image.
    image = np.random.normal(128, 8, size=(100, 1))
    image = np.clip(image, 0, 255)
    image /= 255
    # image = np.random.uniform(size=(100, 3))
    iterations = 10000

    with tf.Session() as sess:
        # Creating graph
        core = MEICore()
        readout = MEIReadout(core.output)
        # trainer = Trainer(readout.predicted_neural_activity)

        saver = tf.train.Saver(max_to_keep=5)
        model_location = "MEI-Models/" + model_name
        checkpoint = tf.train.get_checkpoint_state(model_location)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        # Constants
        step_size = 1.5
        step_gain = 1
        eps = 1e-12

        grad = tf.gradients(readout.predicted_neural_activity, core.observation, name='grad')
        red_grad = tf.reduce_sum(grad)
        gradients = []
        # writer = tf.summary.FileWriter(f"MEI-Models/test/logs/", tf.get_default_graph())
        pred = []
        reds = []
        for i in range(iterations):
            input_image = np.concatenate((np.zeros((100, 1)), image, np.zeros((100, 1))), axis=1)
            dy_dx, p, red = sess.run([grad, readout.predicted_neural_activity, red_grad],
                                     feed_dict={core.observation: np.expand_dims(input_image, 0)})
            gradients.append(dy_dx)
            update = (step_size / (np.mean(np.abs(dy_dx[0])) + eps)) * (step_gain / 255)
            input_image += update * dy_dx[0][0]
            image = input_image[:, 1:2]
            image = np.clip(image, 0, 1)
            reds.append(red)
            if np.max(np.absolute(dy_dx[0])) == 0:
                print(i)
                # Shouldnt ever really occur.
                image = np.random.normal(128, 8, size=(100, 1))
                image = np.clip(image, 0, 255)
                image /= 255
            pred.append(p)

        gradients = np.array(gradients)
        plt.imshow(np.expand_dims(np.concatenate((np.zeros((100, 2)), image), axis=1), axis=0))
        plt.show()


def produce_meis(model_name, n_units=8, iterations=1000):
    """Does the same thing for the multiple neurons of a given model"""

    # Initial, random image.
    all_images = np.zeros((n_units, 100, 3))

    # image = np.random.uniform(size=(100, 3))


    with tf.Session() as sess:
        # Creating graph
        core = MEICore()
        readout_blocks = {}
        for unit in range(n_units):
            readout_blocks[f"Unit {unit}"] = MEIReadout(core.output, my_scope="MEIReadout_" + str(unit))

        saver = tf.train.Saver(max_to_keep=5)
        model_location = "MEI-Models/" + model_name
        checkpoint = tf.train.get_checkpoint_state(model_location)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        # Constants
        step_size = 0.15#1.5
        step_gain = 1
        eps = 1e-12

        for unit in range(n_units):
            input_image = np.random.normal(128, 8, size=(100, 3))
            input_image = np.clip(input_image, 0, 255)
            input_image /= 255

            grad = tf.gradients(readout_blocks[f"Unit {unit}"].predicted_neural_activity, core.observation, name='grad')
            red_grad = tf.reduce_sum(grad)
            gradients = []
            # writer = tf.summary.FileWriter(f"MEI-Models/test/logs/", tf.get_default_graph())
            pred = []
            reds = []
            for i in range(iterations):
                # input_image = np.concatenate((image, np.zeros((100, 1))), axis=1)
                dy_dx, p, red = sess.run([grad, readout_blocks[f"Unit {unit}"].predicted_neural_activity, red_grad],
                                         feed_dict={core.observation: np.expand_dims(input_image, 0)})
                gradients.append(dy_dx)
                update = (step_size / (np.mean(np.abs(dy_dx[0])) + eps)) * (step_gain / 255)
                input_image += update * dy_dx[0][0]
                input_image[16:] = 0
                # image = input_image[:, 0:2]
                input_image = np.clip(input_image, 0, 1)
                reds.append(red)
                if np.max(np.absolute(dy_dx[0])) == 0:
                    print("ERROR")
                    # Shouldnt ever really occur.
                    # image = np.random.normal(128, 8, size=(100, 1))
                    # image = np.clip(image, 0, 255)
                    # image /= 255
                pred.append(p)
            all_images[unit] = input_image

    # plt.imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 1)), all_images[:, :, 1:2]), axis=2))
    # plt.imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 1)), all_images[:, :, 1:2]), axis=2))
    plt.imshow(all_images)
    plt.show()

    plt.imshow(np.concatenate((np.zeros((n_units, 100, 2)), all_images[:, :, 1:2]), axis=2))
    plt.show()

    plt.imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 2))), axis=2))
    plt.show()

    plt.imshow(np.concatenate((np.zeros((n_units, 100, 1)), all_images[:, :, 2:3], np.zeros((n_units, 100, 1))), axis=2))
    plt.show()

    plt.plot([np.sum(g) for g in gradients])
    plt.show()


if __name__ == "__main__":
    produce_meis("layer_3", n_units=8, iterations=2000)
