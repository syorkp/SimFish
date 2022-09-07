import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

from Analysis.Neural.MEI.graphs_for_mei import MEICore, MEIReadout

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
        plt.savefig("./mei")
        plt.clf()


def produce_meis(model_name, n_units=8, iterations=1000):
    """Does the same thing for the multiple neurons of a given model"""

    # Initial, random image.
    all_images = np.zeros((n_units, 100, 3))

    # image = np.random.uniform(size=(100, 3))

    with tf.Session() as sess:
        # Creating graph
        core = MEICore(model_name)
        readout_blocks = {}
        for unit in range(n_units):
            readout_blocks[f"Unit {unit}"] = MEIReadout(core.output, my_scope=model_name + "_readout_" + str(unit))

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
                # image = input_image[:, 0:2]
                input_image = np.clip(input_image, 0, 1)
                reds.append(red)
                if np.max(np.absolute(dy_dx[0])) == 0:
                    print(f"{i}-ERROR")
                    # Shouldn't ever occur.
                    image = np.random.normal(128, 8, size=(100, 1))
                    image = np.clip(image, 0, 255)
                    image /= 255
                pred.append(p)
            all_images[unit] = input_image

    # plt.imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 1)), all_images[:, :, 1:2]), axis=2))
    # plt.imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 1)), all_images[:, :, 1:2]), axis=2))
    fig, axs = plt.subplots(4, 1)
    axs[0].imshow(all_images)
    axs[1].imshow(np.concatenate((np.zeros((n_units, 100, 2)), all_images[:, :, 1:2]), axis=2))
    axs[2].imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 2))), axis=2))
    axs[3].imshow(np.concatenate((np.zeros((n_units, 100, 1)), all_images[:, :, 2:3], np.zeros((n_units, 100, 1))), axis=2))
    plt.savefig(f"./mei-{model_name}")
    plt.clf()

    plt.plot([np.sum(g) for g in gradients])
    plt.savefig(f"./mei-{model_name}-gradients")
    plt.clf()


class Model():

    def __init__(self, my_scope, n_units):
        self.core = MEICore(my_scope)
        self.readout_blocks = {}
        for unit in range(n_units):
            self.readout_blocks[f"Unit {unit}"] = MEIReadout(self.core.output, my_scope=my_scope + "_readout_" + str(unit))


def produce_meis_mulitiple_models(model_names, overall_model_name, n_units=8, iterations=1000):
    """Does the same thing for the multiple neurons of a given model"""

    # Initial, random image.
    all_images = np.zeros((n_units, 100, 3))

    # image = np.random.uniform(size=(100, 3))

    with tf.Session() as sess:
        # Creating graph
        loaded_models = {}
        for model_name in model_names:
            model = Model(model_name, n_units)
            loaded_models[model_name] = model

            saver = tf.train.Saver([v for v in tf.all_variables() if model_name in v.name], max_to_keep=5)
            model_location = f"MEI-Models/{overall_model_name}/{model_name}"

            checkpoint = tf.train.get_checkpoint_state(model_location)
            saver.restore(sess, checkpoint.model_checkpoint_path)

        # Constants
        step_size = 0.15#1.5
        eps = 1e-12

        for unit in range(n_units):
            input_image = np.random.normal(10, 8, size=(100, 3))
            input_image = np.clip(input_image, 0, 255)
            input_image /= 255

            for model in model_names:
                loaded_models[model].grad = tf.gradients(loaded_models[model].readout_blocks[f"Unit {unit}"].predicted_neural_activity,
                                                         loaded_models[model].core.observation, name=model+'grad')
                loaded_models[model].red_grad = tf.reduce_sum(loaded_models[model].grad)
            gradients = []
            # writer = tf.summary.FileWriter(f"MEI-Models/test/logs/", tf.get_default_graph())
            pred = []
            reds = []
            for i in range(iterations):
                # input_image = np.concatenate((image, np.zeros((100, 1))), axis=1)
                grads_step = []
                for model in model_names:
                    dy_dx, p, red = sess.run([loaded_models[model].grad,
                                              loaded_models[model].readout_blocks[f"Unit {unit}"].predicted_neural_activity,
                                              loaded_models[model].red_grad],
                                         feed_dict={loaded_models[model].core.observation: np.expand_dims(input_image, 0)})
                    grads_step.append(dy_dx)
                dy_dx = np.mean(np.array(grads_step), axis=0)
                gradients.append(dy_dx)
                update = (step_size / (np.mean(np.abs(dy_dx[0])) + eps)) * (1 / 255)
                input_image += update * dy_dx[0][0]
                # image = input_image[:, 0:2]
                input_image = np.clip(input_image, 0, 1)
                reds.append(red)
                if np.max(np.absolute(dy_dx[0])) == 0:
                    print(f"{i}-ERROR")
                    # Shouldnt ever really occur.
                    # image = np.random.normal(128, 8, size=(100, 1))
                    # image = np.clip(image, 0, 255)
                    # image /= 255
                pred.append(p)
            all_images[unit] = input_image

    # plt.imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 1)), all_images[:, :, 1:2]), axis=2))
    # plt.imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 1)), all_images[:, :, 1:2]), axis=2))
    fig, axs = plt.subplots(4, 1)
    axs[0].imshow(all_images)
    axs[1].imshow(np.concatenate((np.zeros((n_units, 100, 2)), all_images[:, :, 1:2]), axis=2))
    axs[2].imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 2))), axis=2))
    axs[3].imshow(np.concatenate((np.zeros((n_units, 100, 1)), all_images[:, :, 2:3], np.zeros((n_units, 100, 1))), axis=2))
    plt.savefig(f"./mei-{model_names[0][:-2]}")
    plt.clf()

    # Save Optimal activation
    with open(f"MEI-Models/{overall_model_name}/{model_names[0][:-2]}-optimal_activation.npy", "wb") as f:
        np.save(f, all_images)

    plt.plot([np.sum(g) for g in gradients])
    plt.savefig(f"./mei-{model_names[0][:-2]}-gradients")
    plt.clf()


if __name__ == "__main__":
    # produce_meis("layer_1_4", n_units=16, iterations=4000)
    produce_meis_mulitiple_models(["layer_1_1", "layer_1_2", "layer_1_3", "layer_1_4"],
                                  overall_model_name="dqn_scaffold_18-1", n_units=16, iterations=10000)
    # produce_meis_mulitiple_models(["layer_4_1", "layer_4_2", "layer_4_3", "layer_4_4"], n_units=64, iterations=10000)
    # produce_meis_mulitiple_models(["layer_3_1", "layer_3_2", "layer_3_3", "layer_3_4"], n_units=8, iterations=10000)
    x = True
