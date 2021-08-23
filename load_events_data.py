import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


for summary in tf.train.summary_iterator("Training-Output/scaffold_test-14/logs/events.out.tfevents.1629711656.sam-GL72-7QF"):
    for e in summary.summary.value:
        if e.tag == "mu_impulse":
            x = True


