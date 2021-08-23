import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

for summary in tf.train.summary_iterator("./Training-Output/scaffold_test-15/logs/events.out.tfevents.1629714803.sam-GL72-7QF"):
    if summary["tag"] == "sigma_angle":
        x = True
