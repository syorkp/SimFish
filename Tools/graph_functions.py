
# These functions allows us to update the parameters of our target network with those of the primary network.
def update_target_graph(tf_vars, tau):
    total_vars = len(tf_vars)
    op_holder = []
    for idx, var in enumerate(tf_vars[0:total_vars // 2]):
        op_holder.append(tf_vars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tf_vars[idx + total_vars // 2].value())))
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)
