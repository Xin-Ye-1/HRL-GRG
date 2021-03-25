import tensorflow as tf

def update_target_graph(from_scope, to_scope, tau=1):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var.value()*tau + (1-tau)*to_var.value()))
    return op_holder

def update_multiple_target_graphs(from_scopes, to_scopes, tau=1):
    op_holder = []
    for from_scope, to_scope in zip(from_scopes, to_scopes):
        op_holder += update_target_graph(from_scope, to_scope, tau)
    return op_holder

def get_distinct_list(inputs, add_on=None, remove=None):
    result = []
    if add_on is not None:
        result.append(add_on)
    for input in inputs:
        for element in input:
            if element != remove and element not in result:
                result.append(element)
    return result
