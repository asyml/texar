import tensorflow as tf
reader = tf.train.NewCheckpointReader('./my-model-17030')  ## error
var2shape = reader.get_variable_to_shape_map()
for var in var2shape:
    print('var:{} shape:{}'.format(var, var2shape[var]))

