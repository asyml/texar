import tensorflow as tf
import collections
import re
import pprint

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = {
        'bert/embeddings/word_embeddings': 'bert/embeddings/word_embeddings/word_embeddings',
        'bert/embeddings/token_type_embeddings': 'bert/embeddings/token_type_embeddings/token_type_embeddings',
        'bert/embeddings/position_embeddings': 'bert/encoder/position_embedder/position_embedder',
        'bert/embeddings/LayerNorm/beta': 'bert/encoder/LayerNorm/beta',
        'bert/embeddings/LayerNorm/gamma': 'bert/encoder/LayerNorm/gamma',
    }
    for check_name, shape in init_vars:
        if check_name.startswith('bert'):
            if check_name.startswith('bert/embeddings'):
                continue
            model_name = re.sub('layer_\d+/output/dense',
                                lambda x: x.group(0).replace('output/dense', 'ffn/output'),
                                check_name)
            if model_name == check_name:
                model_name = re.sub('layer_\d+/output/LayerNorm',
                                    lambda x: x.group(0).replace('output/LayerNorm', 'ffn/LayerNorm'),
                                    check_name)
            if model_name == check_name:
                model_name = re.sub('layer_\d+/intermediate/dense',
                                    lambda x: x.group(0).replace('intermediate/dense', 'ffn/intermediate'),
                                    check_name)
            if model_name == check_name:
                model_name = re.sub('attention/output/dense', 'attention/self/output', check_name)
            if model_name == check_name:
                model_name = check_name.replace('attention/output/LayerNorm', 'output/LayerNorm')
            assert model_name in name_to_variable.keys(), 'model name:{} not exists!'.format(model_name)

            assignment_map[check_name] = model_name
            initialized_variable_names[model_name] = 1
            initialized_variable_names[model_name + ":0"] = 1
    return (assignment_map, initialized_variable_names)
