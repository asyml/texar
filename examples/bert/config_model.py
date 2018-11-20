opt = {
    'learning_rate': 2e-5,
}

downstream_config = {
    'initializer': 'truncated_normal_initializer',
    'hparams': {
        'stddev': 0.02,
    },
}
