hidden_dim = 768

opt = {
    'optimizer': {
        'type': 'AdamWeightDecayOptimizer',
        'kwargs': {
            'weight_decay_rate': 0.01,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-6,
            'exclude_from_weight_decay': ['LayerNorm', 'layer_norm', 'bias']
        }
    },
    'gradient_clip': {
        'type': 'clip_by_global_norm',
        'kwargs': {
            'clip_norm': 1.0,
        }
    }
}

# By default, we use warmup and linear decay for learinng rate
lr = {
    'static_lr': 2e-5,
}
