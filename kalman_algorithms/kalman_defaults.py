def kalman():
    return dict(
        kalman_lr=1.,
        kalman_only_last_layer=False,
        kalman_separate_vars=False,
        kalman_onv_coeff=1.,
        kalman_eta=0.01,
        kalman_onv_type='batch-size'
    )