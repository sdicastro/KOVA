import itertools

def grid_search():
    envs = ['Swimmer-v2', 'Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2']
    kalman_lr = [1., 0.1, 0.01]
    kalman_eta = [0.01, 0.1, 0.001]
    kalman_onv_coeff = [1.]  # onv=observation noise variance
    kalman_onv_type = ['batch_size', 'max-ratio']

    combinations = list(itertools.product(envs, kalman_lr, kalman_eta, kalman_onv_coeff, kalman_onv_type))

    for inx, comb in enumerate(combinations):
        if inx >= 0:
            from mujoco_run import main_run
            print('comb: ', inx, comb)
            env = comb[0]
            print("env", env)
            kalman_lr = comb[1]
            kalman_eta = comb[2]
            kalman_onv_coeff = comb[3]
            kalman_onv_type = comb[4]
            comb_num = inx
            main_run(env, kalman_lr, kalman_eta, kalman_onv_coeff, kalman_onv_type, comb_num)


if __name__ == '__main__':
    grid_search()
