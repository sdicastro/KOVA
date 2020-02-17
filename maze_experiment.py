""" The code for the maze is adopted from https://www.samyzaf.com/ML/rl/qmaze.html with changes"""
from __future__ import print_function
import os, time, datetime, random
import numpy as np
from baselines import logger
import json
import shutil
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

import tensorflow as tf
import matplotlib.pyplot as plt

from kalman_algorithms.kalmanoptimizer import KalmanOptimizer
from kalman_algorithms.ktdoptimizer import KTDOptimizer

visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
rat_mark = 0.5      # The current rat cell will be painted by gray 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1


def constfn(val):
    def f(_):
        return val
    return f


# maze is a 2d Numpy array of floats between 0.0 to 1.0
# 1.0 corresponds to a free cell, and 0.0 an occupied cell
# rat = (row, col) initial rat position (defaults to (0,0))
class Experience(object):
    def __init__(self, model, max_memory=100, gamma=0.95):
        self.model = model
        self.max_memory = max_memory
        self.gamma = gamma
        self.memory = list()
        self.num_actions = model.num_actions

    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def predict_target(self, envstate):
        return self.model.predict_target(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        actions = np.zeros((data_size, ), dtype=np.int32)
        next_states = np.zeros((data_size, env_size))
        rewards = np.zeros((data_size, ), dtype=np.float32)
        targets = np.zeros((data_size, ))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            actions[i] = int(action)
            inputs[i] = envstate
            next_states[i] = envstate_next
            rewards[i] = reward
            max_action = np.argmax(self.predict(envstate_next)) # implementation of Double-Q-learning
            Q1 = self.predict_target(envstate_next) # estimate Q using target network
            Q_sa = Q1[max_action]
            if game_over:
                targets[i] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i] = reward + self.gamma * Q_sa
        return inputs, targets, actions, next_states, rewards


class Qmaze(object):
    def __init__(self, maze, rat=(0, 0)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows - 1, ncols - 1)  # target cell where the "cheese" is
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        self.free_cells.remove(self.target)

        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(rat)

    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:  # invalid action, no change in rat position
            mode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return -0.25
        if mode == 'valid':
            return -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = rat_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)

        if row > 0 and self.maze[row - 1, col] == 0.0:
            actions.remove(1)
        if row < nrows - 1 and self.maze[row + 1, col] == 0.0:
            actions.remove(3)

        if col > 0 and self.maze[row, col - 1] == 0.0:
            actions.remove(0)
        if col < ncols - 1 and self.maze[row, col + 1] == 0.0:
            actions.remove(2)

        return actions


def qtrain(model, maze, total_timesteps=10000, **opt):
    global epsilon
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    gamma = opt.get('gamma', 0.99)
    start_time = datetime.datetime.now()

    total_timesteps = total_timesteps
    log_freq = 50
    # check if adam_learning rate is constant or decaying with a function:
    lr = model.adam_lr
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    adam_lr_list = []

    tstart = time.time()
    # Construct environment/game from numpy array: maze (see above)
    qmaze = Qmaze(maze)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory, gamma=gamma)

    win_history = []  # history of win/lose game
    hsize = qmaze.maze.size // 2  # history window size
    win_rate = 0.0
    loss_list, step_to_win_list, timesteps_list, win_count_list, win_rate_list, game_reward_list, \
        time_list, time_so_far_list, predict_list, fit_list = [], [], [], [], [], [], [], [], [], []
    list_of_data = [loss_list, step_to_win_list, timesteps_list, win_count_list, win_rate_list, game_reward_list,
                    time_list, time_so_far_list, predict_list, fit_list]
    list_of_names = ["loss", "steps_to_win", "timesteps", "win_count", "win_rate", "game_reward", "time",
                     "time_so_far", "calls_to_predict", "calls_to_fit"]
    target_network_update_freq = 200  # updating the target network every X steps

    game_over = False
    game_reward = 0
    loss = 0.0
    n_steps_for_win = 0
    rat_cell = random.choice(qmaze.free_cells)
    qmaze.reset(rat_cell)
    # get initial envstate (1d flattened canvas)
    envstate = qmaze.observe()
    for n_timesteps in range(total_timesteps):
        # n_timesteps counts how many environment steps the agent has done
        if game_over:
            rat_cell = random.choice(qmaze.free_cells)
            qmaze.reset(rat_cell)
            # get initial envstate (1d flattened canvas)
            envstate = qmaze.observe()

            n_steps_for_win = 0
            game_reward = 0
            game_over = False

        # while not game_over:
        valid_actions = qmaze.valid_actions()
        if not valid_actions:
            break
        prev_envstate = envstate
        # Get next action
        if np.random.rand() < epsilon:
            action = random.choice(valid_actions)
        else:
            action = np.argmax(experience.predict(prev_envstate))

        # Apply action, get reward and new envstate
        envstate, reward, game_status = qmaze.act(action)
        game_reward += reward
        if game_status == 'win':
            win_history.append(1)
            game_over = True
        elif game_status == 'lose':
            win_history.append(0)
            game_over = True
        else:
            game_over = False

        # Store episode (experience)
        episode = [prev_envstate, action, reward, envstate, game_over]
        experience.remember(episode)
        n_steps_for_win += 1

        if n_timesteps > 0:
            frac = 1.0 - (n_timesteps - 1.0) / total_timesteps
            # Calculate the learning rate
            lrnow = lr(frac)
        else:
            lrnow = 1.
        adam_lr_list.append(lrnow)

        # Train neural network model
        if model.optimizer_method == 'ktd':
            # KTD does not use experience reply. It optimize over current transition.
            env_size = prev_envstate.shape[1]
            states = np.zeros((1, env_size))
            actions = np.zeros((1,), dtype=np.int32)
            next_states = np.zeros((1, env_size))
            rewards = np.zeros((1,), dtype=np.float32)

            states[0] = prev_envstate
            next_states[0] = envstate
            actions[0] = action
            rewards[0] = reward

            model.fit(states, None, actions, next_states=next_states, rewards=rewards)
            calls_to_predict = model.trainer.calls_to_predict

        else:  # start training after collecting at least "data_size" of transitions
            if len(experience.memory) >= data_size:
                inputs, targets, actions, next_states, rewards = experience.get_data(data_size=data_size)
                h = model.fit(inputs, targets, actions, lr=lrnow, next_states=next_states, rewards=rewards)
                loss = model.evaluate(inputs, targets, actions)

                # Update target network for DQN
                if n_timesteps % target_network_update_freq == 0:
                    # Update target network periodically.
                    model.update_target()

            calls_to_predict = model.calls_to_predict

        calls_to_fit = model.calls_to_fit

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        tnow = time.time()
        time_so_far = tnow - tstart

        if n_timesteps % log_freq == 0:  # save every "log_freq" n_timestieps
            template = "Timestep: {:03d}/{:d} | Loss: {:.4f} | steps for win: {:d} | Win count: {:d} | " \
                       "Win rate: {:.3f} | game reward: {:.3f} | time: {}"
            print(
                template.format(n_timesteps, total_timesteps - 1, loss, n_steps_for_win, sum(win_history),
                                win_rate, game_reward, t))

            [x.append(y.item()) if hasattr(y, 'dtype') else x.append(y)
             for x, y in zip(list_of_data, [loss, n_steps_for_win, n_timesteps, sum(win_history),
                                            win_rate, game_reward, t, time_so_far, calls_to_predict, calls_to_fit])]
            dict_for_json = {key: value for key, value in zip(list_of_names, list_of_data)}
            dict_for_json["optimizer"] = model.optimizer_method
            dict_for_json["maze_size"] = "{}".format(maze.shape)
            dict_for_json["batch_size"] = data_size
            dict_for_json["maze"] = qmaze.maze.tolist()
            dict_for_json["kalman_lr"] = model.kalman_lr
            dict_for_json["onv_coeff"] = model.onv_coeff
            dict_for_json["eta"] = model.eta
            dict_for_json["onv_type"] = model.onv_type
            dict_for_json["gamma"] = gamma
            dict_for_json["total_timesteps"] = total_timesteps
            dict_for_json["theta_size"] = model.theta_size

            if model.optimizer_method == 'ktd':
                dict_for_json["theta_noise"] = model.theta_noise
                dict_for_json["reard_noise"] = model.reward_noise
                dict_for_json["P_init"] = model.P_init
            if model.optimizer_method == 'adam':
                dict_for_json["adam_lr"] = adam_lr_list

            with open(logger.get_dir() + '/data.txt', 'w') as datafile:
                datafile.write(json.dumps(dict_for_json))
                datafile.write('\n')

        if win_rate > 0.9:
            epsilon = 0.05

    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    return seconds


# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


class Model:
    def __init__(self, maze, optimizer_method=None, gamma=0.99, kalman_lr=1., onv_coeff=1., eta=0.01,
                 onv_type='batch_size', batch_size=10, adam_lr=0.001):
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, maze.size), name="X")
        self.R = tf.placeholder(tf.float32, [None], name="R_batch")
        self.LR = tf.placeholder(tf.float32, [])
        self.action = tf.placeholder(tf.int32, [None])
        self.maze = maze
        self.q = None
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     intra_op_parallelism_threads=1,
                                                     inter_op_parallelism_threads=1,
                                                     gpu_options=tf.GPUOptions(allow_growth=True)))
        self.num_actions = num_actions
        self.optimizer_method = optimizer_method
        self.kalman_lr = kalman_lr
        self.onv_coeff = onv_coeff
        self.eta = eta
        self.onv_type = onv_type
        self.batch_size = batch_size
        self.calls_to_predict = 0
        self.calls_to_fit = 0
        self.gamma = gamma
        self.theta_size = 0
        self.adam_lr = adam_lr

        self.q, self.params = self.build_model(input=self.X, name="trained", reuse=True)
        self.q_target, self.target_params = self.build_model(input=self.X, name="target", reuse=False)

        # for evaluating a single state
        self.X_single = tf.placeholder(dtype=tf.float32, shape=(1, self.maze.size), name="X_single")
        self.action_single = tf.placeholder(tf.int32, [1], name="action_single")
        self.q_single, _ = self.build_model(input=self.X_single, name="trained", reuse=True)
        self.qa_single = self.q_single[:, self.action_single[0]]

        update_target_expr = []
        for var, var_target in zip(sorted(self.params, key=lambda v: v.name),
                                   sorted(self.target_params, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        self.update_target_expr = tf.group(*update_target_expr)

        self.training_op()

    def build_model(self, input=None, name='network', reuse=True):
        num_layers = 1
        num_hidden = self.maze.size
        if reuse:
            _reuse = tf.AUTO_REUSE
        else:
            _reuse = False
        with tf.variable_scope('{}_network'.format(name)):
            h = tf.layers.flatten(input)

            for i in range(num_layers):
                with tf.variable_scope('mlp_fc{}'.format(i), reuse=_reuse):
                    h = tf.contrib.layers.fully_connected(h, num_outputs=num_hidden, activation_fn=None)
                    h = tf.nn.relu(h)
            with tf.variable_scope('last_layer', reuse=_reuse):
                q = tf.contrib.layers.fully_connected(h, num_outputs=num_actions, activation_fn=None)
        params = [item for item in tf.trainable_variables() if name in item.name]
        total_parameters = 0
        for variable in params:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print(total_parameters)
        self.theta_size = total_parameters
        return q, params

    def training_op(self):
        q_selected = tf.reduce_sum(self.q * tf.one_hot(self.action, num_actions), 1)
        self.loss = .5 * tf.reduce_mean(tf.square(self.R - q_selected))
        if self.optimizer_method == 'adam':
            self.trainer = tf.train.AdamOptimizer(learning_rate=self.LR)
            self.train_op = self.trainer.minimize(self.loss, var_list=self.params)
        elif self.optimizer_method == 'kova':
            self.trainer = KalmanOptimizer(var_list=self.params, learning_rate=self.kalman_lr,
                                           onv_coeff=self.onv_coeff, eta=self.eta, onv_type=self.onv_type)

            q_function_gradient_list = [tf.gradients(q_selected[i], self.params)
                                        for i in range(self.batch_size)]

            self.q_function_gradient_single = [self.trainer.concatenate_tensors(
                [tf.gradients(self.q_single[:, a], self.params)]) for a in range(num_actions)]

            self.train_op = self.trainer.apply_kalman(innovation=self.R - q_selected, ratio=1.,
                                                      q_function_gradient_list=q_function_gradient_list,
                                                      list_of_vars=self.params, batch_size=self.batch_size)

            self.covariance = np.eye(self.trainer.size)

        self.sess.run(tf.global_variables_initializer())

    def predict(self, states):
        td_map = {self.X: states}
        self.calls_to_predict += 1
        return self.sess.run(self.q, td_map)

    def predict_target(self, states):
        td_map = {self.X: states}
        self.calls_to_predict += 1
        return self.sess.run(self.q_target, td_map)

    def fit(self, states, targets, actions, **kwargs):
        self.calls_to_fit += 1
        if self.optimizer_method == 'adam':
            lr = kwargs["lr"]
        else:
            lr = 1.  # doesn't have a meaning and does not affect kova.
        td_map = {self.X: states, self.R: targets, self.action: actions, self.LR: lr}
        return self.sess.run([self.loss, self.train_op], td_map)[:-1]

    def evaluate(self, states, targets, actions):
        td_map = {self.X: states, self.R: targets, self.action: actions}
        return self.sess.run(self.loss, td_map)

    def update_target(self):
        print("target network update")
        return self.sess.run(self.update_target_expr)

    def h_and_gradh(self, states):
        td_map = {self.X_single: states}
        return self.sess.run([self.q_single, self.q_function_gradient_single], td_map)

    def predict_q_single(self, states):
        td_map = {self.X_single: states}
        return self.sess.run([self.q_single], td_map)

    def theta_and_covariance(self, states, targets, actions):
        td_map = {self.X: states, self.R: targets, self.action: actions}

        return self.sess.run([self.params, self.trainer.covariance], td_map)


class ModelKTD(Model):
    def __init__(self, maze, optimizer_method=None, gamma=0.99, kalman_lr=1., onv_coeff=1., eta=0.01,
                 onv_type='batch_size', batch_size=10, theta_noise=0.01, reward_noise=1., P_init=10):
        """Construct a new Kalman optimizer.
        """
        self.theta_noise = theta_noise
        self.reward_noise = reward_noise
        self.P_init = P_init
        super(ModelKTD, self).__init__(maze, optimizer_method=optimizer_method,
                                       gamma=gamma, kalman_lr=kalman_lr,
                                       onv_coeff=onv_coeff, eta=eta,
                                       onv_type=onv_type, batch_size=batch_size)
        self.R_sigma = tf.placeholder(tf.float32, [None], name="R_sigma_points")
        self.covariance = None

    def training_op(self):
        self.X1 = tf.placeholder(dtype=tf.float32, shape=(None, self.maze.size), name="X1")
        self.reward = tf.placeholder(dtype=tf.float32, shape=None, name="reward")
        self.q1, _ = self.build_model(input=self.X1, name="trained", reuse=True)
        self.covariance = tf.Variable(np.eye(self.theta_size), name="covariance")
        q_selected = tf.reduce_sum(self.q * tf.one_hot(self.action, num_actions), 1)
        self.loss = .5 * tf.reduce_mean(tf.square(self.R - q_selected))
        self.new_params = [tf.placeholder(dtype=tf.float32, shape=var.shape) for var in self.params]
        self.update_params = [tf.assign(oldv, newv) for oldv, newv in zip(self.params, self.new_params)]

        self.trainer = KTDOptimizer(var_list=self.params, learning_rate=self.kalman_lr,
                                    gamma=self.gamma, theta_noise=self.theta_noise, reward_noise=self.reward_noise,
                                    P_init=self.P_init,
                                    sess=self.sess, model_name=self.optimizer_method,
                                    q_function0=self.q, q_function1=self.q1,
                                    state_ph=self.X, next_state_ph=self.X1,
                                    action_ph=self.action, reward_ph=self.reward,
                                    update_params=self.update_params, new_params=self.new_params)
        self.sess.run(tf.global_variables_initializer())
        self.train_op = self.trainer.apply_kalman

    def fit(self, states, targets, actions, **kwargs):
        self.calls_to_fit += 1
        next_states = kwargs["next_states"]
        rewards = kwargs["rewards"]
        self.trainer.apply_kalman(states, actions, next_states, rewards)

    def assign_params(self, new_params_and_vars):
        new_params, var = zip(*new_params_and_vars)
        td_map = {self.new_params[i]: new_params[i] for i in range(len(new_params))}

        return self.sess.run(self.update_params, td_map)


def show(qmaze):
    final_data_path, exp_args_vals, arguments_names = arrange_data_to_plot()
    with PdfPages(final_data_path[0] + '/maze_img.pdf') as pdf:
        plt.figure()
        plt.grid(True)
        nrows, ncols = qmaze.maze.shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = np.copy(qmaze.maze)
        for row, col in qmaze.visited:
            canvas[row, col] = 0.6
        rat_row, rat_col, _ = qmaze.state

        mouse_path = os.getcwd() + "/figs/emoji.png"
        fn = matplotlib.cbook.get_sample_data(mouse_path, asfileobj=False)
        arr_img = plt.imread(fn, format='png')

        imagebox = OffsetImage(arr_img, zoom=0.12)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, [rat_row, rat_col])

        ax.add_artist(ab)

        canvas[rat_row, rat_col] = 0.3  # rat cell
        canvas[nrows-1, ncols-1] = 0.9  # cheese cell
        img = plt.imshow(canvas, interpolation='none', cmap='gray')

        plt.text(nrows, ncols - 1.1, 'Exit',
                 horizontalalignment='center',
                 verticalalignment='top',
                 multialignment='center', color='red', fontsize=30)
        pdf.savefig(dpi=200)
        plt.show()

    return img


def play_game(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action

        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True


def read_from_file_for_plotting(dir):
    # reads data from file and returns a list of dictionaries.
    # Each dictionary contains data for plotting for a specific state

    with open(dir + '/data.txt') as json_file:
        data = json.load(json_file)

    return data

def calls_to_predict_experiment():
    final_data_path, exp_args_vals, arguments_names = arrange_data_to_plot()
    data_kova, data_ktd, data_kova_4x4, data_ktd_4x4 = [], [], [], []

    for i, path in enumerate(final_data_path):
        if exp_args_vals[i]["comb"] in (['19', '20', '21', '22']):  # these are the dirs for KOVA
            data_kova.append(read_from_file_for_plotting(path))
            if exp_args_vals[i]["comb"] == '20':  # this is 4x4 maze for kova
                data_kova_4x4.append(read_from_file_for_plotting(path))
        elif exp_args_vals[i]["comb"] in (['7', '8', '9', '10']):
            data_ktd.append(read_from_file_for_plotting(path))
            if exp_args_vals[i]["comb"] == '8':  # this is 4x4 maze for ktd
                data_ktd_4x4.append(read_from_file_for_plotting(path))

    with PdfPages(final_data_path[0] + '/calls_to_predict.pdf') as pdf:
        figsize = (14., 8.)

        margins = {
            "left": 1.0 / figsize[0],
            "bottom": 1.9 / figsize[1],
            "right": 0.85,
            "top": 0.99,
            "hspace": 0.2
        }
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        fig.subplots_adjust(**margins)

        colors_p = ["blue", "red", "green", "orange", 'purple']
        plot_results(data_kova_4x4, color=colors_p[0], axs=axs[:2], line='-')
        plot_results(data_ktd_4x4, color=colors_p[1], axs=axs[:2], line='-')

        d = 2000
        x = [i for i in range(d)]
        y_ktd = [4*i for i in range(d)]
        y_kova = [2 for i in range(d)]
        axs[2]. plot(x, y_kova, color=colors_p[0], label="KOVA", linewidth=3)
        axs[2].plot(x, y_ktd, color=colors_p[1], label="KTD", linewidth=3)
        axs[2].text(1300, 900, 'const=2', horizontalalignment='center', verticalalignment='top',
                    multialignment='center', color=colors_p[0], fontsize=26)

        axs[2].text(1000, 5500, '4d', rotation=45, horizontalalignment='center', verticalalignment='top',
                    multialignment='center', color=colors_p[1], fontsize=26)

        axs[2].set_xlabel(r'd ($\theta$ dimension)' + '\n(c)', fontsize=26)
        axs[2].set_ylabel('# of feed-forward passes', fontsize=30, labelpad=5)
        axs[2].tick_params(axis='both', which='major', labelsize=26)
        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), fancybox=True, shadow=True, ncol=2, fontsize=30)
        plt.tight_layout()
        pdf.savefig(dpi=200)
        plt.show()


def plot_results_for_maze_exp(data, color=None, axs=None, label=None, line='-'):
    win_rate, timesteps = [], []
    for d in data:
        win_rate.append(d["win_rate"])
        timesteps.append(d["timesteps"])
    win_rate = np.array(win_rate)
    timesteps = np.array(timesteps)

    axs.plot(timesteps[0, :], np.mean(win_rate, axis=0), line, label=label, color=color, linewidth=2)
    axs.fill_between(timesteps[0, :], np.mean(win_rate, axis=0) + np.std(win_rate, axis=0),
                     np.mean(win_rate, axis=0) - np.std(win_rate, axis=0), alpha=0.2, facecolor=color)

    base = 100
    t_shape = timesteps.shape
    xticks = [0, timesteps[0, t_shape[1]//2], timesteps[0, -1]]
    xticklabels = [0, timesteps[0, timesteps.shape[1] // 2], int(base * round(timesteps[0, -1] / base))]
    axs.set_xticks(xticks)
    axs.set_xticklabels(xticklabels, fontsize=16)


def plot_results(data, color=None, axs=None, line=None):
    win_rate, time, calls_to_predict, timesteps, game_reward = [], [], [], [], []
    for d in data:
        win_rate.append(d["win_rate"])
        time.append(d["time_so_far"])
        calls_to_predict.append(d["calls_to_predict"])
        timesteps.append(d["timesteps"])
        game_reward.append(d["game_reward"])
    win_rate = np.array(win_rate)
    time = np.array(time)
    timesteps = np.array(timesteps)
    optimizer = data[0]["optimizer"]

    label = optimizer.upper()
    axs[0].plot(timesteps[0, :], np.mean(win_rate, axis=0), line, label=label, color=color, linewidth=3)
    axs[0].fill_between(timesteps[0, :], np.mean(win_rate, axis=0) + np.std(win_rate, axis=0),
                        np.mean(win_rate, axis=0) - np.std(win_rate, axis=0), alpha=0.2, facecolor=color)

    axs[1].semilogy(timesteps[0, :], np.mean(time, axis=0), label=label, color=color, linewidth=3)
    axs[1].fill_between(timesteps[0, :], np.mean(time, axis=0) + np.std(time, axis=0),
                        np.mean(time, axis=0) - np.std(time, axis=0), alpha=0.2, facecolor=color)

    axs[0].set_ylabel('Success rate', fontsize=34, labelpad=5)
    axs[1].set_ylabel('Training time (sec)', fontsize=34, labelpad=5)
    axs[0].tick_params(axis='both', which='both', labelsize=26)

    axs[0].set_xlabel('Timesteps\n(a)', fontsize=30)
    axs[1].set_xlabel('Timesteps\n(b)', fontsize=30)
    axs[1].tick_params(axis='both', which='both', labelsize=26)

    axs[0].set_yticks([0, 0.5, 1])
    axs[0].set_yticklabels([0, 0.5, 1])


def maze_experiment():
    final_data_path, exp_args_vals, arguments_names = arrange_data_to_plot()
    data_kova_01, data_kova_001, data_kova_0001 = [], [], []
    data_adam, data_adam_decay, data_adam_00001, data_adam_000001 = [], [], [], []

    for i, path in enumerate(final_data_path):
        if exp_args_vals[i]["comb"] == '0':  # adam, adam_lr = 0.001
            data_adam.append(read_from_file_for_plotting(path))
            adam_label = r'ADAM ($\alpha=$' + exp_args_vals[i]["adam_lr"] + ')'
        elif exp_args_vals[i]["comb"] == '17':  # adam, adam_lr = 3e-4 decaying
            data_adam_decay.append(read_from_file_for_plotting(path))
            adam_decay_label = r'ADAM ($\alpha=$' + exp_args_vals[i]["adam_lr"] + ' decaying)'
        elif exp_args_vals[i]["comb"] == '18':  # adam, adam_lr = 0.0001
            data_adam_00001.append(read_from_file_for_plotting(path))
            adam_00001_label = r'ADAM ($\alpha=$' + exp_args_vals[i]["adam_lr"] + ')'
        elif exp_args_vals[i]["comb"] == '15':  # kova, eta=0.1
            data_kova_01.append(read_from_file_for_plotting(path))
            kova_01_label = r'KOVA ($\eta=$' + exp_args_vals[i]["eta"] + ')'
        elif exp_args_vals[i]["comb"] == '1':  # kova, eta=0.01
            data_kova_001.append(read_from_file_for_plotting(path))
            kova_001_label = r'KOVA ($\eta=$' + exp_args_vals[i]["eta"] + ')'
        elif exp_args_vals[i]["comb"] == '16':  # kova, eta=0.001
            data_kova_0001.append(read_from_file_for_plotting(path))
            kova_0001_label = r'KOVA ($\eta=$' + exp_args_vals[i]["eta"] + ')'

    with PdfPages(final_data_path[0] + '/maze_exp.pdf') as pdf:
        colors_p = ["blue", "red", "green", "cyan", 'purple', 'coral', 'orange']
        figsize = (7, 6)
        margins = {
            "left": 1.0 / figsize[0],
            "bottom": 1.9 / figsize[1],
            "right": 0.85,
            "top": 0.99,
            "hspace": 0.2
        }
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        fig.subplots_adjust(**margins)

        plot_results_for_maze_exp(data_kova_01, color=colors_p[0], axs=axs, label=kova_01_label, line='-')
        plot_results_for_maze_exp(data_kova_001, color=colors_p[1], axs=axs, label=kova_001_label, line='-')
        plot_results_for_maze_exp(data_kova_0001, color=colors_p[2], axs=axs, label=kova_0001_label, line='-')
        plot_results_for_maze_exp(data_adam, color=colors_p[3], axs=axs, label=adam_label, line='--')
        plot_results_for_maze_exp(data_adam_decay, color=colors_p[4], axs=axs, label=adam_decay_label, line='--')
        plot_results_for_maze_exp(data_adam_00001, color=colors_p[5], axs=axs, label=adam_00001_label, line='--')

        axs.set_ylabel('Success rate', fontsize=20)
        axs.tick_params(axis='y', which='both', labelsize=16)
        axs.set_xlabel('Timesteps', fontsize=20)
        axs.set_yticks([0, 0.5, 1])
        axs.set_yticklabels([0, 0.5, 1])
        axs.set_ylim([0, 1.05])
        axs.set_xlim([0, 100000])

        axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=15)
        pdf.savefig(dpi=200)
        plt.show()


def arrange_data_to_plot():
    data_dir = './results'
    subdirs = os.listdir(data_dir)
    print("subdirs", type(subdirs), len(subdirs), subdirs)
    final_data_path = [os.path.join(data_dir, d) for d in subdirs]
    print("final_data_path", type(final_data_path), len(final_data_path), final_data_path)
    parse_dir = [string.split('_') for string in subdirs]
    arguments_names = ["exp", "date", "env", "pol_alg", "vf_alg", "kalman_lr", "onv_coeff", "eta",
                       "onv_type", "batch_size", "adam_lr", "gamma", "total_timesteps", "seed", "comb", "last_layer"]

    exp_args_vals = [dict(zip(arguments_names, p)) for p in parse_dir]
    print("exp_args_vals", len(exp_args_vals))
    print("parse_dir", len(parse_dir), parse_dir)

    return final_data_path, exp_args_vals, arguments_names


def maze():
    # 10x10 maze
    maze = np.array([
        [1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
        [1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
        [0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.],
        [1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
        [1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
        [1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
        [1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
        [1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]
    ])

    # 4x4 maze:
    maze = np.array([
        [1., 0., 1., 1.],
        [1., 1., 1., 0.],
        [0., 0., 1., 0.],
        [1., 1., 1., 1.],
    ])

    qmaze = Qmaze(maze)
    show(qmaze)

    optimizer_method = 'kova'
    data_size = 32
    gamma = 0.95
    comb_num = 26

    # arguments for KOVA:
    kalman_lr = 1.
    onv_coeff = 1.
    eta = 0.01
    onv_type = 'batch-size'
    kalman_only_last_layer = False

    # arguments for Adam
    adam_lr = 0.0001
    adam_lr_func = lambda f: adam_lr * f

    # arguments for KTD:
    theta_noise = 0.01
    reward_noise = 1.
    P_init = 10

    total_timesteps = 80000
    seed = random.randint(0, 100)

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    exp_dir = os.getcwd()
    print("exp_dir", exp_dir)
    curdir = logger.get_dir()
    print("curdir", curdir)

    exp_name = "/results/exp_{}_maze{}_pol-{}_vf-{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_last-{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
        maze.shape,
        'doubleQlearning',
        optimizer_method,
        kalman_lr,
        onv_coeff,
        eta,
        onv_type,
        data_size,
        adam_lr,
        gamma,
        total_timesteps,
        seed,
        comb_num,
        kalman_only_last_layer)

    print("exp_name", exp_name)

    if optimizer_method == 'ktd':
        model = ModelKTD(maze, optimizer_method=optimizer_method, kalman_lr=kalman_lr,
                         onv_coeff=onv_coeff, eta=eta, onv_type=onv_type, batch_size=data_size,
                         theta_noise=theta_noise, reward_noise=reward_noise, P_init=P_init)
    else:
        model = Model(maze, optimizer_method=optimizer_method, kalman_lr=kalman_lr,
                      onv_coeff=onv_coeff, eta=eta, onv_type=onv_type, batch_size=data_size,
                      adam_lr=adam_lr)

    qtrain(model, maze, total_timesteps=total_timesteps, epochs=1000, max_memory=8 * maze.size, data_size=data_size,
           gamma=gamma)
    shutil.copytree(curdir, exp_dir + exp_name)
    print("We copied to: ", exp_dir, exp_name)


if __name__ == '__main__':
    maze()
    calls_to_predict_experiment()
    maze_experiment()
