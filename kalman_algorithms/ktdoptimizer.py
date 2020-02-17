"""KTDOptimizer for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

import tensorflow as tf
import numpy as np
import baselines
from baselines.common import tf_util as U

class KTDOptimizer(optimizer.Optimizer):
    """Optimizer that implements the KTD algorithm from the paper
    Kalman Temporal Differences
    Matthieu Geist and Olivier Pietquin
    2010

    https://arxiv.org/pdf/1406.3270.pdf
    """

    def __init__(self, var_list=[], learning_rate=1., gamma=0.9, theta_noise=0.01, reward_noise=1., P_init=10,
                 kappa=3, use_locking=False, sess=None, model_name=None, q_function0=None, q_function1=None,
                 state_ph=None, next_state_ph=None, action_ph=None, reward_ph=None, update_params=None, new_params=None,
                 name="KTD"):
        """Construct a new KTD optimizer.
        """
        super(KTDOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._learning_rate_tensor = None

        self.var_list = var_list
        self.eps = 0.00001
        self.kappa = kappa
        self.P_init = P_init
        self.reward_noise = reward_noise
        self.theta_noise = theta_noise
        self.gamma = gamma

        size = sum(U.numel(v) for v in var_list)
        print("little size", [U.numel(v) for v in var_list])
        print("size", size)
        self.size = size
        self.t = 0
        self.sess = sess
        self.model_name = model_name
        self.p_hat_predicted = None

        self.q_function0 = q_function0
        self.q_function1 = q_function1
        self.state_ph = state_ph
        self.next_state_ph = next_state_ph
        self.action_ph = action_ph
        self.reward_ph = reward_ph
        self.update_params = update_params
        self.new_params = new_params
        self.calls_to_predict = 0

        # maintaining numpy copy of parameters and covariance in order to make computations through numpy
        self.theta = np.zeros((size, 1))
        self.covariance = np.eye(size, dtype='float32') * self.P_init

    def _prepare(self):
        if not context.executing_eagerly() or self._learning_rate_tensor is None:
            self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate, name="learning_rate")

    def _apply_dense(self, grad, var):
        return training_ops.apply_gradient_descent(
            var,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking).op

    def apply_kalman(self, states, actions, next_states, rewards):
        # Prediction step
        theta_prev = np.squeeze(self.theta)
        self.p_hat_predicted = np.multiply(self.covariance, (1 + self.theta_noise))
        X_sigma_points, W = self.sample_sigma_points(theta_prev, self.p_hat_predicted, method="np")
        X_sigma_points = X_sigma_points.astype(np.float32)
        W = W.astype(np.float32)
        n = X_sigma_points.shape[1]
        R_list = []
        for j in range(n):
            sampled_params_and_vars = self.convert_vector_to_variables(vector=X_sigma_points[:, j],
                                                                       list_of_var=self.var_list,
                                                                       method='np')
            new_params, var = zip(*sampled_params_and_vars)
            td_map = {self.new_params[i]: new_params[i] for i in range(len(new_params))}
            self.sess.run(self.update_params, td_map)

            q0 = self.sess.run(self.q_function0, {self.state_ph: states})[:, actions]
            q1 = self.sess.run(self.q_function1, {self.next_state_ph: next_states})
            Q_sa = np.reshape(np.max(q1), newshape=(1, 1))
            self.calls_to_predict += 2

            R_list.append((q0 - self.gamma * Q_sa))

        R = np.concatenate(R_list, axis=0)
        rhat, P_theta_r, Pr = self.compute_statistics(R, W, X_sigma_points, theta_prev, method='np')
        K = np.multiply(P_theta_r, Pr ** (-1))

        weight_update = np.matmul(K, (rewards - rhat))
        self.theta = theta_prev + weight_update

        # assign new value of theta into var_list
        sampled_params_and_vars = self.convert_vector_to_variables(vector=self.theta,
                                                                   list_of_var=self.var_list,
                                                                   method='np')
        new_params, var = zip(*sampled_params_and_vars)
        td_map = {self.new_params[i]: new_params[i] for i in range(len(new_params))}

        self.sess.run(self.update_params, td_map)
        covariance_update = np.matmul(np.multiply(Pr, K), K.T)
        self.covariance = self.p_hat_predicted - covariance_update

    def prediction_step(self):
        # Prediction Step
        theta_prev = self.concatenate_tensors([self.var_list])
        self.p_hat_predicted = tf.multiply(self.covariance, 1 / (1 - self.eta))
        return theta_prev, self.p_hat_predicted

    def sample_sigma_points(self, mean, covariance, method='np'):
        n = int(mean.shape[0])
        sigma_points = None
        W = None
        if method == 'np':
            chol = np.linalg.cholesky((n + self.kappa) * covariance)
            sigma_points = np.zeros(shape=(n, 2 * n + 1))
            sigma_points[:, 0] = np.squeeze(mean)
            for j in range(n):
                sliced_chol = chol[:, j]
                sigma_points[:, j+1] = np.squeeze(mean + sliced_chol)
                sigma_points[:, j + n + 1] = np.squeeze(mean - sliced_chol)

            W_first = np.array([[self.kappa / (self.kappa + n)]])
            W_rest = np.ones(shape=(2 * n, 1)) * (1. / (2 * (self.kappa + n)))
            W = np.concatenate([W_first, W_rest], axis=0)

        elif method == 'tf':
            chol = tf.linalg.cholesky((n + self.kappa) * covariance)
            sigma_points_list = []
            sigma_points_list.append(mean)
            for j in range(n):
                sliced_chol = tf.expand_dims(chol[:, j], axis=1)
                sigma_points_list.append(mean + sliced_chol)
                sigma_points_list.append(mean - sliced_chol)

            sigma_points = tf.concat(sigma_points_list, axis=1)
            W_first = tf.constant((self.kappa / (self.kappa + n)), shape=(1, 1))
            W_rest = tf.ones(shape=(2 * n, 1)) * (1. / (2 * (self.kappa + n)))
            W = tf.concat([W_first, W_rest], axis=0)

        return sigma_points, W

    def compute_statistics(self, R, W, X, theta_mean, method='tf'):
        # Compute statistics of interest
        rhat, P_theta_r, Pr = None, None, None
        if method == 'np':
            theta_mean = np.expand_dims(theta_mean, axis=1)
            rhat = np.sum(np.multiply(W, R))
            r_diff = R - rhat
            P_theta_r = np.matmul(X-theta_mean, np.multiply(W, r_diff))
            Pr = max(np.sum(np.multiply(W, np.square(r_diff))), 10e-5) + self.reward_noise
            # ensure a minimum amount of noise to avoid numerical instabilities
        elif method == 'tf':
            theta_mean = tf.cast(tf.expand_dims(theta_mean, axis=1), dtype=tf.float32)
            rhat = tf.reduce_sum(tf.multiply(W, R))
            r_diff = R - rhat
            P_theta_r = tf.matmul(X - theta_mean, tf.multiply(W, r_diff))
            Pr = tf.maximum(tf.reduce_sum(tf.multiply(W, tf.square(r_diff))), 10e-5) + self.reward_noise
            # ensure a minimum amount of noise to avoid numerical instabilities
        return rhat, P_theta_r, Pr

    def concatenate_tensors(self, list_of_lists_of_tensors):
        final_concatenated_tensor = []
        for tensor_list in list_of_lists_of_tensors:
            concatenated_tensor = []
            for tensor in tensor_list:
                reshaped_weight = tf.reshape(tensor, (U.numel(tensor),))
                concatenated_tensor.append(reshaped_weight)
            concatenated_tensor = tf.concat(concatenated_tensor, 0)
            final_concatenated_tensor.append(tf.expand_dims(concatenated_tensor, 1))
        final_concatenated_tensor = tf.concat(final_concatenated_tensor, 1)

        return final_concatenated_tensor

    def convert_vector_to_variables(self, vector, list_of_var, method="tf"):
        index = 0
        var_list = []
        if method == "tf":
            for i, network_mat in enumerate(list_of_var):
                var_shape = network_mat.get_shape()
                num_weights_per_var = U.numel(network_mat)  # the number of weights per a variable
                new_ass_var = tf.gather(vector, tf.constant(
                    np.linspace(index, index + num_weights_per_var - 1, num_weights_per_var).astype(int)))
                new_mat = tf.reshape(new_ass_var, var_shape)  # changing to the dimension of the variable
                index = index + num_weights_per_var
                var_list.append((new_mat, network_mat))
        elif method == "np":
            if len(vector.shape) == 1:
                vector = np.expand_dims(vector, 1)
            for i, network_mat in enumerate(list_of_var):
                var_shape = network_mat.get_shape()
                num_weights_per_var = int(np.prod(network_mat.shape))  # the number of weights per a variable
                new_ass_var = vector[np.linspace(index, index + num_weights_per_var-1, num_weights_per_var).astype(int)]
                new_mat = np.reshape(new_ass_var, var_shape)  # changing to the dimension of the variable
                index = index + num_weights_per_var
                var_list.append((new_mat, network_mat))

        return var_list

    def convert_variables_to_vector(self, vars):
        concatenated_vars = []
        for v in vars:
            reshaped_var = np.reshape(v, (int(np.prod(v.shape)),))
            concatenated_vars.append(reshaped_var)
        final_concatenated_vector = np.concatenate(concatenated_vars, 0)
        return final_concatenated_vector

    def assign_params(self, new_params):
        self.sess.run([tf.assign(oldv, newv) for (oldv, newv) in zip(tf.trainable_variables(self.model_name),
                                                                     new_params)])
