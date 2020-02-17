"""KalmanOptimizer for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export

import tensorflow as tf
import numpy as np
from baselines.common import tf_util as U
import time


@tf_export("train.KalmanOptimizer")
class KalmanOptimizer(optimizer.Optimizer):
    """Optimizer that implements the KOVA algorithm.
    """

    def __init__(self, var_list=[], learning_rate=1., eta=0.01, onv_coeff=1., onv_type='batch-size',
                 use_locking=False, name="Kalman"):
        """Construct a new Kalman optimizer.
        """
        super(KalmanOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._learning_rate_tensor = None

        self.var_list = var_list
        self.eta = eta
        self.eps = 0.00001

        size = sum(U.numel(v) for v in var_list)
        print("size", size)
        self.size = size
        self.t = 0
        self.onv_coeff = tf.constant(onv_coeff, dtype='float32')
        self.observation_noise_var = self.onv_coeff
        self.onv_type = onv_type  # ONV=Observation Noise Var. this is a string that indicate
                                  # how to calculate self.observation_noise_var
        self.p_hat_predicted = None
        self.covariance = tf.eye(size, dtype='float32')

    def _prepare(self):
        if not context.executing_eagerly() or self._learning_rate_tensor is None:
            self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate, name="learning_rate")

    def _apply_dense(self, grad, var):
        return training_ops.apply_gradient_descent(
            var,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking).op

    def apply_kalman(self, innovation=None, ratio=None, q_function_gradient_list=[], list_of_vars=[], batch_size=32):
        self.batch_size = batch_size
        if self.onv_type == 'batch-size':
            self.observation_noise_var = self.onv_coeff * self.batch_size
        elif self.onv_type == "max-ratio":
            self.observation_noise_var = self.onv_coeff * tf.maximum(1., (1. / (ratio + self.eps))) * self.batch_size
        else:
            print("Please choose a valid onv_type")
            self.observation_noise_var = self.onv_coeff * self.batch_size # default is batch-size
            pass

        self.t += 1
        tsuperstart = time.time()
        q_function_gradient = - self.concatenate_tensors(q_function_gradient_list)

        self.p_hat_predicted = tf.multiply(self.covariance, 1 / (1 - self.eta))
        P_theta_r = tf.matmul(self.p_hat_predicted, q_function_gradient)

        self.gradq_P_gradq = tf.matmul(q_function_gradient, P_theta_r, transpose_a=True)
        self.P_r = tf.add(self.gradq_P_gradq, self.observation_noise_var * tf.eye(self.batch_size))
        self.K = tf.matmul(P_theta_r, tf.matrix_inverse(self.P_r))

        weight_update = tf.matmul(self.K, tf.expand_dims(innovation, 1))
        self.innovation = innovation
        self.KP = tf.matmul(self.K, self.P_r)
        self.KPK = self._learning_rate * tf.matmul(self.KP, self.K, transpose_b=True)
        self.covariance = tf.subtract(self.p_hat_predicted, self.KPK)
        gradients = self.convert_vector_to_variables(weight_update, list_of_vars)
        optimize_expr = self.apply_gradients(gradients)

        print(" total time is: %.3f seconds" % (time.time() - tsuperstart))
        return optimize_expr

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
                new_ass_var = vector[np.linspace(index, index + num_weights_per_var - 1, num_weights_per_var).astype(int)]
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
