import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


class SFMcell(layers.Layer):
    def __init__(
            self, 
            num_states=8, 
            num_freq=16,
    ):
        super(SFMcell, self).__init__()
        self.num_freq = num_freq
        self.f_ste = layers.Dense(num_states, activation='sigmoid')
        self.f_freq = layers.Dense(num_freq, activation='sigmoid')
        self.i_dot = layers.Dense(num_states, activation='sigmoid')
        self.c_wave = layers.Dense(num_states, activation='tanh')
                
        self.u_a = self.add_weight('u_a', [num_freq])
        self.b_a = self.add_weight('b_a', [num_states])
        self.plus = layers.Add()
        
        self.o_i = layers.Dense(num_states, activation='sigmoid')
        
    def call(self, x_i, h_i, S_i, t):
        omega_t = tf.cast(tf.stack([
            np.sin(np.arange(self.num_freq) * t % self.num_freq / (self.num_freq / np.pi / 2.)),
            np.cos(np.arange(self.num_freq) * t % self.num_freq / (self.num_freq / np.pi / 2.)),
        ])[:, None], tf.float32)
        
        x_h = tf.concat([x_i, h_i], -1)
        f_ste = self.f_ste(x_h)
        f_freq = self.f_freq(x_h)
        F_t = tf.einsum('...i, ...j -> ...ij', f_ste, f_freq)
        
        i_dot = self.i_dot(x_h)
        c_wave = self.c_wave(x_h)
        i_c_omega = (i_dot * c_wave)[:, None, :, None] * omega_t
        
        S_next = S_i * F_t[:, None] + i_c_omega
        
        A_i = self.plus(tf.unstack(S_next ** 2, axis=1)) ** (1/2)
        c_t = tf.math.sigmoid(tf.einsum('...ij, j -> ...i', A_i, self.u_a) + self.b_a)
        o_i = self.o_i(tf.concat([x_h, c_t], -1))
        h_next = o_i * c_t
        return h_next, S_next

    
class SFM(Model):
    def __init__(
            self, 
            num_states=8, 
            num_freq=16,
    ):
        super(SFM, self).__init__()
        self.num_states = num_states
        self.num_freq = num_freq
        
        self.h_0 = tf.zeros([num_states])[None]
        self.S_0 = tf.zeros([2, num_states, num_freq])[None]
        self.SFMcell = SFMcell(num_states=num_states, num_freq=num_freq)
        
    def call(self, inputs):
        t = inputs.shape[1]
        if t is None:
            c = inputs.shape[-1]
            return inputs @ tf.zeros([c, self.num_states])
        else:
            b = tf.shape(inputs)[0]
            
            x = tf.unstack(inputs, axis=1)
            h_0 = tf.tile(self.h_0, [b, 1])
            S_i = tf.tile(self.S_0, [b, 1, 1, 1])
            
            h = [h_0] + [None] * t
            
            for i, x_i in enumerate(x):
                h[i + 1], S_i = self.SFMcell(x_i, h[i], S_i, i)
                
            return tf.stack(h[1:], 1)
