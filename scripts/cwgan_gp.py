# scripts/cwgan_gp.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

tf.random.set_seed(42)
np.random.seed(42)


class GumbelSoftmax(layers.Layer):
    """Gumbel-Softmax layer for differentiable categorical sampling."""
    def __init__(self, temperature=1.0, hard=False, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.hard = hard

    def call(self, inputs, training=None):
        if training:
            u = tf.random.uniform(tf.shape(inputs), minval=0, maxval=1)
            g = -tf.math.log(-tf.math.log(u + 1e-20) + 1e-20)
            y = tf.nn.softmax((inputs + g) / self.temperature)
            if self.hard:
                y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=-1, keepdims=True)), y.dtype)
                y = tf.stop_gradient(y_hard - y) + y
        else:
            y = tf.nn.softmax(inputs)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({'temperature': self.temperature, 'hard': self.hard})
        return config


class CrossLayer(layers.Layer):
    """
    Cross layer for explicit feature interaction.
    Implements: x_{l+1} = x_0 * (w^T * x_l) + b + x_l
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[1][-1]
        self.w = self.add_weight(
            shape=(feature_dim, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='cross_weight'
        )
        self.b = self.add_weight(
            shape=(feature_dim,),
            initializer='zeros',
            trainable=True,
            name='cross_bias'
        )
        super().build(input_shape)

    def call(self, inputs):
        x0, xl = inputs
        return x0 * tf.matmul(xl, self.w) + self.b + xl

    def get_config(self):
        return super().get_config()


# Helper: safely convert anything to tuple
def _to_tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)


class CWGANGP:
    def __init__(self, n_num, n_cat_dims, latent_dim=256, cond_dim=1,
                 gen_hidden=[256, 256], critic_hidden=[256, 256],
                 gp_weight=10.0, aux_weight=1.0, learning_rate=1e-4,
                 gumbel_temperature=0.5, use_cross_layers=True):
        self.n_num = n_num
        self.n_cat_dims = n_cat_dims
        self.n_cat_groups = len(n_cat_dims)
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.gp_weight = gp_weight
        self.aux_weight = aux_weight
        self.gumbel_temperature = gumbel_temperature
        self.use_cross_layers = use_cross_layers

        self.generator = self._build_generator(gen_hidden)
        self.critic = self._build_critic(critic_hidden)

        self.gen_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.9)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.9)

        self.g_losses = []
        self.c_losses = []
        self.w_distances = []
        self.gp_vals = []

    def _build_generator(self, hidden_units):
        noise = layers.Input(shape=(self.latent_dim,))
        cond = layers.Input(shape=(self.cond_dim,))
        x = layers.Concatenate()([noise, cond])

        for units in hidden_units:
            x = layers.Dense(units)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

        outputs = []
        if self.n_num > 0:
            num_out = layers.Dense(self.n_num, name='numerical')(x)
            num_out = layers.Activation('sigmoid')(num_out)
            outputs.append(num_out)

        for i, dim in enumerate(self.n_cat_dims):
            cat_out = layers.Dense(dim, name=f'cat_{i}')(x)
            cat_out = GumbelSoftmax(temperature=self.gumbel_temperature, hard=False)(cat_out)
            outputs.append(cat_out)

        model = keras.Model(inputs=[noise, cond], outputs=outputs, name='Generator')
        return model

    def _build_critic(self, hidden_units):
        inputs = []
        if self.n_num > 0:
            num_input = layers.Input(shape=(self.n_num,))
            inputs.append(num_input)

        cat_inputs = []
        for i, dim in enumerate(self.n_cat_dims):
            cat_in = layers.Input(shape=(dim,))
            cat_inputs.append(cat_in)
            inputs.append(cat_in)

        cond = layers.Input(shape=(self.cond_dim,))
        inputs.append(cond)

        concat_list = []
        if self.n_num > 0:
            concat_list.append(inputs[0])
        for cat_in in cat_inputs:
            concat_list.append(cat_in)
        concat_list.append(cond)
        x = layers.Concatenate()(concat_list)

        x0 = x

        for units in hidden_units:
            x = layers.Dense(units)(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = layers.Dropout(0.1)(x)

        if self.use_cross_layers:
            x0_projected = layers.Dense(hidden_units[-1], use_bias=False, name='cross_projection')(x0)
            x_cross = CrossLayer()([x0_projected, x])
            x = layers.Add()([x, x_cross])

        score = layers.Dense(1, name='critic_score')(x)
        aux = layers.Dense(1, activation='sigmoid', name='aux_classifier')(x)

        model = keras.Model(inputs=inputs, outputs=[score, aux], name='Critic')
        return model

    def gradient_penalty(self, real_data, fake_data, cond):
        batch_size = tf.shape(cond)[0]
        real_data = _to_tuple(real_data)
        fake_data = _to_tuple(fake_data)

        alpha = tf.random.uniform([batch_size, 1], 0., 1.)
        interpolated_data = []
        for real, fake in zip(real_data, fake_data):
            alpha_expanded = tf.reshape(alpha, [batch_size] + [1] * (len(real.shape) - 1))
            interpolated = alpha_expanded * real + (1 - alpha_expanded) * fake
            interpolated_data.append(interpolated)

        interpolated_inputs = tuple(interpolated_data) + (cond,)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_inputs)
            critic_out, _ = self.critic(interpolated_inputs, training=True)

        grads = gp_tape.gradient(critic_out, interpolated_inputs[:-1])
        grad_norms = []
        for grad in grads:
            grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=list(range(1, len(grad.shape)))))
            grad_norms.append(grad_norm)
        grad_norm = tf.reduce_mean(tf.stack(grad_norms, axis=0), axis=0)
        gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, real_data, cond, real_labels):
        batch_size = tf.shape(cond)[0]

        # --- Train Critic ---
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as critic_tape:
            fake_data_raw = self.generator([noise, cond], training=True)
            fake_data = _to_tuple(fake_data_raw)
            real_data_tuple = _to_tuple(real_data)

            real_inputs = real_data_tuple + (cond,)
            fake_inputs = fake_data + (cond,)

            real_score, real_aux = self.critic(real_inputs, training=True)
            fake_score, fake_aux = self.critic(fake_inputs, training=True)

            c_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
            gp = self.gradient_penalty(real_data_tuple, fake_data, cond)
            aux_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_aux)
            aux_loss_fake = tf.keras.losses.binary_crossentropy(cond, fake_aux)
            aux_loss = tf.reduce_mean(aux_loss_real) + tf.reduce_mean(aux_loss_fake)

            c_loss_total = c_loss + self.gp_weight * gp + self.aux_weight * aux_loss

        critic_grads = critic_tape.gradient(c_loss_total, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # --- Train Generator ---
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as gen_tape:
            fake_data_raw = self.generator([noise, cond], training=True)
            fake_data = _to_tuple(fake_data_raw)
            fake_inputs = fake_data + (cond,)
            fake_score, fake_aux = self.critic(fake_inputs, training=True)

            g_loss = -tf.reduce_mean(fake_score)
            aux_loss_g = tf.reduce_mean(tf.keras.losses.binary_crossentropy(cond, fake_aux))
            g_total_loss = g_loss + self.aux_weight * aux_loss_g

        gen_grads = gen_tape.gradient(g_total_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        w_dist = tf.reduce_mean(real_score) - tf.reduce_mean(fake_score)
        return c_loss, g_loss, w_dist, gp

    def train(self, dataset, epochs, n_critic=5, verbose=True, early_stopping_patience=50):
        best_w_dist = -np.inf
        patience_counter = 0

        for epoch in range(epochs):
            epoch_c_loss = 0.0
            epoch_g_loss = 0.0
            epoch_w_dist = 0.0
            epoch_gp = 0.0
            n_batches = 0

            for real_data, cond, labels in dataset:
                for _ in range(n_critic):
                    c_loss, g_loss, w_dist, gp = self.train_step(real_data, cond, labels)

                epoch_c_loss += c_loss.numpy()
                epoch_g_loss += g_loss.numpy()
                epoch_w_dist += w_dist.numpy()
                epoch_gp += gp.numpy()
                n_batches += 1

            epoch_c_loss /= n_batches
            epoch_g_loss /= n_batches
            epoch_w_dist /= n_batches
            epoch_gp /= n_batches

            self.c_losses.append(epoch_c_loss)
            self.g_losses.append(epoch_g_loss)
            self.w_distances.append(epoch_w_dist)
            self.gp_vals.append(epoch_gp)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | C Loss: {epoch_c_loss:.4f} | G Loss: {epoch_g_loss:.4f} | "
                      f"W Dist: {epoch_w_dist:.4f} | GP: {epoch_gp:.4f}")

            if epoch_w_dist > best_w_dist:
                best_w_dist = epoch_w_dist
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(Wasserstein distance did not improve for {early_stopping_patience} epochs)")
                break

        print(f"\nTraining completed! Best Wasserstein Distance: {best_w_dist:.4f}")

    def generate(self, cond, batch_size=32):
        n_samples = cond.shape[0]
        all_num = []
        all_cat = [[] for _ in range(self.n_cat_groups)]

        for i in range(0, n_samples, batch_size):
            batch_cond_np = cond[i:i+batch_size]
            current_batch_size = batch_cond_np.shape[0]
            noise = tf.random.normal([current_batch_size, self.latent_dim])
            batch_cond_tf = tf.convert_to_tensor(batch_cond_np, dtype=tf.float32)

            fake_data = self.generator([noise, batch_cond_tf], training=False)

            if self.n_num > 0:
                all_num.append(fake_data[0].numpy())
                cat_start_idx = 1
            else:
                cat_start_idx = 0

            for j in range(self.n_cat_groups):
                probs = fake_data[cat_start_idx + j].numpy()
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(current_batch_size), np.argmax(probs, axis=1)] = 1.0
                all_cat[j].append(one_hot)

        if self.n_num > 0:
            num_out = np.vstack(all_num)
        else:
            num_out = None

        cat_out = [np.vstack(arr_list) for arr_list in all_cat]
        return num_out, cat_out

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.generator.save(os.path.join(path, 'generator.keras'))
        self.critic.save(os.path.join(path, 'critic.keras'))
        meta = {
            'n_num': self.n_num,
            'n_cat_dims': self.n_cat_dims,
            'latent_dim': self.latent_dim,
            'cond_dim': self.cond_dim,
            'gp_weight': self.gp_weight,
            'aux_weight': self.aux_weight,
            'gumbel_temperature': self.gumbel_temperature,
            'use_cross_layers': self.use_cross_layers
        }
        joblib.dump(meta, os.path.join(path, 'meta.pkl'))
        losses = {
            'c_losses': self.c_losses,
            'g_losses': self.g_losses,
            'w_distances': self.w_distances,
            'gp_vals': self.gp_vals
        }
        joblib.dump(losses, os.path.join(path, 'losses.pkl'))

    @classmethod
    def load(cls, path):
        meta = joblib.load(os.path.join(path, 'meta.pkl'))
        gan = cls(
            n_num=meta['n_num'],
            n_cat_dims=meta['n_cat_dims'],
            latent_dim=meta['latent_dim'],
            cond_dim=meta['cond_dim'],
            gp_weight=meta['gp_weight'],
            aux_weight=meta.get('aux_weight', 1.0),
            gumbel_temperature=meta.get('gumbel_temperature', 0.5),
            use_cross_layers=meta.get('use_cross_layers', True)
        )
        gan.generator = keras.models.load_model(
            os.path.join(path, 'generator.keras'),
            custom_objects={'GumbelSoftmax': GumbelSoftmax, 'CrossLayer': CrossLayer}
        )
        gan.critic = keras.models.load_model(
            os.path.join(path, 'critic.keras'),
            custom_objects={'CrossLayer': CrossLayer}
        )
        losses = joblib.load(os.path.join(path, 'losses.pkl'))
        gan.c_losses = losses['c_losses']
        gan.g_losses = losses['g_losses']
        gan.w_distances = losses['w_distances']
        gan.gp_vals = losses['gp_vals']
        return gan