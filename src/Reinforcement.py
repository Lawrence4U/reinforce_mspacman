#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pickle
import warnings
import tqdm
warnings.filterwarnings('ignore')
# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')


# In[3]:


##########  Using gym
import gym

# Scikit-Learn ≥0.20 is required
import sklearn

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

n_iterations = 120
n_episodes_per_update = 16
n_max_steps = 1000
discount_rate = 0.99


# MsPacman-ram-v5 environment
env = gym.make("ALE/MsPacman-ram-v5", render_mode='rgb_array')
# In[7]:
obs = env.reset()

# In[10]:


# Let's create a simple policy network with 9 output neurons (one per possible action):
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n

model = keras.models.Sequential([
    keras.layers.Dense(256, activation="relu", input_shape=[n_inputs]),
    keras.layers.Dense(128, activation="relu", input_shape=[n_inputs]),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(n_outputs, activation="softmax"),
])


# In[11]:


def pacman_play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        probas = model(obs[np.newaxis])
        logits = tf.math.log(probas + keras.backend.epsilon())
        action = tf.random.categorical(logits, num_samples=1)
        loss = tf.reduce_mean(loss_fn(action, probas))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, tr,info = env.step(action[0, 0].numpy())
    return obs, reward, done, grads

def pacman_play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        obs=obs[0]
        for step in range(n_max_steps):
            obs, reward, done, grads = pacman_play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads


# In[12]:


def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]


# In[16]:




# In[17]:


optimizer = keras.optimizers.Nadam(learning_rate=0.005)
loss_fn = keras.losses.sparse_categorical_crossentropy


# In[18]:


#env.seed(42)

mean_rewards = []


for iteration in range(n_iterations):
    all_rewards, all_grads = pacman_play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model, loss_fn)
    mean_reward = sum(map(sum, all_rewards)) / n_episodes_per_update
    print("\rIteration: {}/{}, mean reward: {:.1f}  ".format(
        iteration + 1, n_iterations, mean_reward), end="")
    mean_rewards.append(mean_reward)
    all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                       discount_rate)
    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index]
             for episode_index, final_rewards in enumerate(all_final_rewards)
                 for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_grads.append(mean_grads)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))


# In[ ]:


# Guardar la lista en un archivo de texto
# with open("mean_rewards_reinforcement.pkl", "wb") as file:
#     pickle.dumps(mean_reward, file)


# In[ ]:

# In[ ]:


import matplotlib.pyplot as plt

# plt.plot(mean_rewards)
# plt.xlabel("Episode")
# plt.ylabel("Mean reward")
# plt.grid()
# plt.show()


# In[ ]:
test_rewards = []

def pacman_render_policy_net(model, n_max_steps=500, seed=42):
    frames = []
    env = gym.make("ALE/MsPacman-ram-v5")
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    for episode in tqdm(range(10)):
        for step in range(n_max_steps):
            frames.append(env.render(mode="rgb_array"))
            probas = model(obs[np.newaxis])
            logits = tf.math.log(probas + keras.backend.epsilon())
            action = tf.random.categorical(logits, num_samples=1)
            obs, reward, done, info = env.step(action[0, 0].numpy())
            test_rewards.append(reward)
            if done:
                break
    env.close()
    return frames

# In[ ]:


# Now show the animation:

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim


# In[ ]:

print("rendering after train")
frames = pacman_render_policy_net(model, seed=42)
with open("frames_reinforce.pkl", 'wb') as f:
    pickle.dumps(frames, f)
with open("rewards_reinforce.pkl", 'wb') as f:
    pickle.dumps(test_rewards, f)
# plot_animation(frames)

