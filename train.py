import gym
from stable_baselines3 import SAC
from stable_baselines3.common.envs import DummyVecEnv
from env.ac_env import AssettoCorsaEnv  # Il tuo ambiente personalizzato

# Crea l'ambiente
env = DummyVecEnv([lambda: AssettoCorsaEnv()])

# Crea l'agente SAC
model = SAC("CnnPolicy", env, verbose=1, tensorboard_log="./sac_tensorboard/")

# Allenamento dell'agente (ad esempio 100.000 timesteps)
model.learn(total_timesteps=100000)

# Salva il modello addestrato
model.save("sac_assetto_corsa_model")

# Testa il modello addestrato
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones[0]:
        break
