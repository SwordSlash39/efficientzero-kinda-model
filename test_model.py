import gymnasium as gym
import torch
import torch.nn.functional as F
from connect_four_gymnasium import ConnectFourEnv
import numpy as np
from main_model import MainModel
from main import change_observation, get_legal_mask

# Players
from connect_four_gymnasium.players import ChildPlayer, ChildSmarterPlayer, BabyPlayer, AdultPlayer, AdultSmarterPlayer, ConsolePlayer

mode = ["human", "rgb_array"]
env = ConnectFourEnv(render_mode=mode[0], opponent=ConsolePlayer())
observation, info = env.reset() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MainModel(
    dim1=6,
    dim2=7,
    actions=7,
    input_channels=3,
    latent_channels=64,
    device=device
).to(device)
model.load_checkpoint("bomb.pth")
print(device)
model.eval()

wins = 0
losses = 0

with torch.inference_mode():
    for k in range(200):
        curr_obv = change_observation(observation)
        legal_move_mask = get_legal_mask(observation)
        # curr_obv_tensor = torch.tensor(curr_obv, dtype=torch.float32, device=device).unsqueeze(0)
        # worldEncoded = model.eval_worldencoder(curr_obv_tensor)
        # policy_logits = model.eval_policy(worldEncoded)
        # policy_probs = torch.softmax(policy_logits, dim=-1).squeeze()
        
        # MCTS TS
        state = torch.tensor(curr_obv, dtype=torch.float32, device=device).unsqueeze(0)
        policy_out = model.parallel_mcts(state, training_mode=False, mcts_nodes=64).squeeze()
        policy_out[~legal_move_mask] = -float('inf')
        action = torch.argmax(policy_out, dim=-1).item()
        
        observation, reward, terminated, truncated, info = env.step(action)
        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()
            print(f"Game {k} done")
        
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
    print(wins / max(losses, 1))
    print(wins)
    print(losses)

