import gymnasium as gym
import torch
import torch.nn.functional as F
import random
from connect_four_gymnasium import ConnectFourEnv
import numpy as np
from main_model import MainModel
from replaybuffer import PriorityReplay
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create models
model = MainModel(
    dim1=6,
    dim2=7,
    actions=7,
    input_channels=3,
    latent_channels=64,
    device=device
).to(device)
replaybuffer = PriorityReplay(
    max_size=100_000,
    device=device,
    batch_size=256
)
K_rollout_replaybuffer = PriorityReplay(
    max_size=32_000,
    device=device,
    batch_size=256
)
mseloss_none = torch.nn.MSELoss(reduction='none')
crossentropyloss_none = torch.nn.CrossEntropyLoss(reduction='none')

# hyperparams
GAMMA = 0.99
K_ROLLOUT = 5
WORLDMODEL_LOSS = 1
ENDPRED_LOSS = 3
POLICY_LOSS = 0.4
VALUE_LOSS = 2
EPSILON = 0.05

env = ConnectFourEnv(render_mode="rgb_array")

observation, info = env.reset() 

def change_observation(obv):
    p1 = np.zeros((6, 7))
    p2 = np.zeros((6, 7))
    empty = np.zeros((6, 7))
    
    p1[obv == -1] = 1
    p2[obv == 1] = 1
    empty[obv == 0] = 1
    return np.stack([p1, p2, empty], axis=0)    # (3, 6, 7)

def get_legal_mask(obv):
    # check if obv[0] contains anyt
    obv_slice = obv[0]
    return np.where(obv_slice == 0, 1, 0).astype(bool)

def has_illegal_moves(obv):
    return (1 - get_legal_mask(obv)).any()

def get_random_illegal_move(obv):
    mask = get_legal_mask(obv)
    return random.choice(np.where(mask == 0)[0])
def get_random_legal_move(obv):
    mask = get_legal_mask(obv)
    return random.choice(np.where(mask == 1)[0])

def gen_rand_data(steps=256, epsilon=EPSILON):
    observation, info = env.reset() 
    
    print("Generating data...")
    # [steps, 3, 6, 7]
    # Temporary lists to hold current episode history
    episode_obs = []
    episode_actions = []
    episode_dones = []
    
    # Pre-process first observation
    curr_obv = change_observation(observation)
    episode_obs.append(curr_obv)
    for k in tqdm(range(steps)):
        action = get_random_legal_move(observation)
        if has_illegal_moves(observation):
            if random.random() < epsilon:
                action = get_random_illegal_move(observation)
        
        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminate xd or truncated
        observation, reward, terminated, truncated, info = env.step(action)
        
        replaybuffer.add_single_state({
            "observation": torch.tensor(curr_obv, dtype=torch.float32, device=device), 
            "action": torch.tensor(action, dtype=torch.long, device=device),
            "reward": torch.tensor(reward, dtype=torch.float32, device=device),
            "next_observation": torch.tensor(change_observation(observation), dtype=torch.float32, device=device),
            "done": torch.tensor(terminated, dtype=torch.bool, device=device),
        })
        
        # 4. Append to History
        episode_actions.append(action)
        episode_dones.append(1 if terminated else 0)
        episode_obs.append(change_observation(observation)) # obs list will be length N+1

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            # Conv list to np aray            
            np_obs = np.array(episode_obs)       # [T+1, 3, 6, 7]
            np_acts = np.array(episode_actions)  # [T]
            np_dones = np.array(episode_dones)   # [T]
            
            episode_len = len(np_acts)
            
            # Extract sliding windows of size K_ROLLOUT
            # We need K transitions, so we need K+1 observations
            if episode_len >= K_ROLLOUT:
                for i in range(episode_len - K_ROLLOUT + 1):
                    # Slice data
                    slice_obs = np_obs[i : i + K_ROLLOUT + 1] # shape [K+1, 3, 6, 7]
                    slice_acts = np_acts[i : i + K_ROLLOUT]   # shape [K]
                    slice_dones = np_dones[i : i + K_ROLLOUT] # shape [K]
                    
                    K_rollout_replaybuffer.add_single_state({
                        "observations": torch.tensor(slice_obs, dtype=torch.float32, device=device),
                        "actions": torch.tensor(slice_acts, dtype=torch.long, device=device),
                        "dones": torch.tensor(slice_dones, dtype=torch.long, device=device)
                    })

            # Reset environment and lists
            observation, info = env.reset()
            curr_obv = change_observation(observation)
            episode_obs = [curr_obv]
            episode_actions = []
            episode_dones = []

@torch.inference_mode()
def gen_data(steps=256):
    observation, info = env.reset() 
    
    print("Generating data...")
    # [steps, 3, 6, 7]
    # Temporary lists to hold current episode history
    episode_obs = []
    episode_actions = []
    episode_dones = []
    
    # Pre-process first observation
    curr_obv = change_observation(observation)
    episode_obs.append(curr_obv)
    for k in tqdm(range(steps)):
        # this is where you would insert your policy
        curr_obv = change_observation(observation)
        
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            legal_move_mask = get_legal_mask(observation)
            curr_obv_tensor = torch.tensor(curr_obv, dtype=torch.float32, device=device).unsqueeze(0)
            worldEncoded = model.eval_worldencoder(curr_obv_tensor)
            policy_logits = model.eval_policy(worldEncoded).squeeze()
            policy_logits[~legal_move_mask] = -float('inf')
            
            # Get probs and dist
            policy_probs = torch.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(policy_probs)
            
            action = dist.sample().item()
        
        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminate xd or truncated
        observation, reward, terminated, truncated, info = env.step(action)
        
        replaybuffer.add_single_state({
            "observation": torch.tensor(curr_obv, dtype=torch.float32, device=device), 
            "action": torch.tensor(action, dtype=torch.long, device=device),
            "reward": torch.tensor(reward, dtype=torch.float32, device=device),
            "next_observation": torch.tensor(change_observation(observation), dtype=torch.float32, device=device),
            "done": torch.tensor(terminated, dtype=torch.bool, device=device),
        })
        
        # 4. Append to History
        episode_actions.append(action)
        episode_dones.append(1 if terminated else 0)
        episode_obs.append(change_observation(observation)) # obs list will be length N+1

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            # Conv list to np aray            
            np_obs = np.array(episode_obs)       # [T+1, 3, 6, 7]
            np_acts = np.array(episode_actions)  # [T]
            np_dones = np.array(episode_dones)   # [T]
            
            episode_len = len(np_acts)
            
            # Extract sliding windows of size K_ROLLOUT
            # We need K transitions, so we need K+1 observations
            if episode_len >= K_ROLLOUT:
                for i in range(episode_len - K_ROLLOUT + 1):
                    # Slice data
                    slice_obs = np_obs[i : i + K_ROLLOUT + 1] # shape [K+1, 3, 6, 7]
                    slice_acts = np_acts[i : i + K_ROLLOUT]   # shape [K]
                    slice_dones = np_dones[i : i + K_ROLLOUT] # shape [K]
                    
                    K_rollout_replaybuffer.add_single_state({
                        "observations": torch.tensor(slice_obs, dtype=torch.float32, device=device),
                        "actions": torch.tensor(slice_acts, dtype=torch.long, device=device),
                        "dones": torch.tensor(slice_dones, dtype=torch.long, device=device)
                    })

            # Reset environment and lists
            observation, info = env.reset()
            curr_obv = change_observation(observation)
            episode_obs = [curr_obv]
            episode_actions = []
            episode_dones = []
            
def train_data(epochs=1, batch_size=256):
    print("Training...")
    running_loss = 0
    for e in tqdm(range(epochs)):
        # We have the data now, we js gotta sample
        batch_data = replaybuffer.sample(batch_size=batch_size)
        observation_data = batch_data["observation"]
        reward_data = batch_data["reward"]
        action_data = batch_data["action"]
        next_observation_data = batch_data["next_observation"]
        done_data = batch_data["done"]
        weights = batch_data["_weight"]
        
        batch_size = len(done_data)

        # Train worldencoder + latent2latent
        curr_latents = model.worldencoder_net(observation_data)
        next_latents = model.worldencoder_net(next_observation_data)
        latent2latent, end_pred = model.latent2latent_net(curr_latents, action_data)
        v1 = latent2latent.view(batch_size, -1)
        v2 = next_latents.view(batch_size, -1)
        loss_worldmodel_vec = 1.0 - F.cosine_similarity(v1, v2.detach())

        # Get end prediction head loss
        loss_endpred_vec = crossentropyloss_none(end_pred, done_data.long().detach())

        # get policy loss
        target_policy_output = model.parallel_mcts(observation_data, training_mode=True) # [batch, actions]
        policy_logits = model(curr_latents.detach())
        loss_policy_vec = crossentropyloss_none(policy_logits, target_policy_output.detach().clone())

        # get value loss
        value_output = model.value_net(curr_latents, action_data)
        next_observation_actions = torch.argmax(model.eval_policy(next_latents), dim=1) #[batch] of next actions
        # Negative cos 0 sum game
        target_value = reward_data + (~done_data) * GAMMA * -model.target_value_net(next_latents, next_observation_actions).squeeze()
        loss_value_vec = mseloss_none(value_output, target_value.unsqueeze(1).detach()).squeeze(-1)

        loss_vec = (
            WORLDMODEL_LOSS * loss_worldmodel_vec + 
            ENDPRED_LOSS * loss_endpred_vec + 
            POLICY_LOSS * loss_policy_vec + 
            VALUE_LOSS * loss_value_vec
        )
        loss = (weights * loss_vec).mean()
        
        # Addition K_ROLLOUT training
        # Only run this if we have enough data in the buffer
        seq_data = K_rollout_replaybuffer.sample()
        
        # Shapes: 
        # seq_obs: [Batch, K+1, C, H, W]
        # seq_act: [Batch, K]
        seq_obs = seq_data["observations"]
        seq_act = seq_data["actions"]
        seq_dones = seq_data["dones"]
        seq_weights = seq_data["_weight"]
        
        # A. Encode ALL observations at once (Ground Truths)
        # Flatten: [Batch * (K+1), C, H, W]
        batch_k, T_plus_1, C, H, W = seq_obs.shape
        flat_obs = seq_obs.view(-1, C, H, W)
        
        # Get all real latents
        flat_real_latents = model.worldencoder_net(flat_obs)
        # Reshape back: [Batch, K+1, LatentC, H, W]
        real_latents = flat_real_latents.view(batch_k, T_plus_1, -1, 6, 7)
        
        # B. The Unroll Loop
        # Start with real s_0
        curr_pred_latent = real_latents[:, 0] 
        
        rollout_loss_vec = 0
        
        for k in range(K_ROLLOUT):
            # 1. Get action taken at step k
            action_k = seq_act[:, k]
            
            # 2. Predict s_{k+1} using s_k (predicted) and a_k
            next_pred_latent, next_pred_done = model.latent2latent(curr_pred_latent, action_k)
            
            # 3. Get Ground Truth s_{k+1} (Encoded real state)
            # IMPORTANT: Detach target. We don't want the encoder to change to match the prediction.
            real_next_latent = real_latents[:, k+1].detach()
            real_next_done = seq_dones[:, k]
            
            # 4. Consistency Loss
            # Cosine sim over flattened latent
            sim_loss_vec = 1.0 - F.cosine_similarity(
                next_pred_latent.view(batch_k, -1), 
                real_next_latent.view(batch_k, -1)
            )
            
            # 5. Done Loss (Did the game end?)
            done_loss_vec = crossentropyloss_none(next_pred_done, real_next_done.long())
            
            rollout_loss_vec += seq_weights * (sim_loss_vec + done_loss_vec)
            
            # prepare for next step
            curr_pred_latent = next_pred_latent

        loss_rollout = rollout_loss_vec.mean() / K_ROLLOUT
        loss += loss_rollout
        
        # Update replay buffer
        K_rollout_replaybuffer.updateLossesFromSample(rollout_loss_vec.detach())
        
        # Grad updates
        model.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        model.optim.step()
        
        # Polyak averaging!
        model.polyak_value_update()
        
        # Update loss
        replaybuffer.updateLossesFromSample(loss_vec.detach())
        running_loss += loss.item()
        
    return running_loss / epochs

if __name__ == '__main__':
    
    # First try
    gen_rand_data(1500, epsilon=0.15)
    gen_rand_data(1000, epsilon=0)
    train_data(epochs=10, batch_size=1024)
    
    # Continue training
    # model.load_checkpoint("bomb.pth")
    # gen_data(2500)
    try:
        for i in range(5000):
            print(f"EPOCH {i+1}")
            gen_data(500)
            avg_loss = train_data(epochs=8, batch_size=256)
            print(f"Avg loss: {avg_loss}")
    except KeyboardInterrupt:
        pass

    model.save_checkpoint("bomb.pth")
