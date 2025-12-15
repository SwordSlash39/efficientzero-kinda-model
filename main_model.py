import torch
import torch.nn as nn
import torch.nn.functional as F
from models import WorldEncoder, Latent2Latent, Policy, Value

class MainModel(nn.Module):
    def __init__(
        self, 
        dim1: int, 
        dim2: int, 
        actions: int, 
        input_channels: int, 
        latent_channels: int, 
        device, 
        
        # Hyperparams
        tau=0.95, gamma=1.0, lr=5e-4, weight_decay=1e-4,
        
        # MCTS settings
        mcts_nodes=64, cpuct=1.5, mcts_max_depth=8
    ):
        super().__init__()
        
        self.device = device
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.actions = actions
        
        self.dim1 = dim1
        self.dim2 = dim2
        
        # Hyperparameters
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.mcts_nodes = mcts_nodes
        self.cpuct = cpuct
        self.mcts_max_depth = mcts_max_depth
        
        # World encoder model
        self.worldencoder = WorldEncoder(
            inchannels=self.input_channels,
            latent_channels=self.latent_channels
        ).to(device)
        
        # Latent to latent (with terminated prediction)
        self.latent2latent = Latent2Latent(
            dim1=self.dim1,
            dim2=self.dim2,
            actions=self.actions,
            device=self.device,
            latent_channels=self.latent_channels
        ).to(device)
        
        # Policy network
        self.policy = Policy(
            dim1=self.dim1,
            dim2=self.dim2,
            actions=self.actions,
            latent_channels=self.latent_channels
        ).to(device)
        
        # Value and Target value (for slower updates)
        self.value = Value(
            dim1=self.dim1,
            dim2=self.dim2,
            actions=self.actions,
            latent_channels=self.latent_channels,
            device=self.device
        ).to(device)
        self.target_value = Value(
            dim1=self.dim1,
            dim2=self.dim2,
            actions=self.actions,
            latent_channels=self.latent_channels,
            device=self.device
        ).to(device)
        self.target_value.load_state_dict(self.value.state_dict())
        
        for param in self.target_value.parameters():
            param.requires_grad = False 
        
        # Create optim
        self.optim = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def polyak_value_update(self):
        for target_param, online_param in zip(self.target_value.parameters(), self.value.parameters()):
            # In-place update: target = target * (1-tau) + online * tau
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(online_param.data, alpha=self.tau)
    
    def save_checkpoint(self, path):
        torch.save({
            'model': self.state_dict(),
            'optim': self.optim.state_dict()
        }, path)
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optim'])
    
    def forward(self, x):
        return self.policy(x)
    
    @torch.inference_mode()
    def predict(self, x, **args):
        return self.policy(x), None
    
    def worldencoder_net(self, x):
        return self.worldencoder(x)
    
    def latent2latent_net(self, x, action):
        return self.latent2latent(x, action)
    
    def value_net(self, x, action):
        return self.value(x, action)
    
    @torch.inference_mode()
    def target_value_net(self, x, action):
        return self.target_value(x, action)

    @torch.inference_mode()
    def parallel_mcts(self, x, training_mode=True, mcts_nodes=None, debug=False, max_depth=None):
        # Set self to eval mode
        self.eval()
        
        # x: [batch, in_channels, 6, 7]
        x = x.to(self.device)
                
        nodes = self.mcts_nodes if mcts_nodes is None else mcts_nodes
        nodes += 1 # base node (root)
        loop_count = nodes
        nodes *= self.actions # we need a new node for every policy, so at best its +7 per action
        batch_size = x.shape[0]       
        
        max_depth = self.mcts_max_depth if max_depth is None else max_depth
        
        # Nodes + 1 to accomodate the root node
        visits = torch.zeros((batch_size, nodes + 1), dtype=torch.float32, device=self.device)
        values = torch.zeros((batch_size, nodes + 1), dtype=torch.float32, device=self.device)
        policy_net_outputs = torch.zeros((batch_size, nodes+1, self.actions), dtype=torch.float32, device=self.device)
        nextNodeIndex = torch.full((batch_size, nodes + 1, self.actions), fill_value=nodes+10, dtype=torch.long, device=self.device)
        latent_states = torch.zeros((batch_size, nodes + 1, self.latent_channels, self.dim1, self.dim2), dtype=torch.float32, device=self.device)        
        
        # parent and terminal
        parent_node = torch.zeros((batch_size, nodes + 1), dtype=torch.long, device=self.device)
        parent_action = torch.zeros((batch_size, nodes + 1), dtype=torch.long, device=self.device)
        node_computed = torch.zeros((batch_size, nodes + 1), dtype=torch.bool, device=self.device)
        terminal_node = torch.zeros((batch_size, nodes + 1), dtype=torch.bool, device=self.device)
        
        # Current index of selected node
        curr_node_idx = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        
        # Create iterables to save memory
        batch_iter = torch.arange(batch_size, dtype=torch.long, device=self.device)
        
        # Track which batch nodes are a leaf
        active_mask = torch.ones((batch_size,), dtype=torch.bool, device=self.device)
        
        # Track node usage count to add new nodes
        node_usage_count = torch.ones((batch_size,), dtype=torch.long, device=self.device)
        action_iter = torch.arange(self.actions, dtype=torch.long, device=self.device)
        
        # Track path and depth for backpropagation
        path = torch.zeros((batch_size, max_depth + 1), dtype=torch.long, device=self.device)
        depth_tracker = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        
        # Add root node
        latent_states[:, 0] = self.worldencoder(x)
        policy_logits = self.eval_policy(latent_states[batch_iter, 0])
        root_probs = torch.softmax(policy_logits, dim=-1)
        
        # Add Dirichlet noise
        if training_mode:
            dirichlet_alpha = 10/self.actions
            exploration_fraction = 0.25
            
            m = torch.distributions.dirichlet.Dirichlet(
                torch.full((batch_size, self.actions), dirichlet_alpha, device=self.device)
            )
            noise = m.sample()
            
            # Mix the noise
            root_probs = (1 - exploration_fraction) * root_probs + (exploration_fraction * noise)
        
        policy_net_outputs[batch_iter, 0] = root_probs
        node_computed[:, 0] = True
        visits[:, 0] = 1
        # Create first children
        child_idx = action_iter + node_usage_count.unsqueeze(1)
        nextNodeIndex[batch_iter, 0] = child_idx
        parent_node[batch_iter.unsqueeze(1), child_idx] = curr_node_idx.unsqueeze(1) # [batch, actions]
        parent_action[batch_iter.unsqueeze(1), child_idx] = action_iter 
        node_usage_count += self.actions
        # Set children node value 0.126 (12.6% of games result in win)
        values[batch_iter.unsqueeze(1), child_idx] = 0.1264
        
        for n in range(loop_count-1):
            # Start from root node and find next node
            curr_node_idx.zero_()
            
            # Reset active mask
            active_mask[:] = 1
            
            # Reset path and depth tracker
            path.zero_()
            depth_tracker.zero_()
            
            # Iterate for max_depth
            for d in range(max_depth):          
                path[:, d] = curr_node_idx
                depth_tracker[active_mask] = d
                
                # Update if we're looking at a leaf node
                isLeaf = ~node_computed[batch_iter, curr_node_idx]
                isTerminal = terminal_node[batch_iter, curr_node_idx]
                shouldStop = isLeaf | isTerminal         
                
                # Update active mask
                active_mask = active_mask & (~shouldStop)
                
                # if all batches leaf, break
                if not active_mask.any():
                    break
                
                # Splice to only run UCT on active nodes (non leaf)
                active_batch_iter = batch_iter[active_mask]
                active_node_idx = curr_node_idx[active_mask]
                
                # Not leaf node? lets go down
                # We need to construct a UCT
                child_index = nextNodeIndex[active_batch_iter, active_node_idx]                    
                curr_uct = (
                    # Value net data
                    # Non negative despite zero sum game (why?)
                    # Value at node is Q(s, a) where s is the parent node and A is action took to reach that node
                    # Q(s, a) will always be positive = good since child nodes are parent nodes taking specific actions
                    # And backpropagating will handle the opponent - Player things
                    # So keep positive
                    values[active_batch_iter.unsqueeze(1), child_index] / visits[active_batch_iter.unsqueeze(1), child_index].clamp(min=1) +   # nextNodeIndex[batch_iter, curr_idx].shape is self.actions
                    
                    # cpuct * policy net * exploration factor
                    self.cpuct * 
                    policy_net_outputs[active_batch_iter, active_node_idx] * 
                    torch.sqrt(visits[active_batch_iter, active_node_idx].unsqueeze(1)) / (1 + visits[active_batch_iter.unsqueeze(1), child_index]) # visits should be shape self.actions (please)
                ) # [batch, actions]
                
                # but now we need to find the best best best best best node!!!!!
                # uct[:, curr_node_idx].shape: [batch, selected nodes, children]
                if d+1 == max_depth:
                    # If we are max depth, dont consider moving on to children
                    terminal_node[active_batch_iter, active_node_idx] = True
                else:
                    curr_node_idx[active_mask] = nextNodeIndex[active_batch_iter, active_node_idx, torch.argmax(curr_uct, dim=1)] # shape: [batch]
            
            # Only expand non terminal states
            notTerminal = ~terminal_node[batch_iter, curr_node_idx]
            
            # Update iter to go through no terminal
            batch_iter_non_terminal = batch_iter[notTerminal]
            curr_node_idx_non_terminal = curr_node_idx[notTerminal]
            
            if batch_iter_non_terminal.numel() > 0:            
                # Now we know all curr_node_idx are on leaf nodes, and can expand
                # To expand:
                # - Get latent and value net of current node
                # - Run Policy net on current node
                leaf_node_parent_actions = parent_action[batch_iter_non_terminal, curr_node_idx_non_terminal] # [batch]
                leaf_node_parent_pos = parent_node[batch_iter_non_terminal, curr_node_idx_non_terminal] # [batch]
                leaf_node_parent_latent_states = latent_states[batch_iter_non_terminal, leaf_node_parent_pos]
                
                # Get latent state of leaf node
                latent_states[batch_iter_non_terminal, curr_node_idx_non_terminal], terminated = self.latent2latent(leaf_node_parent_latent_states, leaf_node_parent_actions)
                terminated = torch.argmax(terminated, dim=1)
                consider_terminated = (terminated == 1)
                terminal_node[batch_iter_non_terminal, curr_node_idx_non_terminal] = consider_terminated
                
                # We rerun this to update non terminal nodes
                notTerminal = ~terminal_node[batch_iter, curr_node_idx]
                batch_iter_non_terminal = batch_iter[notTerminal]
                curr_node_idx_non_terminal = curr_node_idx[notTerminal]
                
                if batch_iter_non_terminal.numel() > 0: # check again :(
                    # Run policy network
                    current_latents = latent_states[batch_iter_non_terminal, curr_node_idx_non_terminal]  
                    policy_logits = self.eval_policy(current_latents)
                    policy_probs = torch.softmax(policy_logits, dim=-1)
                    policy_net_outputs[batch_iter_non_terminal, curr_node_idx_non_terminal] = policy_probs
                    
                    # Create new children
                    # Each leaf node has self.actions new children
                    # child_idx: [batch, actions] but its actually [batch, node_pos] so yeah!
                    child_idx = action_iter + node_usage_count.unsqueeze(1)
                    child_idx = child_idx[notTerminal]
                    
                    # Update child, and update child parents
                    nextNodeIndex[batch_iter_non_terminal, curr_node_idx_non_terminal] = child_idx
                    parent_node[batch_iter_non_terminal.unsqueeze(1), child_idx] = curr_node_idx_non_terminal.unsqueeze(1) # [batch, actions]
                    parent_action[batch_iter_non_terminal.unsqueeze(1), child_idx] = action_iter
                    
                    # Set children node value to negative parent node (0 is too harsh / lenient)
                    values[batch_iter_non_terminal.unsqueeze(1), child_idx] = -values[batch_iter_non_terminal, curr_node_idx_non_terminal].unsqueeze(1)
                    
                    # Update global node usage
                    node_usage_count[notTerminal] += self.actions 
            
            # Get value of leaf node for backprop
            # Value's a bit different; we still calculate the value for terminated states to propagate upwards
            # We dont do this for the rest as it isnt needed
            value_leaf_node_parent_actions = parent_action[batch_iter, curr_node_idx]
            value_leaf_node_parent_pos = parent_node[batch_iter, curr_node_idx]
            value_leaf_node_parent_latent_states = latent_states[batch_iter, value_leaf_node_parent_pos]
            parent_values = self.eval_value(value_leaf_node_parent_latent_states, value_leaf_node_parent_actions).squeeze(-1)
                    
            # -------- Backprop --------         
            # Find max depth
            current_max_depth = depth_tracker.max().item()
            
            # Iterate backwards from max possible depth
            for d in range(current_max_depth, -1, -1):
                # Only update if this depth was actually reached in the path
                mask = (d <= depth_tracker)
                if mask.sum() == 0:
                    continue
                    
                nodes_to_update = path[mask, d]
                b_indices = batch_iter[mask]
                
                # Update values
                # We add the value. 
                # we flip signs at each level for zero sum game
                relative_depth = depth_tracker[mask] - d
                # Since relative depth starts at 0 and we start at parent of new leaf, it should be 1 as we calculate value of parent node.
                sign = 1.0 - 2.0 * (relative_depth % 2).float()
                values[b_indices, nodes_to_update] += parent_values[mask] * sign * (self.gamma ** relative_depth)
                
                # Update visits
                visits[b_indices, nodes_to_update] += 1
            
            # Set node as searched
            node_computed[batch_iter, curr_node_idx] = True
            
        # Search ended!
        # Get distribution for each one
        root_visit_batch = visits[batch_iter.unsqueeze(1), nextNodeIndex[batch_iter, 0]]
        prob_dist_sum = root_visit_batch.sum(axis=1).unsqueeze(1)
        prob_dist_norm = root_visit_batch / prob_dist_sum.clamp(min=1)
        
        # END: set self to train mode
        if training_mode:
            self.train()
        
        # Return dist prob
        if debug:
            print(root_visit_batch)
            print(values[batch_iter.unsqueeze(1), nextNodeIndex[batch_iter, 0]] / root_visit_batch.clamp(min=1))
            print()
        return prob_dist_norm

    @torch.inference_mode()
    def eval_latent2latent(self, x, action):
        return self.latent2latent(x, action)
    
    @torch.inference_mode()
    def eval_policy(self, x):
        return self.policy(x)
    
    @torch.inference_mode()
    def eval_value(self, x, action):
        return self.value(x, action)
    
    @torch.inference_mode()
    def eval_worldencoder(self, x):
        return self.worldencoder(x)
