import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        self.x_forward = nn.Identity()
        if in_channels != out_channels:
            self.x_forward = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x):
        return self.x_forward(x) + self.resblock(x)

class WorldEncoder(nn.Module):
    def __init__(self, inchannels=3, latent_channels=32):
        super().__init__()
        
        # input is six seven ajklfasdjklfjklasdfjklashjklajksdgjklasdgas
        # (3,6,7) actually erm!!!
        # This one encodes to latent space of latent_dim
        self.resnet = nn.Sequential(
            ResBlock(inchannels, 96),
            ResBlock(96, 96),
            ResBlock(96, latent_channels), # [latent_channels, 6, 7]
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.resnet(x)

class Latent2Latent(nn.Module):
    def __init__(self, dim1, dim2, actions: int, device, latent_channels=32):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dimsum = dim1 * dim2
        self.actions = actions
        self.device = device
        
        self.resnet = nn.Sequential(
            ResBlock(latent_channels + actions, 96),
            ResBlock(96, 96),
        )
        self.latent_head = nn.Sequential(
            ResBlock(96, latent_channels),
            nn.Tanh()
        )
        self.terminated_head = nn.Linear(96*self.dimsum, 2) # 0 -> running, 1 -> terminated
    
    def forward(self, latent_state, action):
        # Latent shape: [batch, latent_channels, dim1, dim2]
        # Current action shape: [batch]
        # Ideal action shape: [batch, actiobns, dim1, dim2]
        batch_size = action.shape[0]
        action_tensor = torch.zeros((batch_size, self.actions, self.dim1, self.dim2), device=self.device)
        action_tensor[torch.arange(batch_size, device=self.device), action] = 1
        x = torch.cat([latent_state, action_tensor], dim=1)
        resnet = self.resnet(x)
        return self.latent_head(resnet), self.terminated_head(resnet.view(-1, 96*self.dimsum))
    
    @torch.inference_mode()
    def forward_no_reward_head(self, latent_state, action):
        batch_size = action.shape[0]
        action_tensor = torch.zeros((batch_size, self.actions, self.dim1, self.dim2), device=self.device)
        action_tensor[torch.arange(batch_size, device=self.device), action] = 1
        x = torch.cat([latent_state, action_tensor], dim=1)
        resnet = self.resnet(x)
        return self.latent_head(resnet)
        

class Policy(nn.Module):
    def __init__(self, dim1, dim2, actions: int, latent_channels=32):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dimsum = dim1 * dim2
        
        self.resnet = nn.Sequential(
            ResBlock(latent_channels, 24),
            ResBlock(24, 24),
        )
        self.rnn = nn.Sequential(
            nn.Linear(24*self.dimsum, actions)
        )
    
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 24*self.dimsum)
        return self.rnn(x)

class Value(nn.Module):
    def __init__(self, dim1, dim2, actions: int, device, latent_channels=32):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dimsum = dim1 * dim2
        self.actions = actions
        self.device = device
        
        self.resnet = nn.Sequential(
            ResBlock(latent_channels + actions, 32),
            ResBlock(32, 32),
        )
        self.rnn = nn.Sequential(
            nn.Linear(32*self.dimsum, 1),
            nn.Tanh()
        )
    
    def forward(self, latent_state, action):
        # Latent shape: [batch, latent_channels, dim1, dim2]
        # Current action shape: [batch]
        # Ideal action shape: [batch, actions, dim1, dim2]
        batch_size = action.shape[0]
        action_tensor = torch.zeros((batch_size, self.actions, self.dim1, self.dim2), device=self.device)
        action_tensor[torch.arange(batch_size, device=self.device), action] = 1
        x = torch.cat([latent_state, action_tensor], dim=1)
        x = self.resnet(x)
        x = x.view(-1, 32*self.dimsum)
        return self.rnn(x)