import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Type
from termcolor import cprint

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class ResNetEncoder(nn.Module):
    """
    Encoder for Images using pretrained ResNet18
    """
    def __init__(self,
                 observation_space: Dict,
                 out_channel=256,
                 state_mlp_size=(64, 64), 
                 state_mlp_activation_fn=nn.ReLU,
                 pretrained=True,
                 freeze_resnet=False
                 ):
        super().__init__()
        
        self.image_key = 'image'
        self.state_key = 'agent_pos'
        
        # Image processing
        # Assuming image shape is (C, H, W)
        self.image_shape = observation_space[self.image_key]
        cprint(f"[ResNetEncoder] image shape: {self.image_shape}", "yellow")
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the fully connected layer (fc)
        # ResNet18's avgpool output is 512
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        resnet_out_dim = 512
        
        if freeze_resnet:
            for param in self.resnet_features.parameters():
                param.requires_grad = False
            cprint("[ResNetEncoder] ResNet18 weights frozen", "cyan")
        
        # Projection layer to match out_channel
        self.image_projection = nn.Linear(resnet_out_dim, out_channel)
        
        # State processing
        self.state_shape = observation_space[self.state_key]
        cprint(f"[ResNetEncoder] state shape: {self.state_shape}", "yellow")
        
        net_arch = list(state_mlp_size)
        if len(net_arch) == 0:
            state_out_dim = 0
        else:
            state_out_dim = net_arch[-1]
            
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], state_out_dim, net_arch, state_mlp_activation_fn))
        
        self.n_output_channels = out_channel + state_out_dim
        cprint(f"[ResNetEncoder] total output dim: {self.n_output_channels}", "red")
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, observations: Dict) -> torch.Tensor:
        # Image processing
        images = observations[self.image_key] # (B, C, H, W) or (B, T, C, H, W)
        
        # Handle time dimension if present
        has_time_dim = len(images.shape) == 5
        if has_time_dim:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
            
        # Normalize images from [0, 255] to [0, 1] then apply ImageNet normalization
        images = images.float() / 255.0
        images = self.normalize(images)
            
        # ResNet expects (B, 3, H, W). If C != 3, we might need adjustment.
        # Assuming standard RGB images for now.
        
        img_feat = self.resnet_features(images) # (B, 512, 1, 1)
        img_feat = torch.flatten(img_feat, 1)   # (B, 512)
        img_feat = self.image_projection(img_feat) # (B, out_channel)
        
        if has_time_dim:
            img_feat = img_feat.view(B, T, -1)
            # If we need to aggregate over time or just return sequence, depends on usage.
            # Usually for policy, we might take the latest or keep sequence.
            # Here we assume the caller handles sequence or we just process batch.
            # But wait, FlowPolicy usually expects (B, feature_dim).
            # If input was (B, T, ...), output should probably be (B, T, ...) or aggregated.
            # Let's assume standard usage is (B, C, H, W) for single step or stacked frames.
            pass

        # State processing
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)
        
        # Concatenate
        final_feat = torch.cat([img_feat, state_feat], dim=-1)
        
        return final_feat

    def output_shape(self):
        return self.n_output_channels
