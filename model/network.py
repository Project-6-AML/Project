
import torch
import logging
import torchvision
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from model.AttenNetVLAD import NetVLAD
from copy import deepcopy

from model.layers import Flatten, L2Norm, GeM
from self_attention_GAN import Self_Attn
from transformer import TransformerEncoderLayer, TransformerEncoder

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
}


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone : str, fc_output_dim : int, self_attn = False, rerank = False):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim, self.avg_fc = get_backbone(backbone)       
        print(f"Backbone {self.backbone}")
        print(f"Ultimi 2 layer {self.avg_fc}")
        self.avg2DPooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 1000)
        
        self.netvlad_layer = None
        if self_attn:
            print()
            #self.attn = Self_Attn( 512, 'relu')
            self.netvlad_layer = NetVLAD(num_clusters=32, dim=512)

        self.aggregation = nn.Sequential(
            #L2Norm(),
            #GeM(),
            #Flatten(),
            nn.Linear(16384, fc_output_dim),
            L2Norm()
        )
        
        self.rerank = None
        if rerank:
            nhead = 4
            dropout = 0.1
            activation = 'relu'
            normalize_before = False
            dim_feedforward = features_dim #dim of the blocks of the inner dropout
            encoder_layer = TransformerEncoderLayer(features_dim, nhead, dim_feedforward, dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(features_dim) if normalize_before else None
            self.rerank = TransformerEncoder(encoder_layer, 1, encoder_norm)
        
    def forward(self, x):
        #print(f"Original dimension: {x.shape}")
        #fc_out, feature_conv, feature_convNBN = self.backbone(x)
        #print(f'{fc_out.size()}, {feature_conv.size()}, {feature_convNBN.size()}')
        x = self.backbone(x)
        print(f"Dimension after backbone: {x.shape}")
        x = torch.squeeze(x, 1)
        print(f"Dimension after added squeeze: {x.shape}")
        """if self.attn:
            x, _ = self.attn(x)
            print(f"Dimension after attention layer: {x.shape}")"""
        #print(f"Dimension after backbone: {x.shape}")
        feature_conv = x #deepcopy(x)  
        x = self.avg2DPooling(x)
        #print(f"Dimension after {self.avg2DPooling}: {x.shape}")
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #print(f"Dimension after {self.fc}: {x.shape}")
        
        self.weight_softmax = self.fc.weight
        #x = torch.squeeze(x, 1) SQUEEZE PER IL RE-RANKING
        #print(f"Dimension after added squeeze: {x.shape}")
        if self.netvlad_layer:
            #x, feature_conv, feature_convNBN = self.backbone(x)
            #print(f'{x.size()}, {feature_conv.size()}, {feature_convNBN.size()}')
            bz, nc, h, w = feature_conv.size()
            feature_conv_view = feature_conv.view(bz, nc, h * w)
            probs, idxs = x.sort(1, True)
            class_idx = idxs[:, 0]
            scores = self.weight_softmax[class_idx].to(x.device)
            cam = torch.bmm(scores.unsqueeze(1), feature_conv_view)
            attention_map = F.softmax(cam.squeeze(1), dim=1)
            attention_map = attention_map.view(attention_map.size(0), 1, h, w)
            attention_features = attention_map.expand_as(feature_conv)
            x = self.netvlad_layer(attention_features)
            #print(f"Dimension after attention layer: {x.shape}")
        if self.rerank:
            x = self.rerank(x)
            #print(f"Dimension after reranking layer: {x.shape}")
            
        #x = x.unsqueeze(-1)
        #x = x.unsqueeze(-1)
        x = self.aggregation(x)
        return x


def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(backbone_name : str) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
        last_layers = list(backbone.children())[-2:]
        
    
    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        last_layers = list(backbone.children())[-2:]
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")

    # RERANKING LAYERS -------------------------------------------------------------------------------------------------------------------------------------------------
    #layers.append(conv1x1(512, 1))
    #layers.append(torch.nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    
    backbone = torch.nn.Sequential(*layers)
    
    avg_fc = last_layers
    #avg_fc = [torch.nn.Sequential(*[last_layer]) for last_layer in last_layers]
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim, avg_fc

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
