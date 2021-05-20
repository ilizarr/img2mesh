import torch
import torch.nn as nn
from torchvision.models import vgg16
from pytorch_model_summary import summary

class Vgg16Encoder(nn.Module):
    def __init__(self):
        super(Vgg16Encoder, self).__init__()
        # Load pretrained and freeze weights
        self.orig_model = vgg16(pretrained = True)
        self.orig_model.eval()
        for param in self.orig_model.parameters():
            param.requires_grad = False

        # stop at FC 4096
        feature_layer = nn.Sequential(
            *list(self.orig_model.classifier.children())[:-5]#[:-2]
        )
        self.orig_model.classifier = feature_layer
        self.out_feature_dim = 4096
    def forward(self, x):
        x = self.orig_model.forward(x)
        return x

class Img2Mesh(nn.Module):
    def __init__(self, num_vertices, num_views=2, att_heads=2, att_layers=1):
        super(Img2Mesh, self).__init__()
        self.num_vertices = num_vertices

        self.feature_model = Vgg16Encoder()
        num_features = self.feature_model.out_feature_dim

        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_features, nhead=att_heads), num_layers=att_layers)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, num_features),
            #nn.BatchNorm1d(num_features),
            nn.ReLU(True)
        )
        self.mesh_fc = nn.Sequential(
            nn.Linear(num_features, self.num_vertices*3)
        )
        self.mesh_pos_fc = nn.Sequential(
            nn.Linear(num_features, 3)
        )
        # Want RGB colors between [0,1]
        self.mesh_color_fc = nn.Sequential(
            nn.Linear(num_features, self.num_vertices*3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input shape N x V x 3 x H x W
        #   N: batch, V: num views of object
        # Output shapes:
        #   y_vertices (N x Vertices x 3)
        #   y_color    (N x Vertices x 3)
        #   y_pos      (N x 1 x 3) 
        N, V, Cin, H, W = tuple(x.shape)
        x = x.view(N*V, Cin, H, W)
        x = self.feature_model(x)                             # (N*V) x F
        x = x.view(N, V, -1)                                  # N x V x F
        x = self.transformer_enc(x)                           # N x V x F
        x = torch.mean(x, dim=1)                              # N x F
        x = x + self.mlp(x)                                   # N x F 
        
        vertices = self.mesh_fc(x)                            # N x (vertices*3)
        y_vertices = vertices.view(N, self.num_vertices, 3)
        color = self.mesh_color_fc(x)                         # N x (vertices*3)
        y_color = color.view(N, self.num_vertices, 3)
        pos = self.mesh_pos_fc(x)                             # N x 3
        y_pos = pos.view(N, 1, 3)
        return y_vertices, y_color, y_pos

def create_model(num_views=2, num_vertices=642, att_heads=2, att_layers=1):
    model = Img2Mesh(num_vertices, num_views, att_heads, att_layers)
    print(summary(model, torch.zeros((1, num_views, 3, 224, 224))))
    return model