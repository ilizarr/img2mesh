import math
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision import transforms
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

class Vgg16ConvFrontEnd(nn.Module):
  """
    This front end stops at early CONV layers for VGG16 and flattens out its output
    by using a few learned conv layers at the end
  """
  def __init__(self, stop_at_maxpool=3, img_height=1024., img_width=1024., use_bn=False):
    super(Vgg16ConvFrontEnd, self).__init__()
    # Load pretrained
    orig_model = vgg16(pretrained = True)
    # Freeze weights
    # orig_model.eval()
    # for param in orig_model.parameters():
    #     param.requires_grad = False

    # VGG input preprocessing
    vgg_input_height = img_height // 4
    vgg_input_width = img_width // 4
    self.preprocessing = nn.Sequential(
        # This resize will not distort the image
        transforms.Resize(int(min(vgg_input_height, vgg_input_width)))
    )

    # Extract VGG 16 conv layers. Architecture from code:
    # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    # numbers are channels for conv3x3, each conv is followed by relu, M is maxpool
    num_channels=[64,128,256,512,512]
    out_conv_height = int(vgg_input_height / (2.0**stop_at_maxpool))
    out_conv_width = int(vgg_input_width / (2.0**stop_at_maxpool))
    out_conv_depth=num_channels[stop_at_maxpool-1]

    # stop at the desired conv layer
    vgg_layers = []
    mp_count = 0
    for l in orig_model.features.children():
        vgg_layers.append(l)
        if isinstance(l, nn.MaxPool2d):
            mp_count += 1
            if mp_count >= stop_at_maxpool:
                break

    self.vgg_features = nn.Sequential(*vgg_layers)

    # Now Add a few custom conv layers to reduce to channels
    reduce_factor = 4
    num_convs = math.floor(math.log(out_conv_depth)/math.log(reduce_factor))
    target_c = 1 # stop when num channels is <=

    layers = []
    in_c = out_conv_depth # start with the VGG output channels
    for i in range(num_convs):
        if in_c <= target_c:
            break
        out_c = in_c // reduce_factor
        layers += [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
        if use_bn:
            layers += [nn.BatchNorm2d(out_c)]
        layers += [nn.LeakyReLU(inplace=True)]
        in_c = out_c
    layers += [nn.Flatten()]
    self.features = nn.Sequential(*layers)

    # Main net needs this defined
    self.out_feature_dim = out_c * out_conv_height * out_conv_width
      
  def forward(self, x):
      x = self.preprocessing(x)
      x = self.vgg_features(x)
      x = self.features(x)
      return x

class Img2Mesh(nn.Module):
    def __init__(self, front_end, num_vertices, num_views=2, att_heads=2, att_layers=1):
        super(Img2Mesh, self).__init__()
        self.num_vertices = num_vertices

        self.feature_model = front_end
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
            nn.Hardsigmoid()
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

def create_base_model(num_views=2, num_vertices=642, att_heads=2, att_layers=1):
    encoder_model = Vgg16Encoder()
    model = Img2Mesh(encoder_model, num_vertices, num_views, att_heads, att_layers)
    print("[BASE] VGG FC ENCODER:")
    print(summary(model, torch.zeros((1, num_views, 3, 224, 224))))
    return model

def create_cnn_encoder_model(num_views=2, num_vertices=642, att_heads=2, att_layers=1):
    encoder_model = Vgg16ConvFrontEnd()
    model = Img2Mesh(encoder_model, num_vertices, num_views, att_heads, att_layers)
    print("VGG CONV ENCODER:")
    print(summary(encoder_model, torch.zeros((1, 3, 1024, 1024))))
    print(summary(model, torch.zeros((1, num_views, 3, 1024, 1024))))
    return model

if __name__ == '__main__':
    base_model = create_base_model()
    print()
    cnn_model = create_cnn_encoder_model()