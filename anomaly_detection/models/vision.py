import torch
from torch import nn


class VisionModel(nn.Module):
    def __init__(self, backbone, hidden_dim, dim_ff, series_len, classes):
        super(VisionModel, self).__init__()

        if backbone == 'dinov2_vits14':
            # Load the pre-trained DINO model
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14')
            # Freeze all parameters first
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze the last two blocks
            for param in self.backbone.blocks[-2:].parameters():
                param.requires_grad = True
            input_dim = 384
        elif backbone == 'resnet18':
            # Load the pre-trained ResNet model and remove the final fully
            # connected layer
            self.backbone = torch.hub.load(
                "pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
            self.backbone = nn.Sequential(
                *list(self.backbone.children())[:-2])
            input_dim = 512
        else:
            raise ValueError(
                f"Unknown backbone: {backbone}, "
                "please choose from ['dinov2_vits14', 'resnet18']")

        self.backbone_name = backbone

        '''
        CONVLUTIONAL LAYER
        '''
        # input [nBatch, nChannels, time_steps]
        self.cnn1 = nn.Conv2d(in_channels=input_dim,
                              out_channels=hidden_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        # channels are given by filters added in conv layer
        self.bn1 = nn.BatchNorm2d(hidden_dim, momentum=0.9)
        self.act = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()

        '''
        FULLY CONNECTED LAYER
        '''
        self.fc1 = nn.Linear(series_len * hidden_dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        '''
        input: (batch, seq_len, feat_w, feat_h, channels)
        '''
        batch, seq_len = x.shape[:2]

        if self.backbone_name == 'dinov2_vits14':
            y = x.new(batch, seq_len, 256, 384)
            for i in range(batch):
                y[i] = self.backbone(
                    x[i], is_training=True)['x_norm_patchtokens']
            x = y.permute(0, 1, 3, 2)
            x = x.reshape(batch, seq_len, 384, 16, 16)
        else:
            y = x.new(batch, seq_len, 512, 7, 7)
            for i in range(batch):
                y[i] = self.backbone(x[i])
            x = y

        outs = []
        for i in range(x.size(0)):
            out = self.act(self.bn1(self.cnn1(x[i])))
            out = self.flat(self.pool(out))
            outs.append(out)

        outs = torch.stack(outs, dim=0)
        outs = self.flat(outs)

        outs = self.fc2(self.dropout(self.act(self.fc1(outs))))

        return outs
