import torch
import torch.nn as nn

# Shared building blocks for RandLANetDA

class SharedMLP(nn.Module):
    """Conv/BN/activation block used throughout RandLANet variants."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 transpose=False,
                 bn=True,
                 activation_fn=None):
        super().__init__()

        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=(kernel_size - 1) // 2)
        else:
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=(kernel_size - 1) // 2)

        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6,
                                         momentum=0.01) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    """Compute k-NN positional encodings for each point."""

    def __init__(self, dim_in, dim_out, num_neighbors, encode_pos=False):
        super().__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(dim_in, dim_out, activation_fn=nn.LeakyReLU(0.2))
        self.encode_pos = encode_pos

    def gather_neighbor(self, coords, neighbor_indices):
        B, N, K = neighbor_indices.size()
        dim = coords.shape[2]

        extended_indices = neighbor_indices.unsqueeze(1).expand(B, dim, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(
            B, dim, N, K)
        neighbor_coords = torch.gather(extended_coords, 2, extended_indices)
        return neighbor_coords

    def forward(self,
                coords,
                features,
                neighbor_indices,
                relative_features=None):
        B, N, K = neighbor_indices.size()

        if self.encode_pos:
            neighbor_coords = self.gather_neighbor(coords, neighbor_indices)
            extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(
                B, 3, N, K)

            relative_pos = extended_coords - neighbor_coords
            relative_dist = torch.sqrt(
                torch.sum(torch.square(relative_pos), dim=1, keepdim=True))

            relative_features = torch.cat(
                [relative_dist, relative_pos, extended_coords, neighbor_coords],
                dim=1)
        else:
            if relative_features is None:
                raise ValueError(
                    "LocalSpatialEncoding: Require relative_features for second pass."
                )

        relative_features = self.mlp(relative_features)
        neighbor_features = self.gather_neighbor(
            features.transpose(1, 2).squeeze(3), neighbor_indices)

        return torch.cat([neighbor_features, relative_features],
                         dim=1), relative_features


class AttentivePooling(nn.Module):
    """Attention-weighted neighbor pooling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.score_fn = nn.Sequential(nn.Linear(in_channels, in_channels),
                                      nn.Softmax(dim=-2))
        self.mlp = SharedMLP(in_channels,
                             out_channels,
                             activation_fn=nn.LeakyReLU(0.2))

    def forward(self, x):
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        features = torch.sum(scores * x, dim=-1, keepdim=True)
        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):
    """Two-stage local feature aggregation with attentive pooling."""

    def __init__(self, d_in, d_out, num_neighbors):
        super().__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.lse1 = LocalSpatialEncoding(10,
                                         d_out // 2,
                                         num_neighbors,
                                         encode_pos=True)
        self.pool1 = AttentivePooling(d_out, d_out // 2)

        self.lse2 = LocalSpatialEncoding(d_out // 2, d_out // 2, num_neighbors)
        self.pool2 = AttentivePooling(d_out, d_out)
        self.mlp2 = SharedMLP(d_out, 2 * d_out)

        self.shortcut = SharedMLP(d_in, 2 * d_out)
        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, feat, neighbor_indices):
        x = self.mlp1(feat)

        x, neighbor_features = self.lse1(coords, x, neighbor_indices)
        x = self.pool1(x)

        x, _ = self.lse2(coords,
                         x,
                         neighbor_indices,
                         relative_features=neighbor_features)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(feat))
