import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Encoder
        self.encoder = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), # inp x inp
            nn.GELU(),
            # Conv layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # inp // 2 x inp // 2
            nn.GELU(),
            # Conv layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # inp // 4 x inp // 4
            nn.GELU(),
            # Conv layer 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), # inp // 8 x inp // 8
            nn.GELU(),
            # Conv layer 5
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1), # inp // 16 x inp // 16
        )

        fc_dim = self.encoder[-1].out_channels * (input_dim // 16) ** 2
        # Latent layers
        self.enc_fc = nn.Linear(in_features=fc_dim, out_features=2*latent_dim)
        self.dec_fc = nn.Linear(in_features=latent_dim, out_features=fc_dim//2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.shape[0], -1)
        mu, logvar = self.enc_fc(encoded).chunk(2, dim=1)

        # Reparametrization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # Decoding
        decoded = self.dec_fc(z)
        decoded = decoded.view(-1, 256, self.input_dim // 16, self.input_dim // 16)
        decoded = self.decoder(decoded)

        return decoded, mu, logvar