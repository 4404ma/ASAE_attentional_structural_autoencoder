import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=1, hidden_dims=None, batch_size=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_dims = latent_dim * 12
        self.batch_size = batch_size

        # Build Decoder
        hidden_dims = hidden_dims.reverse() if hidden_dims else [64] * 12

        self.conv_block01 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[0], out_channels=hidden_dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU()
        )

        self.conv_block02 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[1], out_channels=hidden_dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.LeakyReLU()
        )

        self.conv_block03 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[2], out_channels=hidden_dims[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[3]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.conv_block04 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[3], out_channels=hidden_dims[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[4]),
            nn.LeakyReLU()
        )

        self.conv_block05 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[4], out_channels=hidden_dims[5], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[5]),
            nn.LeakyReLU()
        )

        self.conv_block06 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[5], out_channels=hidden_dims[6], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[6]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.conv_block07 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[6], out_channels=hidden_dims[7], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[7]),
            nn.LeakyReLU()
        )

        self.conv_block08 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[7], out_channels=hidden_dims[8], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[8]),
            nn.LeakyReLU()
        )

        self.conv_block09 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[8], out_channels=hidden_dims[9], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[9]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.conv_block10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[9], out_channels=hidden_dims[10], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[10]),
            nn.LeakyReLU()
        )

        self.conv_block11 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[10], out_channels=hidden_dims[11], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[11]),
            nn.LeakyReLU()
        )

        self.conv_block12 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[11], out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, out_channels=out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    # structural transform of latent variable
    def structural_tfm(self, q=[]):
        str_tfm = [nn.Sequential(
             nn.Linear(self.latent_dim, 256),
             nn.Linear(256, 2)
        )] * 12

        latent = torch.ones(self.batch_size, 24, 1, 1, 1)
        for segment in range(12):
            tfm = q[:, segment*self.latent_dim:(segment+1)*self.latent_dim]
            tfm = str_tfm[segment](tfm)
            latent[:, segment * 2] = tfm[:, 0].view(self.batch_size, 1, 1, 1) # (batch_size, Z_s)
            latent[:, segment * 2 + 1] = tfm[:, 1].view(self.batch_size, 1, 1, 1) # (batch_size, Z_b)

        return latent    # (batch_size, 24)

    def forward(self, x, q):
        latent = self.structural_tfm(q)
        x = self.conv_block01(x*(latent[:, 0].repeat(torch.unsqueeze(x[0], 0).shape))+latent[:, 1].repeat(torch.unsqueeze(x[0], 0).shape))
        x = self.conv_block02(x*latent[:, 2].repeat(torch.unsqueeze(x[0], 0).shape)+latent[:, 3].repeat(torch.unsqueeze(x[0], 0).shape))
        x = self.conv_block03(x*latent[:, 4].repeat(torch.unsqueeze(x[0], 0).shape)+latent[:, 5].repeat(torch.unsqueeze(x[0], 0).shape))
        x = self.conv_block04(x*latent[:, 6].repeat(torch.unsqueeze(x[0], 0).shape)+latent[:, 7].repeat(torch.unsqueeze(x[0], 0).shape))
        x = self.conv_block05(x*latent[:, 8].repeat(torch.unsqueeze(x[0], 0).shape)+latent[:, 9].repeat(torch.unsqueeze(x[0], 0).shape))
        x = self.conv_block06(x*latent[:, 10].repeat(torch.unsqueeze(x[0], 0).shape)+latent[:, 11].repeat(torch.unsqueeze(x[0], 0).shape))
        x = self.conv_block07(x*latent[:, 12].repeat(torch.unsqueeze(x[0], 0).shape)+latent[:, 13].repeat(torch.unsqueeze(x[0], 0).shape))
        x = self.conv_block08(x*latent[:, 14].repeat(torch.unsqueeze(x[0], 0).shape)+latent[:, 15].repeat(torch.unsqueeze(x[0], 0).shape))
        x = self.conv_block09(x*latent[:, 16].repeat(torch.unsqueeze(x[0], 0).shape)+latent[:, 17].repeat(torch.unsqueeze(x[0], 0).shape))
        x = self.conv_block10(x*latent[:, 18].repeat(torch.unsqueeze(x[0], 0).shape)+latent[:, 19].repeat(torch.unsqueeze(x[0], 0).shape))
        x = self.conv_block11(x*latent[:, 20].repeat(torch.unsqueeze(x[0], 0).shape)+latent[:, 21].repeat(torch.unsqueeze(x[0], 0).shape))
        x = self.conv_block12(x*latent[:, 22].repeat(torch.unsqueeze(x[0], 0).shape)+latent[:, 23].repeat(torch.unsqueeze(x[0], 0).shape))

        return x