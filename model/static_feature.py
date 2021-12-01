from torch import nn

class StaticFeature(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=None):
        super().__init__()

        # Build Downsampling-CNN
        hidden_dims = hidden_dims if hidden_dims else [64] * 12

        self.conv_block01 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=hidden_dims[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block02 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[0], out_channels=hidden_dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU()
        )

        self.conv_block03 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[1], out_channels=hidden_dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.LeakyReLU()
        )

        self.conv_block04 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[2], out_channels=hidden_dims[3], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(hidden_dims[3]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block05 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[3], out_channels=hidden_dims[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[4]),
            nn.LeakyReLU()
        )

        self.conv_block06 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[4], out_channels=hidden_dims[5], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[5]),
            nn.LeakyReLU()
        )

        self.conv_block07 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[5], out_channels=hidden_dims[6], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(hidden_dims[6]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block08 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[6], out_channels=hidden_dims[7], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[7]),
            nn.LeakyReLU()
        )

        self.conv_block09 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[7], out_channels=hidden_dims[8], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[8]),
            nn.LeakyReLU()
        )

        self.conv_block10 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[8], out_channels=hidden_dims[9], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(hidden_dims[9]),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block11 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[9], out_channels=hidden_dims[10], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[10]),
            nn.LeakyReLU()
        )

        self.conv_block12 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[10], out_channels=hidden_dims[11], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims[11]),
            nn.LeakyReLU()
        )

    def forward(self, x):

        x = self.conv_block01(x)
        x = self.conv_block02(x)
        x = self.conv_block03(x)
        x = self.conv_block04(x)
        x = self.conv_block05(x)
        x = self.conv_block06(x)
        x = self.conv_block07(x)
        x = self.conv_block08(x)
        x = self.conv_block09(x)
        x = self.conv_block10(x)
        x = self.conv_block11(x)
        x = self.conv_block12(x)

        return x # (batch_size, feature_size, feature_size)