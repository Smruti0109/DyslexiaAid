import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, "Image height must be a multiple of 16"

        self.cnn = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),  # (batch, 64, H, W)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch, 64, H/2, W/2)

            # Conv Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch, 128, H/4, W/4)

            # Conv Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            # Conv Layer 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # only downsample height

            # Conv Layer 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Conv Layer 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Conv Layer 7
            nn.Conv2d(512, 512, kernel_size=2, stride=1),  # (batch, 512, 1, W)
            nn.ReLU(True)
        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, x):
        conv = self.cnn(x)  # (batch, channels=512, height=1, width=W)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # (batch, channels, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, channels)
        output = self.rnn(conv)
        return output


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)  # (seq_len, batch, 2 * hidden)
        output = self.embedding(recurrent)  # (seq_len, batch, output_size)
        return output
