import torch.nn as nn
import torch

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,scale_factor=2):
        super(up_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Unet_upsample(nn.Module):

    def __init__(self,start_fm=16):
        super(Unet_upsample, self).__init__()

        # Input size= batchx1x128x128

        # Contracting Path

        # (Double) Convolution 1 ——>size=batchx16x128x128
        self.double_conv1 = double_conv(1, start_fm, 3, 1, 1)
        # Max Pooling 1 ——>size=batchx16x64x64
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2 ——>size=batchx32x64x64
        self.double_conv2 = double_conv(start_fm, start_fm * 2, 3, 1, 1)
        # Max Pooling 2 ——>size=batchx32x32x32
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Convolution 3 ——>size=batchx64x32x32
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4, 3, 1, 1)
        # Max Pooling 3 ——>size=batchx64x16x16
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # Convolution 4 ——>size=batchx128x16x16
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8, 3, 1, 1)
        # Max Pooling 4 ——>size=batchx128x8x8
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Convolution 5 ——>size=batchx256x8x8
        self.double_conv5 = double_conv(start_fm * 8, start_fm * 16, 3, 1, 1)
        # Max Pooling 5 ——>size=batchx256x4x4
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        # Convolution 6 ——>size=batchx512x4x4
        self.double_conv6 = double_conv(start_fm * 16, start_fm * 32, 3, 1, 1)
        # Dropout ——>size=batchx512x4x4
        self.dropout=nn.Dropout(0.5)

        # Transposed Convolution 5
        self.t_conv5 = up_conv(start_fm * 32, start_fm * 16, 3, 1,1,2)
        # Expanding Path Convolution 5
        self.ex_double_conv5 = double_conv(start_fm * 32, start_fm * 16, 3, 1, 1)

        # Transposed Convolution 4
        self.t_conv4 = up_conv(start_fm * 16, start_fm * 8, 3, 1,1,2)
        # Expanding Path Convolution 4
        self.ex_double_conv4 = double_conv(start_fm * 16, start_fm * 8, 3, 1, 1)

        # Transposed Convolution 3
        self.t_conv3 = up_conv(start_fm * 8, start_fm * 4, 3, 1,1,2)
        # Convolution 3
        self.ex_double_conv3 = double_conv(start_fm * 8, start_fm * 4, 3, 1, 1)

        # Transposed Convolution 2
        self.t_conv2 = up_conv(start_fm * 4, start_fm * 2, 3, 1,1,2)
        # Convolution 2
        self.ex_double_conv2 = double_conv(start_fm * 4, start_fm * 2, 3, 1, 1)

        # Transposed Convolution 1
        self.t_conv1 = up_conv(start_fm * 2, start_fm , 3, 1,1,2)
        # Convolution 1
        self.ex_double_conv1 = double_conv(start_fm * 2, start_fm, 3, 1, 1)

        # One by One Conv
        self.one_by_one = nn.Conv2d(start_fm, 1, 1, 1, 0)


    def forward(self, inputs):
        # Contracting Path
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.double_conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        conv5 = self.double_conv5(maxpool4)
        maxpool5 = self.maxpool5(conv5)

        # Bottom
        conv6 = self.double_conv6(maxpool5)
        drop_bottom=self.dropout(conv6)

        # Expanding Path
        t_conv5 = self.t_conv5(drop_bottom)
        cat5 = torch.cat([conv5, t_conv5], 1)
        ex_conv5 = self.ex_double_conv5(cat5)

        t_conv4 = self.t_conv4(ex_conv5)
        cat4 = torch.cat([conv4, t_conv4], 1)
        ex_conv4 = self.ex_double_conv4(cat4)

        t_conv3 = self.t_conv3(ex_conv4)
        cat3 = torch.cat([conv3, t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)

        t_conv2 = self.t_conv2(ex_conv3)
        cat2 = torch.cat([conv2, t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)

        t_conv1 = self.t_conv1(ex_conv2)
        cat1 = torch.cat([conv1, t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)

        one_by_one = self.one_by_one(ex_conv1)
        #one_with_sigmoid=self.final_act(one_by_one)

        return one_by_one
