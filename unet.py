from pyvqnet.nn.module import Module
from pyvqnet.nn.conv import Conv2D, ConvT2D
from pyvqnet.nn import activation as F
from pyvqnet.nn.batch_norm import BatchNorm2d
from pyvqnet.tensor import tensor


import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass


class DownsampleLayer(Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                            padding="same")
        self.BatchNorm2d1 = BatchNorm2d(out_ch)
        self.Relu1 = F.ReLu()
        self.conv2 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                            padding="same")
        self.BatchNorm2d2 = BatchNorm2d(out_ch)
        self.Relu2 = F.ReLu()
        self.conv3 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
                            padding=(1,1))
        self.BatchNorm2d3 = BatchNorm2d(out_ch)
        self.Relu3 = F.ReLu()

    def forward(self, x):
        """
        :param x:
        :return: out(Output to deep)，out_2(enter to next level)，
        """
        x1 = self.conv1(x)
        x2 = self.BatchNorm2d1(x1)
        x3 = self.Relu1(x2)
        x4 = self.conv2(x3)
        x5 = self.BatchNorm2d2(x4)
        out = self.Relu2(x5)
        x6 = self.conv3(out)
        x7 = self.BatchNorm2d3(x6)
        out_2 = self.Relu3(x7)
        return out, out_2


class UpSampleLayer(Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampleLayer, self).__init__()

        self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1),
                            padding="same")
        self.BatchNorm2d1 = BatchNorm2d(out_ch * 2)
        self.Relu1 = F.ReLu()
        self.conv2 = Conv2D(input_channels=out_ch * 2, output_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1),
                            padding="same")
        self.BatchNorm2d2 = BatchNorm2d(out_ch * 2)
        self.Relu2 = F.ReLu()

        self.conv3 = ConvT2D(input_channels=out_ch * 2, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
                             padding=(1,1))
        self.BatchNorm2d3 = BatchNorm2d(out_ch)
        self.Relu3 = F.ReLu()

    def forward(self, x):
        '''
        :param x: input conv layer
        :param out: connect with UpsampleLayer
        :return:
        '''
        x = self.conv1(x)
        x = self.BatchNorm2d1(x)
        x = self.Relu1(x)
        x = self.conv2(x)
        x = self.BatchNorm2d2(x)
        x = self.Relu2(x)
        x = self.conv3(x)
        x = self.BatchNorm2d3(x)
        x_out = self.Relu3(x)
        return x_out

# Unet
class UNet(Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels = [2 ** (i + 4) for i in range(5)]

        # DownSampleLayer
        self.d1 = DownsampleLayer(1, out_channels[0])  # 3-64
        self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
        self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
        self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
        # UpSampleLayer
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
        self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
        # output
        self.conv1 = Conv2D(input_channels=out_channels[1], output_channels=out_channels[0], kernel_size=(3, 3),
                            stride=(1, 1), padding="same")
        self.BatchNorm2d1 = BatchNorm2d(out_channels[0])
        self.Relu1 = F.ReLu()
        self.conv2 = Conv2D(input_channels=out_channels[0], output_channels=out_channels[0], kernel_size=(3, 3),
                            stride=(1, 1), padding="same")
        self.BatchNorm2d2 = BatchNorm2d(out_channels[0])
        self.Relu2 = F.ReLu()
        self.conv3 = Conv2D(input_channels=out_channels[0], output_channels=1, kernel_size=(3, 3),
                            stride=(1, 1), padding="same")
        self.Sigmoid = F.Sigmoid()


    def forward(self, x):
        out_1, out1 = self.d1(x)
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)

        out5 = self.u1(out4)
        out5_pad_out4 = tensor.pad2d(out5, (1, 0, 1, 0), 0)
        cat_out5 = tensor.concatenate([out5_pad_out4, out_4], axis=1)

        out6 = self.u2(cat_out5)
        out6_pad_out_3 = tensor.pad2d(out6, (1, 0, 1, 0), 0)
        cat_out6 = tensor.concatenate([out6_pad_out_3, out_3], axis=1)

        out7 = self.u3(cat_out6)
        out7_pad_out_2 = tensor.pad2d(out7, (1, 0, 1, 0), 0)
        cat_out7 = tensor.concatenate([out7_pad_out_2, out_2], axis=1)

        out8 = self.u4(cat_out7)
        out8_pad_out_1 = tensor.pad2d(out8, (1, 0, 1, 0), 0)
        cat_out8 = tensor.concatenate([out8_pad_out_1, out_1], axis=1)

        out = self.conv1(cat_out8)
        out = self.BatchNorm2d1(out)
        out = self.Relu1(out)
        out = self.conv2(out)
        out = self.BatchNorm2d2(out)
        out = self.Relu2(out)
        out = self.conv3(out)
        out = self.Sigmoid(out)
        return out
