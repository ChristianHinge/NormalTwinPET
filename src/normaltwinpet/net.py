import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, ch_i, ch_o, dropout=0,kernel_size=3,padding=1,n_conv=2):
        self.sequence = []
        super(conv_block, self).__init__()

        for i in range(n_conv):
            self.sequence += [nn.Conv3d(ch_i, ch_o, padding=padding, kernel_size=kernel_size)]
            self.sequence += [nn.BatchNorm3d(ch_o)]
            self.sequence += [nn.ReLU(True)]
            if dropout:
                self.sequence += [nn.Dropout(dropout)]

            ch_i = ch_o
        self.model = nn.Sequential(*self.sequence)

    def forward(self, x):
        #print("conv",x.shape)
        return self.model(x)


class up_conv(nn.Module):
    def __init__(self, ch_i, ch_o, output_padding=1,kernel_size=3,padding=1):
        self.sequence = []
        super(up_conv, self).__init__()

        self.sequence += [
            nn.ConvTranspose3d(
                ch_i, ch_o, stride=2, padding=padding, kernel_size=kernel_size, output_padding=output_padding
            )
        ]
        self.sequence += [nn.BatchNorm3d(ch_o)]
        self.sequence += [nn.ReLU(True)]

        self.model = nn.Sequential(*self.sequence)

    def forward(self, x, skip):
        a = self.model(x)
        #print("up",x.shape)
        return torch.cat((skip, a), axis=1)

class down_conv(nn.Module):
    def __init__(self, ch_i, ch_o,kernel_size=3,padding=1):
        self.sequence = []
        super(down_conv, self).__init__()
        self.sequence += [nn.Conv3d(ch_i, ch_o, stride=2, padding=padding, kernel_size=kernel_size)]
        self.sequence += [nn.BatchNorm3d(ch_o)]
        self.sequence += [nn.LeakyReLU(0.2, True)]
        self.model = nn.Sequential(*self.sequence)
    def forward(self, x):
        #print("down",x.shape)
        return self.model(x)


class UNet(nn.Module):
    def _calc_pad(self, x):
        residual = (16- x%16)%16
        if residual % 2 == 0:
            return int(residual/2), int(residual/2)
        else:
            return int(residual // 2), int(residual//2 + 1)

    def _get_pad_unpad_funcs(self,calc_pad_from_shape):
        x,y,z = calc_pad_from_shape
        xl,xu = self._calc_pad(x)
        yl,yu = self._calc_pad(y)
        zl,zu = self._calc_pad(z)
        return torch.nn.ReplicationPad3d((zl,zu,yl,yu,xl,xu)), torch.nn.ReplicationPad3d((-zl,-zu,-yl,-yu,-xl,-xu))


    def __init__(self, dropout="normal", n_channels_in = 3,kernel_size=3,n_conv_layers=2,channel_multiplier=1,calc_pad_from_first_batch=False,n_channels_out = 1):
        super(UNet, self).__init__()
        assert dropout in [None, "normal","high"]
        if dropout is None:
            dropout = 0
        elif dropout == "normal":
            dropout = 1
        else:
            dropout = 2

        self.pad_fun = None
        self.unpad_fun = None
        self.calc_pad_from_first_batch = calc_pad_from_first_batch

        cm = channel_multiplier
        self.conv_d1 = conv_block(n_channels_in, 32*cm, dropout=0.1 * dropout,kernel_size=kernel_size,n_conv=n_conv_layers)  # 32, 196, 196

        # self.conv_d1 = conv_block(1,32,dropout=0.1*do) #32, 196, 196
        self.down_1 = down_conv(32*cm, 128*cm,kernel_size=kernel_size,padding=1)  # 16, 48, 48

        self.conv_d2 = conv_block(128*cm, 64*cm, dropout=0.1 * dropout,kernel_size=kernel_size,n_conv=n_conv_layers)  # 16, 48, 48
        self.down_2 = down_conv(64*cm, 256*cm,kernel_size=kernel_size)  # 8, 24, 24

        self.conv_d3 = conv_block(256*cm, 128*cm, dropout=0.2 * dropout,kernel_size=kernel_size,n_conv=n_conv_layers)  # 8, 24, 24
        self.down_3 = down_conv(128*cm, 512*cm,kernel_size=kernel_size)  # 4, 12, 12

        self.conv_d4 = conv_block(512*cm, 256*cm, dropout=0.2 * dropout,kernel_size=kernel_size,n_conv=n_conv_layers)  # 4, 12, 12
        self.down_4 = down_conv(256*cm, 512*cm,kernel_size=kernel_size)  # 2, 6, 6

        ## Bottom
        self.conv_b = conv_block(512*cm, 512*cm, dropout=0.3 * dropout,kernel_size=kernel_size,n_conv=n_conv_layers)  # 2, 6, 6

        ## Going up
        self.up_1 = up_conv(512*cm, 256*cm,kernel_size=kernel_size)  # 4, 12, 12
        self.conv_u1 = conv_block(512*cm, 256*cm, dropout=0.3 * dropout,kernel_size=kernel_size,n_conv=n_conv_layers)  # 4, 12, 12

        self.up_2 = up_conv(256*cm, 128*cm,kernel_size=kernel_size)  # 8, 24, 24
        self.conv_u2 = conv_block(256*cm, 128*cm, dropout=0.2 * dropout,kernel_size=kernel_size,n_conv=n_conv_layers)  # 8, 24, 24

        self.up_3 = up_conv(128*cm, 64*cm,kernel_size=kernel_size)  # 16, 48, 48
        self.conv_u3 = conv_block(128*cm, 64*cm, dropout=0.2 * dropout,kernel_size=kernel_size,n_conv=n_conv_layers)  # 16, 48, 48

        self.up_4 = up_conv(64*cm, 32*cm,kernel_size=kernel_size)  # 32, 196, 196
        self.conv_u4 = conv_block(64*cm, 32*cm, dropout=0.1 * dropout,kernel_size=kernel_size,n_conv=n_conv_layers)  # 32, 196, 196

        self.last = nn.Conv3d(32*cm, n_channels_out, padding=1,kernel_size=kernel_size)  # 32, 196, 196

    def forward(self, x):
        if self.calc_pad_from_first_batch:
            if self.pad_fun is None:
                self.pad_fun, self.unpad_fun = self._get_pad_unpad_funcs(x.shape[-3:])
            x = self.pad_fun(x)
        ## Going down
        c1 = self.conv_d1(x)
        x = self.down_1(c1)

        c2 = self.conv_d2(x)
        x = self.down_2(c2)

        c3 = self.conv_d3(x)
        x = self.down_3(c3)

        c4 = self.conv_d4(x)
        x = self.down_4(c4)

        ## Bottom
        x = self.conv_b(x)

        ## Going up
        x = self.conv_u1(self.up_1(x, c4))
        x = self.conv_u2(self.up_2(x, c3))
        x = self.conv_u3(self.up_3(x, c2))
        x = self.conv_u4(self.up_4(x, c1))

        # x = self.last_sigmoid(self.last(x))
        x = self.last(x)

        if self.calc_pad_from_first_batch:
            x = self.unpad_fun(x)
        return x
