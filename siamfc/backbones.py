from __future__ import absolute_import

import torch.nn as nn


__all__ = ['AlexNetV1', 'FDAlexNet', 'FDAlexNetLite']


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class _AlexNet(nn.Module):
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


class FDAlexNetLite(nn.Module):
    def __init__(self, squeeze_rate=None):
        super(FDAlexNetLite, self).__init__()
        # 预设值压缩率参数
        if squeeze_rate is None:
            squeeze_rate = [1, 1, 1, 1, 1]
        self.squeeze_rate = squeeze_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(int(self.squeeze_rate[0]*96/2)*2), 11, 2),
            _BatchNorm2d(int(int(self.squeeze_rate[0]*96/2)*2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(int(self.squeeze_rate[0]*96/2)*2), int(int(self.squeeze_rate[1]*256/2)*2), 5, 1, groups=2),
            _BatchNorm2d(int(int(self.squeeze_rate[1]*256/2)*2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(int(self.squeeze_rate[1]*256/2)*2), int(int(self.squeeze_rate[2]*384/2)*2), 3, 1),
            _BatchNorm2d(int(int(self.squeeze_rate[2]*384/2)*2)),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(int(int(self.squeeze_rate[2]*384/2)*2), int(int(self.squeeze_rate[3]*384/2)*2), 3, 1, groups=2),
            _BatchNorm2d(int(int(self.squeeze_rate[3]*384/2)*2)),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(int(int(self.squeeze_rate[3]*384/2)*2), int(int(self.squeeze_rate[4]*256/2)*2), 3, 1, groups=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class FDAlexNet(nn.Module):
    def __init__(self, squeeze_rate=None, step=0):
        super(FDAlexNet, self).__init__()

        if squeeze_rate is None:
            squeeze_rate = [1, 1, 1, 1, 1]
        self.squeeze_rate = squeeze_rate
        self.step = step

        # conv1
        self.conv1 = self._make_conv_layer(3, 96, 11, 2, 1)

        # conv2
        if step <= 1:
            self.bn1 = self._make_bn_layer(96, True, True)
            self.conv2 = self._make_conv_layer(96, 256, 5, 1, 2)
        else:
            self.bn1 = self._make_bn_layer(int(int(self.squeeze_rate[0]*96/2)*2), True, True)
            self.conv2 = self._make_conv_layer(int(int(self.squeeze_rate[0]*96/2)*2), 256, 5, 1, 2)

        # conv3
        if step <= 3:
            self.bn2 = self._make_bn_layer(256, True, True)
            self.conv3 = self._make_conv_layer(256, 384, 3, 1, 1)
        else:
            self.bn2 = self._make_bn_layer(int(int(self.squeeze_rate[1]*256/2)*2), True, True)
            self.conv3 = self._make_conv_layer(int(int(self.squeeze_rate[1]*256/2)*2), 384, 3, 1, 1)

        # conv4
        if step <= 5:
            self.bn3 = self._make_bn_layer(384, True, False)
            self.conv4 = self._make_conv_layer(384, 384, 3, 1, 2)
        else:
            self.bn3 = self._make_bn_layer(int(int(self.squeeze_rate[2]*384/2)*2), True, False)
            self.conv4 = self._make_conv_layer(int(int(self.squeeze_rate[2]*384/2)*2), 384, 3, 1, 2)

        if step <= 7:
            self.bn4 = self._make_bn_layer(384, True, False)
            self.conv5 = self._make_conv_layer(384, 256, 3, 1, 2)
        else:
            self.bn4 = self._make_bn_layer(int(int(self.squeeze_rate[3]*384/2)*2), True, False)
            self.conv5 = self._make_conv_layer(int(int(self.squeeze_rate[3]*384/2)*2), 256, 3, 1, 2)

        self.con1_squeeze, self.con1_combine = self._make_squeeze_combine_layer(
            channel=96, squeeze_rate=self.squeeze_rate[0])
        self.con2_squeeze, self.con2_combine = self._make_squeeze_combine_layer(
            channel=256, group=2, squeeze_rate=self.squeeze_rate[1])
        self.con3_squeeze, self.con3_combine = self._make_squeeze_combine_layer(
            channel=384, squeeze_rate=self.squeeze_rate[2])
        self.con4_squeeze, self.con4_combine = self._make_squeeze_combine_layer(
            channel=384, group=2, squeeze_rate=self.squeeze_rate[3])
        self.con5_squeeze, _ = self._make_squeeze_combine_layer(
            channel=256, group=2, squeeze_rate=self.squeeze_rate[4])

    def _make_conv_layer(self, in_channel=256, out_channel=256, kernel=3, stride=1, group=1):
        conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride, groups=group))
        return conv

    def _make_bn_layer(self, channel=256, relu=True, maxpool=False):
        bn = [_BatchNorm2d(channel)]
        if relu:
            bn.append(
                nn.ReLU(inplace=True))
        if maxpool:
            bn.append(
                nn.MaxPool2d(3, 2))
        return nn.Sequential(*bn)

    def _make_squeeze_combine_layer(self, channel=256, group=1, squeeze_rate=1):
        squeeze = nn.Sequential(
            nn.Conv2d(channel, int(int(squeeze_rate*channel/2)*2), 1, 1, groups=group)
        )
        combine = nn.Sequential(
            nn.Conv2d(int(int(squeeze_rate*channel/2)*2), channel, 1, 1, groups=group)
        )
        return squeeze, combine

    def forward(self, x):
        x = self.conv1(x)

        if self.step == 1:
            x = self.con1_squeeze(x)
            x = self.con1_combine(x)
        elif self.step >= 2:
            x = self.con1_squeeze(x)

        x = self.bn1(x)
        x = self.conv2(x)

        if self.step == 3:
            x = self.con2_squeeze(x)
            x = self.con2_combine(x)
        elif self.step >= 4:
            x = self.con2_squeeze(x)

        x = self.bn2(x)
        x = self.conv3(x)

        if self.step == 5:
            x = self.con3_squeeze(x)
            x = self.con3_combine(x)
        elif self.step >= 6:
            x = self.con3_squeeze(x)

        x = self.bn3(x)
        x = self.conv4(x)

        if self.step == 7:
            x = self.con4_squeeze(x)
            x = self.con4_combine(x)
        elif self.step >= 8:
            x = self.con4_squeeze(x)

        x = self.bn4(x)
        x = self.conv5(x)

        if self.step == 9:
            x = self.con5_squeeze(x)

        return x

