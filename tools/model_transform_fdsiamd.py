import torch
from siamfc import TrackerFDSiamFC_8l_lite
import scipy.io as scio


def single_kernel_simplify(conv_weight, conv_bias, squeeze_weight, squeeze_bias):
    # 在没有分组的情况下，使用如下部分代码可以实现压缩后的卷积层的转换。
    sw = squeeze_weight.permute(2, 3, 0, 1)
    cw = conv_weight.permute(2, 3, 0, 1)
    weight = torch.matmul(sw, cw)
    weight = weight.permute(2, 3, 0, 1)

    sw = squeeze_weight.permute(2, 3, 0, 1)
    bias1 = torch.matmul(sw, conv_bias)
    bias = bias1.squeeze(0).squeeze(0) + squeeze_bias
    return weight, bias


def kernel_simplify(conv_weight, conv_bias, squeeze_weight, squeeze_bias, groups=-1):
    # 提升精度，防止模型退化。
    # use float64 precision or the transformed model is possibly to loss precision around 1%
    conv_weight = conv_weight.type(torch.float64)
    conv_bias = conv_bias.type(torch.float64)
    squeeze_weight = squeeze_weight.type(torch.float64)
    squeeze_bias = squeeze_bias.type(torch.float64)
    if groups == 1:
        weight, bias = single_kernel_simplify(conv_weight, conv_bias, squeeze_weight, squeeze_bias)
    elif groups == 2:
        # 先按照分组情况将卷积核拆开
        # unpack the group kernels and transform them separate, then concatenate them.
        num_in_channel = conv_weight.shape[0]
        num_out_channel = squeeze_weight.shape[0]
        conv_weight0 = conv_weight[0:int(num_in_channel/2), :, :, :]
        conv_weight1 = conv_weight[int(num_in_channel/2):num_in_channel, :, :, :]
        conv_bias0 = conv_bias[0:int(num_in_channel/2)]
        conv_bias1 = conv_bias[int(num_in_channel/2):num_in_channel]
        squeeze_weight0 = squeeze_weight[0:int(num_out_channel/2), :, :, :]
        squeeze_weight1 = squeeze_weight[int(num_out_channel/2):num_out_channel, :, :, :]
        squeeze_bias0 = squeeze_bias[0:int(num_out_channel/2)]
        squeeze_bias1 = squeeze_bias[int(num_out_channel/2):num_out_channel]

        weight0, bias0 = single_kernel_simplify(conv_weight0, conv_bias0, squeeze_weight0, squeeze_bias0)
        weight1, bias1 = single_kernel_simplify(conv_weight1, conv_bias1, squeeze_weight1, squeeze_bias1)
        weight = torch.cat((weight0, weight1), 0)
        bias = torch.cat((bias0, bias1), 0)
    else:
        print('illegal input of groups')
        return
    weight = weight.type(torch.float32)
    bias = bias.type(torch.float32)
    return weight, bias


def model_transform(model_path=None, squeeze_rate=None, save_name=None):
    if model_path is None:
        model_path = r'../pretrained/SiamFC_new/Finetune_0_15_0_38.pth'  # 用于转换的模型的路径
    if squeeze_rate is None:
        squeeze_rate = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    if save_name is None:
        save_name = 'transformed_model_double.pth'

    tracker = TrackerFDSiamFC_8l_lite(net_path=None, name='FDSiamFC_8l_lite', squeeze_rate=squeeze_rate)
    model_dict = tracker.net.state_dict()

    saved_model = torch.load(model_path)
    state_dict = {}

    # conv1
    conv_weight = saved_model['backbone.conv1.0.weight']
    conv_bias = saved_model['backbone.conv1.0.bias']
    squeeze_weight = saved_model['backbone.con1_squeeze.0.weight']
    squeeze_bias = saved_model['backbone.con1_squeeze.0.bias']
    conv1_weight, conv1_bias = kernel_simplify(conv_weight, conv_bias, squeeze_weight, squeeze_bias, groups=1)
    conv1_bn_w = saved_model['backbone.bn1.0.weight']
    conv1_bn_b = saved_model['backbone.bn1.0.bias']
    conv1_mean = saved_model['backbone.bn1.0.running_mean']
    conv1_var  = saved_model['backbone.bn1.0.running_var']

    # conv2
    conv_weight = saved_model['backbone.conv2.0.weight']
    conv_bias = saved_model['backbone.conv2.0.bias']
    squeeze_weight = saved_model['backbone.con2_squeeze.0.weight']
    squeeze_bias = saved_model['backbone.con2_squeeze.0.bias']
    conv2_weight, conv2_bias = kernel_simplify(conv_weight, conv_bias, squeeze_weight, squeeze_bias, groups=1)
    conv2_bn_w = saved_model['backbone.bn2.0.weight']
    conv2_bn_b = saved_model['backbone.bn2.0.bias']
    conv2_mean = saved_model['backbone.bn2.0.running_mean']
    conv2_var  = saved_model['backbone.bn2.0.running_var']

    # conv3
    conv_weight = saved_model['backbone.conv3.0.weight']
    conv_bias = saved_model['backbone.conv3.0.bias']
    squeeze_weight = saved_model['backbone.con3_squeeze.0.weight']
    squeeze_bias = saved_model['backbone.con3_squeeze.0.bias']
    conv3_weight, conv3_bias= kernel_simplify(conv_weight, conv_bias, squeeze_weight, squeeze_bias, groups=1)
    conv3_bn_w = saved_model['backbone.bn3.0.weight']
    conv3_bn_b = saved_model['backbone.bn3.0.bias']
    conv3_mean = saved_model['backbone.bn3.0.running_mean']
    conv3_var  = saved_model['backbone.bn3.0.running_var']

    # conv4
    conv_weight = saved_model['backbone.conv4.0.weight']
    conv_bias = saved_model['backbone.conv4.0.bias']
    squeeze_weight = saved_model['backbone.con4_squeeze.0.weight']
    squeeze_bias = saved_model['backbone.con4_squeeze.0.bias']
    conv4_weight, conv4_bias = kernel_simplify(conv_weight, conv_bias, squeeze_weight, squeeze_bias, groups=2)
    conv4_bn_w = saved_model['backbone.bn4.0.weight']
    conv4_bn_b = saved_model['backbone.bn4.0.bias']
    conv4_mean = saved_model['backbone.bn4.0.running_mean']
    conv4_var  = saved_model['backbone.bn4.0.running_var']

    # conv5
    conv_weight = saved_model['backbone.conv5.0.weight']
    conv_bias = saved_model['backbone.conv5.0.bias']
    squeeze_weight = saved_model['backbone.con5_squeeze.0.weight']
    squeeze_bias = saved_model['backbone.con5_squeeze.0.bias']
    conv5_weight, conv5_bias = kernel_simplify(conv_weight, conv_bias, squeeze_weight, squeeze_bias, groups=2)
    conv5_bn_w = saved_model['backbone.bn5.0.weight']
    conv5_bn_b = saved_model['backbone.bn5.0.bias']
    conv5_mean = saved_model['backbone.bn5.0.running_mean']
    conv5_var  = saved_model['backbone.bn5.0.running_var']

    # conv6
    conv_weight = saved_model['backbone.conv6.0.weight']
    conv_bias = saved_model['backbone.conv6.0.bias']
    squeeze_weight = saved_model['backbone.con6_squeeze.0.weight']
    squeeze_bias = saved_model['backbone.con6_squeeze.0.bias']
    conv6_weight, conv6_bias = kernel_simplify(conv_weight, conv_bias, squeeze_weight, squeeze_bias, groups=1)
    conv6_bn_w = saved_model['backbone.bn6.0.weight']
    conv6_bn_b = saved_model['backbone.bn6.0.bias']
    conv6_mean = saved_model['backbone.bn6.0.running_mean']
    conv6_var  = saved_model['backbone.bn6.0.running_var']

    # conv7
    conv_weight = saved_model['backbone.conv7.0.weight']
    conv_bias = saved_model['backbone.conv7.0.bias']
    squeeze_weight = saved_model['backbone.con7_squeeze.0.weight']
    squeeze_bias = saved_model['backbone.con7_squeeze.0.bias']
    conv7_weight, conv7_bias = kernel_simplify(conv_weight, conv_bias, squeeze_weight, squeeze_bias, groups=2)
    conv7_bn_w = saved_model['backbone.bn7.0.weight']
    conv7_bn_b = saved_model['backbone.bn7.0.bias']
    conv7_mean = saved_model['backbone.bn7.0.running_mean']
    conv7_var  = saved_model['backbone.bn7.0.running_var']

    # conv8
    conv_weight = saved_model['backbone.conv8.0.weight']
    conv_bias = saved_model['backbone.conv8.0.bias']
    squeeze_weight = saved_model['backbone.con8_squeeze.0.weight']
    squeeze_bias = saved_model['backbone.con8_squeeze.0.bias']
    conv8_weight, conv8_bias = kernel_simplify(conv_weight, conv_bias, squeeze_weight, squeeze_bias, groups=2)

    # 将计算得到的新的权重写入到新的模型当中。
    # conv1
    state_dict['backbone.conv1.0.weight'] = conv1_weight
    state_dict['backbone.conv1.0.bias'] = conv1_bias
    state_dict['backbone.conv1.1.weight'] = conv1_bn_w
    state_dict['backbone.conv1.1.bias'] = conv1_bn_b
    state_dict['backbone.conv1.1.running_mean'] = conv1_mean
    state_dict['backbone.conv1.1.running_var'] = conv1_var

    # conv2
    state_dict['backbone.conv2.0.weight'] = conv2_weight
    state_dict['backbone.conv2.0.bias'] = conv2_bias
    state_dict['backbone.conv2.1.weight'] = conv2_bn_w
    state_dict['backbone.conv2.1.bias'] = conv2_bn_b
    state_dict['backbone.conv2.1.running_mean'] = conv2_mean
    state_dict['backbone.conv2.1.running_var'] = conv2_var

    # conv3
    state_dict['backbone.conv3.0.weight'] = conv3_weight
    state_dict['backbone.conv3.0.bias'] = conv3_bias
    state_dict['backbone.conv3.1.weight'] = conv3_bn_w
    state_dict['backbone.conv3.1.bias'] = conv3_bn_b
    state_dict['backbone.conv3.1.running_mean'] = conv3_mean
    state_dict['backbone.conv3.1.running_var'] = conv3_var

    # conv4
    state_dict['backbone.conv4.0.weight'] = conv4_weight
    state_dict['backbone.conv4.0.bias'] = conv4_bias
    state_dict['backbone.conv4.1.weight'] = conv4_bn_w
    state_dict['backbone.conv4.1.bias'] = conv4_bn_b
    state_dict['backbone.conv4.1.running_mean'] = conv4_mean
    state_dict['backbone.conv4.1.running_var'] = conv4_var

    # conv5
    state_dict['backbone.conv5.0.weight'] = conv5_weight
    state_dict['backbone.conv5.0.bias'] = conv5_bias
    state_dict['backbone.conv5.1.weight'] = conv5_bn_w
    state_dict['backbone.conv5.1.bias'] = conv5_bn_b
    state_dict['backbone.conv5.1.running_mean'] = conv5_mean
    state_dict['backbone.conv5.1.running_var'] = conv5_var

    # conv6
    state_dict['backbone.conv6.0.weight'] = conv6_weight
    state_dict['backbone.conv6.0.bias'] = conv6_bias
    state_dict['backbone.conv6.1.weight'] = conv6_bn_w
    state_dict['backbone.conv6.1.bias'] = conv6_bn_b
    state_dict['backbone.conv6.1.running_mean'] = conv6_mean
    state_dict['backbone.conv6.1.running_var'] = conv6_var

    # conv7
    state_dict['backbone.conv7.0.weight'] = conv7_weight
    state_dict['backbone.conv7.0.bias'] = conv7_bias
    state_dict['backbone.conv7.1.weight'] = conv7_bn_w
    state_dict['backbone.conv7.1.bias'] = conv7_bn_b
    state_dict['backbone.conv7.1.running_mean'] = conv7_mean
    state_dict['backbone.conv7.1.running_var'] = conv7_var

    # conv8
    state_dict['backbone.conv8.0.weight'] = conv8_weight
    state_dict['backbone.conv8.0.bias'] = conv8_bias

    # update model and save the transformed model.
    model_dict.update(state_dict)
    tracker.net.load_state_dict(model_dict)
    torch.save(tracker.net.state_dict(), save_name)


if __name__ == '__main__':
    for i in range(2, 52):
        model_path = r'../pretrained/SiamFC_new/FDSiamFC8L_0_15_0_' + str(i) + '.pth'
        squeeze_rate = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        save_name = r'../pretrained/transformed_model/FDSiam_8l_'+str(i)+'.pth'
        model_transform(model_path, squeeze_rate, save_name)










