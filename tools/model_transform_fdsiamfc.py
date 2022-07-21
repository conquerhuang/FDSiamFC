import torch
from siamfc.fdsiamfc import TrackerFDSiamFC
import scipy.io as scio

model_name = r'../pretrained/SiamFC_new/FDSiamFC_20_8_0_39.pth'

squeeze_rate = 0.2
tracker = TrackerFDSiamFC(net_path=None, name='FDSiamFC', squeeze_rate=squeeze_rate)

saved_model = torch.load(model_name)

# conv1
cv1_w = saved_model['backbone.conv1.0.weight']
cv1_b = saved_model['backbone.conv1.0.bias']
cv1_s_w = saved_model['backbone.conv1_squeeze.0.weight']
cv1_s_b = saved_model['backbone.conv1_squeeze.0.bias']
# conv2
cv2_w = saved_model['backbone.conv2.0.weight']
cv2_b = saved_model['backbone.conv2.0.bias']
cv2_s_w = saved_model['backbone.conv2_squeeze.0.weight']
cv2_s_b = saved_model['backbone.conv2_squeeze.0.bias']
# conv3
cv3_w = saved_model['backbone.conv3.0.weight']
cv3_b = saved_model['backbone.conv3.0.bias']
cv3_s_w = saved_model['backbone.conv3_squeeze.0.weight']
cv3_s_b = saved_model['backbone.conv3_squeeze.0.bias']
# conv4
cv4_w = saved_model['backbone.conv4.0.weight']
cv4_b = saved_model['backbone.conv4.0.bias']
cv4_s_w = saved_model['backbone.conv4_squeeze.0.weight']
cv4_s_b = saved_model['backbone.conv4_squeeze.0.bias']

fd_w = {}

fd_w['cv1_w'] = cv1_w.data.cpu().numpy()
fd_w['cv1_b'] = cv1_b.data.cpu().numpy()
fd_w['cv1_s_w'] = cv1_s_w.data.cpu().numpy()
fd_w['cv1_s_b'] = cv1_s_b.data.cpu().numpy()

fd_w['cv2_w'] = cv2_w.data.cpu().numpy()
fd_w['cv2_b'] = cv2_b.data.cpu().numpy()
fd_w['cv2_s_w'] = cv2_s_w.data.cpu().numpy()
fd_w['cv2_s_b'] = cv2_s_b.data.cpu().numpy()

fd_w['cv3_w'] = cv3_w.data.cpu().numpy()
fd_w['cv3_b'] = cv3_b.data.cpu().numpy()
fd_w['cv3_s_w'] = cv3_s_w.data.cpu().numpy()
fd_w['cv3_s_b'] = cv3_s_b.data.cpu().numpy()

fd_w['cv4_w'] = cv4_w.data.cpu().numpy()
fd_w['cv4_b'] = cv4_b.data.cpu().numpy()
fd_w['cv4_s_w'] = cv4_s_w.data.cpu().numpy()
fd_w['cv4_s_b'] = cv4_s_b.data.cpu().numpy()

scio.savemat('fd_w.mat', {'fd_w': fd_w})

model_dict = tracker.net.state_dict()
state_dict = {}
trans_model = scio.loadmat(r'./trans_model.mat')

# 从训练好的模型中迁移数据。
conv1_weight = torch.from_numpy(trans_model['cv1_w']).to(tracker.device)
conv1_bias = torch.from_numpy(trans_model['cv1_b'].squeeze(axis=0)).to(tracker.device)
conv1_bn_w = saved_model['backbone.bn1.0.weight']
conv1_bn_b = saved_model['backbone.bn1.0.bias']
conv1_mean = saved_model['backbone.bn1.0.running_mean']
conv1_var  = saved_model['backbone.bn1.0.running_var']

conv2_weight = torch.from_numpy(trans_model['cv2_w']).to(tracker.device)
conv2_bias = torch.from_numpy(trans_model['cv2_b'].squeeze(axis=0)).to(tracker.device)
conv2_bn_w = saved_model['backbone.bn2.0.weight']
conv2_bn_b = saved_model['backbone.bn2.0.bias']
conv2_mean = saved_model['backbone.bn2.0.running_mean']
conv2_var  = saved_model['backbone.bn2.0.running_var']

conv3_weight = torch.from_numpy(trans_model['cv3_w']).to(tracker.device)
conv3_bias = torch.from_numpy(trans_model['cv3_b'].squeeze(axis=0)).to(tracker.device)
conv3_bn_w = saved_model['backbone.bn3.0.weight']
conv3_bn_b = saved_model['backbone.bn3.0.bias']
conv3_mean = saved_model['backbone.bn3.0.running_mean']
conv3_var  = saved_model['backbone.bn3.0.running_var']

conv4_weight = torch.from_numpy(trans_model['cv4_w']).to(tracker.device)
conv4_bias = torch.from_numpy(trans_model['cv4_b'].squeeze(axis=0)).to(tracker.device)
conv4_bn_w = saved_model['backbone.bn4.0.weight']
conv4_bn_b = saved_model['backbone.bn4.0.bias']
conv4_mean = saved_model['backbone.bn4.0.running_mean']
conv4_var  = saved_model['backbone.bn4.0.running_var']

conv5_weight = saved_model['backbone.conv5.0.weight']
conv5_bias = saved_model['backbone.conv5.0.bias']

state_dict['backbone.conv1.0.weight'] = conv1_weight
state_dict['backbone.conv1.0.bias'] = conv1_bias
state_dict['backbone.conv1.1.weight'] = conv1_bn_w
state_dict['backbone.conv1.1.bias'] = conv1_bn_b
state_dict['backbone.conv1.1.running_mean'] = conv1_mean
state_dict['backbone.conv1.1.running_var'] = conv1_var

state_dict['backbone.conv2.0.weight'] = conv2_weight
state_dict['backbone.conv2.0.bias'] = conv2_bias
state_dict['backbone.conv2.1.weight'] = conv2_bn_w
state_dict['backbone.conv2.1.bias'] = conv2_bn_b
state_dict['backbone.conv2.1.running_mean'] = conv2_mean
state_dict['backbone.conv2.1.running_var'] = conv2_var

state_dict['backbone.conv3.0.weight'] = conv3_weight
state_dict['backbone.conv3.0.bias'] = conv3_bias
state_dict['backbone.conv3.1.weight'] = conv3_bn_w
state_dict['backbone.conv3.1.bias'] = conv3_bn_b
state_dict['backbone.conv3.1.running_mean'] = conv3_mean
state_dict['backbone.conv3.1.running_var'] = conv3_var

state_dict['backbone.conv4.0.weight'] = conv4_weight
state_dict['backbone.conv4.0.bias'] = conv4_bias
state_dict['backbone.conv4.1.weight'] = conv4_bn_w
state_dict['backbone.conv4.1.bias'] = conv4_bn_b
state_dict['backbone.conv4.1.running_mean'] = conv4_mean
state_dict['backbone.conv4.1.running_var'] = conv4_var

state_dict['backbone.conv5.0.weight'] = conv5_weight
state_dict['backbone.conv5.0.bias'] = conv5_bias

model_dict.update(state_dict)
tracker.net.load_state_dict(model_dict)
torch.save(tracker.net.state_dict(), 'transformed_model.pth')














