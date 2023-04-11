import time
import random
random.seed(a=None, version=2)
from find_layer_pt import find_densenet_layer, find_resnet_layer, find_vgg_layer # 导入自定义的函数，用于找到不同模型的目标层

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import transforms

lr = 1e-4 # 学习率
epochs = 15 # 训练轮数
H, W = 224, 224 # 图像高度和宽度

transform = transforms.Compose(
        [
                #transforms.Resize((H, W)),
                transforms.ToTensor(), # 将图像转换为张量
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 对图像进行归一化处理
        ])

def load_image(path, preprocess=False):

    """Load and preprocess image."""

    img = Image.open(path).convert('RGB').resize((H, W)) # 打开图像，转换为RGB格式，并调整大小
    
    if preprocess:
        
        img = transform(img) # 对图像进行预处理
        
        img = img.unsqueeze(0) # 在第一个维度上增加一个维度，用于表示批次大小

    return img

def build_model(arch):
    model = models.__dict__[arch](pretrained=True) # 根据参数选择预训练的模型
    model = model.eval() # 将模型设置为评估模式，不更新参数
    model = model.cuda() # 将模型移动到GPU上
    return model
        
class GradCAM(object): # 定义GradCAM类，用于实现梯度加权类激活映射（Grad-CAM）
    def __init__(self, model, layer_name):
        self.model = model # 模型
        self.gradients = dict() # 梯度字典，用于存储目标层的梯度值
        self.activations = dict() # 激活字典，用于存储目标层的激活值
        def backward_hook(module, grad_input, grad_output): # 定义反向钩子函数，用于获取目标层的梯度值
        
            self.gradients['value'] = grad_output[0].detach() # 将梯度值从计算图中分离，并存储到字典中
        
            return None
        
        def forward_hook(module, input, output): # 定义前向钩子函数，用于获取目标层的激活值
        
            self.activations['value'] = output.detach() # 将激活值从计算图中分离，并存储到字典中
        
            return None
        if self.model.__class__.__name__ == 'ResNet': # 根据模型的类名，选择不同的函数来找到目标层
            target_layer = find_resnet_layer(self.model, layer_name)
        # elif self.model.__class__.__name__ == 'VGG':
        #     target_layer = find_vgg_layer(self.model, layer_name)
        # elif self.model.__class__.__name__ == 'DenseNet':
        #     target_layer = find_densenet_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook) # 在目标层上注册前向钩子函数
        target_layer.register_backward_hook(backward_hook) # 在目标层上注册反向钩子函数

    def forward(self, img, cls, retain_graph=False): # 定义前向传播函数，输入图像和类别，输出模型预测结果、类别、权重、ReLU后的权重和激活值
        logit = self.model(img) # 将图像输入模型，得到输出结果（未经过softmax）
        if cls == -1: # 如果类别为-1，表示没有指定类别，则选择模型预测的最大概率的类别
            cls = logit.max(1)[-1] # 得到最大概率的类别索引
            score = logit[:, cls].squeeze() # 得到最大概率的类别的分数
        else: # 如果类别不为-1，表示有指定类别，则选择指定的类别
            score = logit[:, cls].squeeze() # 得到指定类别的分数
        self.model.zero_grad() # 将模型的梯度清零
        score.backward(retain_graph=retain_graph) # 对分数进行反向传播，计算梯度值
        gradients = self.gradients['value'] # 从字典中获取目标层的梯度值
        activations = self.activations['value'] # 从字典中获取目标层的激活值
        b, k, u, v = gradients.size() # 获取梯度值的形状，b为批次大小，k为通道数，u和v为高度和宽度
        weights = gradients.view(b, k, -1).mean(2).view(b, k, 1, 1) # 将梯度值在空间维度上求平均，得到每个通道的权重
        weights_relu = F.relu(weights) # 对权重进行ReLU操作，去除负值
        return logit, cls, weights, weights_relu, activations # 返回模型预测结果、类别、权重、ReLU后的权重和激活值
    
    def fowrad_acmp(self, img, retain_graph=False): # 定义另一个前向传播函数，输入图像，输出模型预测结果和激活值
        logit = self.model(img) # 将图像输入模型，得到输出结果（未经过softmax）
        activations = self.activations['value'] # 从字典中获取目标层的激活值
        return logit, activations # 返回模型预测结果和激活值
    
    def saliency(self, weights, activations):   # 定义生成显著性图的函数，输入权重和激活值，输出显著性图
        saliency_map = (weights * activations).sum(1, keepdim=True) # 将权重和激活值相乘，并在通道维度上求和，得到显著性图
        saliency_map = F.relu(saliency_map) # 对显著性图进行ReLU操作，去除负值
        saliency_map = F.upsample(saliency_map, size=(H, W), mode='bilinear', align_corners=False) # 对显著性图进行上采样，调整到原始图像大小
        saliency_map = saliency_map / saliency_map.max()  # 对显著性图进行归一化处理
        
        return saliency_map # 返回显著性图
    
    def __call__(self, img, cls=-1, retain_graph=True): # 定义调用函数，输入图像和类别，输出前向传播函数的结果

        return self.forward(img, cls, retain_graph) # 调用前向传播函数

def calc_smoothness_loss(mask, power=2, border_penalty=0.1): # 定义计算平滑损失的函数，输入[mask 为一个热力图，power 为损失的幂次方，border_penalty 为边界处的惩罚系数]，输出平滑损失
    ''' For a given image this loss should be more or less invariant to image resize when using power=2...
        let L be the length of a side
        EdgesLength ~ L
        EdgesSharpness ~ 1/L, easy to see if you imagine just a single vertical edge in the whole image'''
    x_loss = torch.sum((torch.abs(mask[:,:,1:,:] - mask[:,:,:-1,:]))**power)
    y_loss = torch.sum((torch.abs(mask[:,:,:,1:] - mask[:,:,:,:-1]))**power)
    if border_penalty>0:
        border = float(border_penalty)*torch.sum(mask[:,:,-1,:]**power + mask[:,:,0,:]**power + mask[:,:,:,-1]**power + mask[:,:,:,0]**power)
    else:
        border = 0.
    return (x_loss + y_loss + border) / float(power * mask.size(0))  # watch out, normalised by the batch size!

# 定义一个函数，计算给定图像和类别的显著性
def compuate_sal(model, img_path, cls):
    # 加载图像，并进行预处理和转换为cuda张量
    img = load_image(img_path, preprocess=True).cuda()
    
    # 使用gradcam方法获取图像的logit，类别，权重，激活值
    logit, cls, weights, weights_relu, activations = gradcam(img, cls) 
    # 使用softmax函数计算logit的概率，并取出目标类别的概率值
    predictions = F.softmax(logit, dim=1)[0][cls]
    
    # 将权重转换为可训练的变量，并赋值给trainable_weights
    trainable_weights = weights_relu
    trainable_weights = Variable(trainable_weights, requires_grad=True).cuda()
    # 计算目标类别的概率值的对数的负数，作为alpha值
    alpha = 1-torch.log(predictions)

    # 定义一个优化器，使用Adam算法，学习率为lr
    optimizer = torch.optim.Adam([trainable_weights], lr=lr)
    # 进行epochs次迭代
    for epoch in range(epochs):    
        # 使用gradcam方法计算显著性图
        cam = gradcam.saliency(trainable_weights, activations)
        # 计算显著性图的平滑度损失，边界惩罚为0.3
        smooth_loss = calc_smoothness_loss(cam, border_penalty=0.3)
        # 将显著性图复制三次，并扩展维度
        cam_rp = cam.repeat(1,3,1,1)
        # 计算显著性图的反向图，即1减去显著性图
        cam_rp_inverse = torch.ones_like(cam_rp) - cam_rp
        # 将原始图像和显著性图相乘，得到显著区域的图像
        prod_img = img * cam_rp
        # 将原始图像和反向显著性图相乘，得到非显著区域的图像
        prod_img_inverse = img * cam_rp_inverse
        # 使用gradcam方法计算显著区域图像的logit和激活值
        logit_prod, activations_prod = gradcam.fowrad_acmp(prod_img)
        # 使用gradcam方法计算非显著区域图像的logit和激活值
        logit_prod_inverse, activations_prod_inverse = gradcam.fowrad_acmp(prod_img_inverse)
        # 计算得分，包括alpha值，logit值，平滑度损失，权重和激活值的差异，显著性图和反向显著性图的差异等因素
        score = -alpha*logit_prod[:, cls].squeeze() + logit_prod_inverse[:, cls].squeeze() + smooth_loss - 50*(weights_relu*(activations_prod - activations_prod_inverse)).norm() - 0.05*(cam_rp - cam_rp_inverse).norm()
        # 清空优化器的梯度缓存
        optimizer.zero_grad()
        # 反向传播计算梯度
        score.backward(retain_graph=True)
        # 更新权重参数
        optimizer.step()

    # 迭代结束后，再次使用gradcam方法计算最终的显著性图
    cam = gradcam.saliency(trainable_weights, activations)
    # 将目标类别的概率值转换为numpy数组，并赋值给pred_probablity
    pred_probablity = predictions.detach().cpu().numpy()
    # 将最终的显著性图转换为numpy数组，并赋值给expl
    expl = cam.repeat(1,3,1,1).squeeze().detach().cpu().numpy()
    
    # 定义一个空列表，用于存储预测值的变化
    pred_y = []
    # 将原始图像复制一份，并展平为一维向量
    X_copy = img.clone().flatten()
    # 定义一个计数器，从0.01开始，每次增加0.01，直到1.0
    count = 0.01
    while(count <= 1.0):
        # 根据计数器的值，计算要删除的像素的数量
        thres_hold = int(H * W * 3 * count)
        # 根据显著性图的值，找出要删除的像素的索引
        delete_index = np.argpartition(expl, thres_hold, axis=None)[:thres_hold]   
        # 将要删除的像素的值设为0，相当于遮挡掉非显著区域
        X_copy[delete_index] = 0
        # 将修改后的一维向量恢复为原始图像的形状
        X_copy2 = X_copy.reshape(1,3,H,W)
        # 使用模型对修改后的图像进行预测，并取出目标类别的概率值
        logit = F.softmax(model(X_copy2), dim=1).T
        logit = logit[cls].item()
        # 计算修改后的图像的预测值和原始图像的预测值之间的差异，并除以原始图像的预测值，得到一个相对误差
        pred_y.append(abs(pred_probablity - logit)/pred_probablity)
        # 更新计数器的值
        count += 0.01
    # 返回预测值变化的列表
    return pred_y

model = build_model(arch='resnet50')
gradcam = GradCAM(model, layer_name='layer4')

path = './ImageNet2012_val'
file_nums = len(os.listdir(path))

delete_drop_sum = [0 for i in range(1, 100)]
cnt = 0
t1 = time.time()
for file_path in [os.path.join(path, item) for item in os.listdir(path)]:
    # if random.uniform(0, 1) < 0.1: # 抽样10%
    t3 = time.time()
    delete_drop = compuate_sal(model, img_path = file_path, cls=-1)
    for i, value in enumerate(delete_drop):
        delete_drop_sum[i] += float(value)
    cnt += 1
    print('{}/{} done, cost time:{}'.format(cnt, file_nums, time.time()-t3))
    # else:
    #     continue
t2 = time.time()
delete_drop_sum = [num / file_nums for num in delete_drop_sum]
print('cost time:', t2-t1)
print('delete_drop_sum:', delete_drop_sum)
