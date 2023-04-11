import os
import time
from find_layer_pt import find_densenet_layer, find_resnet_layer, find_vgg_layer
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import transforms

H, W = 224, 224

transform = transforms.Compose(
        [
                #transforms.Resize((H, W)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def load_image(path, preprocess=False):

    """Load and preprocess image."""

    img = Image.open(path).convert('RGB').resize((H, W))
    
    if preprocess:
        
        img = transform(img)
        
        img = img.unsqueeze(0)

    return img

def build_model(arch):
    model = models.__dict__[arch](pretrained=True)
    model = model.eval()
    model = model.cuda()
    return model
        
class GradCAM(object):
    def __init__(self, model, layer_name):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
        
            self.gradients['value'] = grad_output[0]
            
            return None
        
        def forward_hook(module, input, output):
        
            self.activations['value'] = output
        
            return None
        if self.model.__class__.__name__ == 'ResNet':
            target_layer = find_resnet_layer(self.model, layer_name)
        elif self.model.__class__.__name__ == 'VGG':
            target_layer = find_vgg_layer(self.model, layer_name)
        elif self.model.__class__.__name__ == 'DenseNet':
            target_layer = find_densenet_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, img, cls, retain_graph=False):
        logit = self.model(img)
        if cls == -1:
            cls = logit.max(1)[-1]
            score = logit[:, cls].squeeze()
        else:
            score = logit[:, cls].squeeze()
        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()
        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)
        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.upsample(saliency_map, size=(H, W), mode='bilinear', align_corners=False)
        saliency_map = F.relu(saliency_map)
        saliency_map = saliency_map / saliency_map.max() 
        #saliency_map = cv2.resize(saliency_map.squeeze().detach().cpu().numpy(), (H,W), cv2.INTER_LINEAR)
        return saliency_map, logit, cls
    def __call__(self, img, cls=-1, retain_graph=False):

        return self.forward(img, cls, retain_graph)

def compuate_sal(model, img_path, cls):
    img = load_image(img_path, preprocess=True).cuda()
    
    cam, logit, cls = gradcam(img, cls)
    pred_probablity = F.softmax(logit, dim=1)[0][cls].detach().cpu().numpy()
    expl = cam.repeat(1,3,1,1).squeeze().detach().cpu().numpy()
    
    pred_y = []
    X_copy = img.clone().flatten()
    count = 0.01
    while(count <= 1.0):
        thres_hold = int(H * W * 3 * count)
        delete_index = np.argpartition(expl, thres_hold, axis=None)[:thres_hold]   
        X_copy[delete_index] = 0
        X_copy2 = X_copy.reshape(1,3,H,W)
        logit = F.softmax(model(X_copy2), dim=1).T
        logit = logit[cls].item()
        pred_y.append(abs(pred_probablity - logit)/pred_probablity)
        count += 0.01
    return pred_y
    
if __name__ == '__main__':

    model = build_model(arch='vgg19')
    gradcam = GradCAM(model, layer_name='features_34')
    
    path = './ILSVRC2012_img_val'
    file_nums = len(os.listdir(path))
    
    delete_drop_sum = [0 for i in range(1, 100)]
    cnt = 0
    t1 = time.time()
    for file_path in [os.path.join(path, item) for item in os.listdir(path)]:  
        t3 = time.time()
        delete_drop = compuate_sal(model, img_path = file_path, cls=-1)
        for i, value in enumerate(delete_drop):
            delete_drop_sum[i] += float(value)
        cnt += 1
        print('{}/{} done, cost time:{}'.format(cnt, file_nums, time.time()-t3))
    t2 = time.time()
    delete_drop_sum = [num / file_nums for num in delete_drop_sum]
    print('cost time:', t2-t1)
    print('delete_drop_sum:', delete_drop_sum)
