# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:35:11 2019

@author: hp
"""

def find_resnet_layer(arch, target_layer_name):

    """Find resnet layer to calculate GradCAM and GradCAM++

    

    Args:

        arch: default torchvision densenet models

        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.

            target_layer_name = 'conv1'

            target_layer_name = 'layer1'

            target_layer_name = 'layer1_basicblock0'

            target_layer_name = 'layer1_basicblock0_relu'

            target_layer_name = 'layer1_bottleneck0'

            target_layer_name = 'layer1_bottleneck0_conv1'

            target_layer_name = 'layer1_bottleneck0_downsample'

            target_layer_name = 'layer1_bottleneck0_downsample_0'

            target_layer_name = 'avgpool'

            target_layer_name = 'fc'

            

    Return:

        target_layer: found layer. this layer will be hooked to get forward/backward pass information.

    """

    if 'layer' in target_layer_name:

        hierarchy = target_layer_name.split('_')

        layer_num = int(hierarchy[0].lstrip('layer'))#把layer几个字母截去留下数字

        if layer_num == 1:

            target_layer = arch.layer1

        elif layer_num == 2:

            target_layer = arch.layer2

        elif layer_num == 3:

            target_layer = arch.layer3

        elif layer_num == 4:

            target_layer = arch.layer4

        else:

            raise ValueError('unknown layer : {}'.format(target_layer_name))



        if len(hierarchy) >= 2:

            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))

            target_layer = target_layer[bottleneck_num]



        if len(hierarchy) >= 3:

            target_layer = target_layer._modules[hierarchy[2]]

                

        if len(hierarchy) == 4:

            target_layer = target_layer._modules[hierarchy[3]]



    else:

        target_layer = arch._modules[target_layer_name]



    return target_layer





def find_densenet_layer(arch, target_layer_name):

    """Find densenet layer to calculate GradCAM and GradCAM++

    

    Args:

        arch: default torchvision densenet models

        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.

            target_layer_name = 'features'

            target_layer_name = 'features_transition1'

            target_layer_name = 'features_transition1_norm'

            target_layer_name = 'features_denseblock2_denselayer12'

            target_layer_name = 'features_denseblock2_denselayer12_norm1'

            target_layer_name = 'features_denseblock2_denselayer12_norm1'

            target_layer_name = 'classifier'

            

    Return:

        target_layer: found layer. this layer will be hooked to get forward/backward pass information.

    """

    

    hierarchy = target_layer_name.split('_')

    target_layer = arch._modules[hierarchy[0]]



    if len(hierarchy) >= 2:

        target_layer = target_layer._modules[hierarchy[1]]



    if len(hierarchy) >= 3:

        target_layer = target_layer._modules[hierarchy[2]]



    if len(hierarchy) == 4:

        target_layer = target_layer._modules[hierarchy[3]]



    return target_layer





def find_vgg_layer(arch, target_layer_name):

    """Find vgg layer to calculate GradCAM and GradCAM++

    

    Args:

        arch: default torchvision densenet models

        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.

            target_layer_name = 'features'

            target_layer_name = 'features_42'

            target_layer_name = 'classifier'

            target_layer_name = 'classifier_0'

            

    Return:

        target_layer: found layer. this layer will be hooked to get forward/backward pass information.

    """

    hierarchy = target_layer_name.split('_')



    if len(hierarchy) >= 1:
        if hierarchy[0] == 'features':
            target_layer = arch.features
        elif hierarchy[0] == 'classifier':
            target_layer = arch.classifier

    if len(hierarchy) == 2:

        target_layer = target_layer[int(hierarchy[1])]



    return target_layer





def find_alexnet_layer(arch, target_layer_name):

    """Find alexnet layer to calculate GradCAM and GradCAM++

    

    Args:

        arch: default torchvision densenet models

        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.

            target_layer_name = 'features'

            target_layer_name = 'features_0'

            target_layer_name = 'classifier'

            target_layer_name = 'classifier_0'

            

    Return:

        target_layer: found layer. this layer will be hooked to get forward/backward pass information.

    """

    hierarchy = target_layer_name.split('_')



    if len(hierarchy) >= 1:

        target_layer = arch.features



    if len(hierarchy) == 2:

        target_layer = target_layer[int(hierarchy[1])]



    return target_layer





def find_squeezenet_layer(arch, target_layer_name):

    """Find squeezenet layer to calculate GradCAM and GradCAM++

    

    Args:

        arch: default torchvision densenet models

        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.

            target_layer_name = 'features_12'

            target_layer_name = 'features_12_expand3x3'

            target_layer_name = 'features_12_expand3x3_activation'

            

    Return:

        target_layer: found layer. this layer will be hooked to get forward/backward pass information.

    """

    hierarchy = target_layer_name.split('_')

    target_layer = arch._modules[hierarchy[0]]



    if len(hierarchy) >= 2:

        target_layer = target_layer._modules[hierarchy[1]]



    if len(hierarchy) == 3:

        target_layer = target_layer._modules[hierarchy[2]]



    elif len(hierarchy) == 4:

        target_layer = target_layer._modules[hierarchy[2]+'_'+hierarchy[3]]



    return target_layer
