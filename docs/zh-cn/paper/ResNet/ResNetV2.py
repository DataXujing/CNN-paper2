# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
# ResNet V2
# 载入模块、TensorFlow
import collections
import tensorflow as tf
 
slim = tf.contrib.slim
# 注意tf.contrib类,很多和卷积相关的模块都在slim中，例如AlexNet,VGG
 
# 定义Block,结构体,自动以的一种数据结构
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'A named tuple describing a ResNet block'

# 说明： 见bottleneck这个函数
# scope:"block1","block2".....
# unit_fn: bottleneck 残差学习单元
# args: 参数 见resnet_v2_50

 
# 定义降采样subsample方法
def subsample(inputs, factor, scope=None):
    '''
    factor: 1不做修改直接返回input,其他时使用最大池化1X1的尺寸，步长是factor
    scope: 变量的命名空间
    '''
 
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)
 
# 定义conv2d_same函数创建卷积层
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    '''
    如果factor=1直接“SAME"填充
    如果factor!=1,input做填充
    '''
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        # kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size - 1  #pad的总数
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        # input,张量维度：在8个维度上的填充，填充模式，填充值默认为0
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)
 
 
@slim.add_arg_scope
# 装饰目标函数，arg_scope为目标函数设置默认参数
# 定义堆叠Blocks函数
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):
 
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            #name_or_scope,default_name=None,values=None
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net, # 残差学习的生成函数 bittleneck函数
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
            # 将输出net添加到collection
 
    return net
 
# 创建ResNet通用arg_scope，定义函数默认参数值
# arg_sope用来定义默写函数参数的默认值
def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
 
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
 
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
 
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
 
 
@slim.add_arg_scope
# 定义核心bottleneck残差学习单元
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
 
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # 获取输入的最后一个维度，即channel数，min_rank限定最少为4个维度
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        # batch normalization,并进行Relu预激活
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')  # 降采样 使得输入通道数epth_in与depth一致
        else: #用1X1的卷积改变通道数
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
 
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')
 
        output = shortcut + residual  # identity的
        
        # output加入到集合并返回output
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)
 
# 定义生成ResNet V2的主函数
def resnet_v2(inputs, 
              blocks,  # 定义好的类的列表
              num_classes=None, #最后输出的类数
              global_pool=True, # 是否加入最后一层的全局平均池化
              include_root_block=True,  # 是否加入最前面通常使用的7X7的卷积池化
              reuse=None,  #是否重用
              scope=None):  #整个网络的名称（命名空间）
 
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
 
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)
 
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True) # avg_pool
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            # Convert end_points_collection into a dictionary of end_points.
            # 将collection转化为dict
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            # 返回net和end_points
            return net, end_points
 

#----------------------------------------------------------
# 设计层数为50的ResNet V2
def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        # Depth:输出通道数,depth_bottleneck,stride
        # (256,64,1): 256第三层输出的通道数，64:前两层输出的通道数，1:中间层的步长
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)
 
# 设计101层的ResNet V2
def resnet_v2_101(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_101'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)
 
# 设计152层的ResNet V2
def resnet_v2_152(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)
 
# 设计200层的ResNet V2
def resnet_v2_200(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_200'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)
 
 
from datetime import datetime
import math
import time
 
# 评测函数
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))
 
 
batch_size = 32
height, width = 224, 224
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    net, end_points = resnet_v2_152(inputs, 1000) # 152层评测
 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, net, "Forward")
