from theano import tensor as T
import lasagne as nn
from lasagne.layers import batch_norm as bn

def sorenson_dice(pred, tgt, ss=10):
    return -2*(T.sum(pred*tgt)+ss)/(T.sum(pred) + T.sum(tgt) + ss) 

def network(input_var, label_var, shape):
    layer = nn.layers.InputLayer(shape, input_var)                          # 256
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=8, filter_size=5))  # 252
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=16, filter_size=3)) # 250
    layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)                    # 125
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=4)) # 122
    layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)                    # 61
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=4)) # 58
    layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)                    # 29
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=5)) # 25 
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=5, pad='full')) # 25 + 8 - 4 =  29
    layer = nn.layers.Upscale2DLayer(layer, scale_factor=2)                 # 58
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=4, pad='full')) # 62
    layer = nn.layers.Upscale2DLayer(layer, scale_factor=2)                 # 124
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=16, filter_size=4, pad='full')) # 128
    layer = nn.layers.Upscale2DLayer(layer, scale_factor=2)                 # 256
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=8, filter_size=3, pad='full'))
    layer = nn.layers.Conv2DLayer(layer, num_filters=1, filter_size=5, pad='full',
                nonlinearity=nn.nonlinearities.sigmoid)                     # 256

    #for l in nn.layers.get_all_layers(layer):
    #    print nn.layers.get_output_shape(l)

    output = nn.layers.get_output(layer)
    output_det = nn.layers.get_output(layer, deterministic=True)

    loss = sorenson_dice(output, label_var) #, ss=ss)
    te_loss = sorenson_dice(output_det, label_var) #,ss=ss)
    te_acc = nn.objectives.binary_accuracy(output_det, label_var).mean()

    return layer, loss, [te_loss, te_acc]
