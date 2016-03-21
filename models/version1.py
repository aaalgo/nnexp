from theano import tensor as T
import lasagne as nn
from lasagne.layers import batch_norm as bn

def sorenson_dice(pred, tgt, ss=10):
    return -2*(T.sum(pred*tgt)+ss)/(T.sum(pred) + T.sum(tgt) + ss) 

def network(input_var, label_var, shape):
    layer = nn.layers.InputLayer(shape, input_var)
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=8, filter_size=5))
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=16, filter_size=3))
    layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=4))
    layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=4))
    layer = nn.layers.MaxPool2DLayer(layer, pool_size=2)
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=5))
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=64, filter_size=5, pad='full'))
    layer = nn.layers.Upscale2DLayer(layer, scale_factor=2)
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=32, filter_size=4, pad='full'))
    layer = nn.layers.Upscale2DLayer(layer, scale_factor=2)
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=16, filter_size=4, pad='full'))
    layer = nn.layers.Upscale2DLayer(layer, scale_factor=2)
    layer = bn(nn.layers.Conv2DLayer(layer, num_filters=8, filter_size=3, pad='full'))
    layer = nn.layers.Conv2DLayer(layer, num_filters=1, filter_size=5, pad='full',
                nonlinearity=nn.nonlinearities.sigmoid)

    output = nn.layers.get_output(layer)
    output_det = nn.layers.get_output(layer, deterministic=True)

    loss = sorenson_dice(output, label_var) #, ss=ss)
    te_loss = sorenson_dice(output_det, label_var) #,ss=ss)
    te_acc = nn.objectives.binary_accuracy(output_det, label_var).mean()

    return layer, loss, [te_loss, te_acc]
