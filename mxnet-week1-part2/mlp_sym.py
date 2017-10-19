import mxnet as mx


def mlp_layer(input_layer, n_hidden, activation=None, BN=False):

    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param n_hidden: # of hidden neurons
    :param activation: the activation function
    :return: the symbol as output
    """

    # get a FC layer
    l = mx.sym.FullyConnected(data=input_layer, num_hidden=n_hidden)
    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    return l


def get_mlp_sym():

    """
    :return: the mlp symbol
    """

    data = mx.sym.Variable("data")
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data_f = mx.sym.flatten(data=data)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp


def conv_layer(data_in, filter_size, pooling_size, is_conv,is_pooling):
    """
    :return: a single convolution layer symbol
    """
    #input layer
    layer = data_in
    if is_conv:
        input_layer = mx.sym.Convolution(data = layer, 
                                kernel = (filter_size, filter_size), 
                                num_filter = 64,
                                pad = (1, 1),
                                stride = (1, 1)
                                )
    
        #batchnorm
        batch_layer = mx.sym.BatchNorm(input_layer)
    
        #Activation layer
        layer = mx.sym.Activation(batch_layer,
                              act_type='relu',
                              )
    #pooling layer
    if is_pooling: 
        layer = mx.sym.Pooling(layer, 
                               kernel=(pooling_size,pooling_size),
                               pool_type='max'
                                   )
    return layer


# Optional
def inception_layer(inputdata):
    
    l1 = conv_layer(inputdata, 1, 1, True, False)
    l21 = conv_layer(inputdata, 1, 1, True, False)
    l2 = conv_layer(l21, 3, 2, True, False)
    l31 = conv_layer(inputdata, 1, 1, True, False)
    l3 = conv_layer(l31, 5, 2, True, False)
    l41 = conv_layer(inputdata, 1, 3, False, True)
    l4 = conv_layer(l41, 1, 1, True, False)
    l = mx.sym.Concat(l1,l2,l3,l4)
    return l


def get_conv_sym(n_layer):

    """
    :return: symbol of a convolutional neural network
    """
    data = mx.sym.Variable("data")
    data_f = mx.sym.flatten(data=data)
    
    layer = conv_layer(data_f,3,2,True,True)
    for i in range(n_layer - 1):
        layer = conv_layer(layer,3,2,True,True)
    layer = mx.sym.Flatten(layer)
    l = mx.sym.FullyConnected(layer,num_hidden = 10)
                              
    # Softmax 
    myl = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return myl
