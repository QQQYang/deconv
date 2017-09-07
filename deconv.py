# function: classify new image using trained model, and visualize the feature map through convolution and deconvolution
# author:   Qian Yang
# date:     2017/9/7 
import numpy as np
import sys
import caffe
from skimage import io
import matplotlib.pyplot as plt

import FindMaxes as fm


#---------------------parameters--------------------------
caffe.set_mode_gpu()
model_path = '/home/qy/documents/caffe/examples/compact_bilinear/deconv/ft_last_layer_deconv_deploy.prototxt'
invmodel_path = '/home/qy/documents/caffe/examples/compact_bilinear/deconv/invdeploy_norelu.prototxt'
weight_path = '/home/qy/documents/caffe/examples/compact_bilinear/snapshot_iter_60000_last_layer.caffemodel'
image_name = '/home/qy/documents/PythonPrj/CaffePy/val.txt'
image_dir = '/home/qy/documents/PythonPrj/CaffePy/val_resize/'

plane_name = ['Airbus A300', 'Airbus A320', 'Hydroglider', 'Gyroplane', 'Old-fashioned propeller', \
            'Airship', 'Helicopter', 'Jet aircraft', 'UAV', 'Turboprop']
# calculation of receptive field: 
# bilinear cnn: conv1_1,conv1_2,pool1,conv2_1,conv2_2,pool2,conv3_1,conv3_2,conv3_3,pool3,conv4_1,conv4_2,conv4_3,pool4,conv5_1,conv5_2,conv5_3
# all convolutional layers have kernel size of 3*3, with appropriate padding to keep the size of feature maps unchanged
# all pooling layers have have kernel size of 2*2 and stride equals 2.
# therefore, the size of receptive field for one point on the feature map in layer conv5_3 is ((((1+2*3)*2+2*3)*2+2*3)*2+2*2)*2+2*2=196
receptive_field = 196
#----------------------------------------------------

# visualize the feature map in form of grid
def vis_square(data):
    # normalization
    data = (data - data.min()) / (data.max() - data.min())

    # calculate the number of grid in each side 
    n = int(np.ceil(np.sqrt(data.shape[0])))    
    padding = (((0, n**2 - data.shape[0]), (0, 1), (0, 1))+((0, 0), )*(data.ndim - 3))
    data = np.pad(data, padding, mode='constant', constant_values=1)

    data = data.reshape((n,n) + data.shape[1:]).transpose((0, 2, 1, 3)+tuple(range(4, data.ndim + 1)))
    data = data.reshape((n*data.shape[1], n*data.shape[3]) + data.shape[4:])

    return data

def process(net, img_path):

    # parameters of image preprocessing
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    mu = np.load('/home/qy/documents/caffe/data/plane/plane_mean2.npy')
    mu = mu.mean(1).mean(1)
    transformer.set_mean('data', mu)

    # reshape
    net.blobs['data'].reshape(1, 3, 448, 448)

    # load image
    img = caffe.io.load_image(img_path)
    # crop
    h_off = (img.shape[0] - 448)/2
    w_off = (img.shape[1] - 448)/2
    img = img[h_off:h_off+448, w_off:w_off+448,:]

    # begin to preprocess
    net.blobs['data'].data[...] = transformer.preprocess('data', img)

    # run
    out = net.forward()

    # visualize
    n_top = 9
    max_activation_point = fm.MaxActivationPoint(n_top)
    data = net.blobs['conv5_3'].data
    max_activation_point.update(data)
    feat = data[0,max_activation_point.max_feature_map.max_vals_ind] # 
    data = vis_square(feat)

    # image matting
    loc = max_activation_point.max_loc
    height = img.shape[0]
    width = img.shape[1]
    plt.figure(1)
    grid_num = int(np.ceil(np.sqrt(n_top))) 
    for i in range(n_top):
        tlx = max(loc[i,0]*16-receptive_field/2, 0)
        brx = min(loc[i,0]*16+receptive_field/2,height)
        tly = max(loc[i,1]*16-receptive_field/2, 0)
        bry = min(loc[i,1]*16+receptive_field/2,width)
        p = plt.subplot(grid_num,grid_num,i)
        p.imshow(img[tlx:brx, tly:bry, :])
        p.axis('off')

    plt.figure(2)
    plt.imshow(data)
    plt.axis('off')

    plt.figure(3)
    plt.imshow(img)
    plt.axis('off')

    return out['prob'][0],transformer

def norm(x, s=1.0):
    x -= x.min()
    x /= x.max()
    return x*s



#------------initialize-------------------
net = caffe.Net(model_path, weight_path, caffe.TEST)    # load original network
invnet = caffe.Net(invmodel_path, caffe.TEST)           # load deconvolutional network

image_name_list = np.loadtxt(image_name, str, delimiter=' ')    # load image list
correctNum = 0 # record the number of samples which are classified into correct categories
plt.ion() # interactive mode
index = np.arange(image_name_list.shape[0])
prob = np.zeros((10,10))    # record the output probability
#np.random.shuffle(index)   # shuffle
#-----------------------------------------

#--------------set parameters of deconv network----------------
for b in invnet.params:
    if b!='conv5_3':    # assign the parameters to the deconvolutional network except the input layer
        invnet.params[b][0].data[...] = net.params[b][0].data.reshape(invnet.params[b][0].data.shape)
#------------------------------------------------------

for i in index:
    img = image_dir + image_name_list[i][0]
    label = image_name_list[i][1]
    out_prob, transformer = process(net, img)
    pred = out_prob.argmax()

    #------------------feed input for deconv network----------------
    data = net.blobs['conv5_3'].data

    # just adopt one feature map with the strongest activation for deconvolution
    max_feature_map = fm.MaxFeatureMap(9)
    max_feature_map.update(data)
    feat = data[:,max_feature_map.max_vals_ind[0]]
    # assign the parameter to the input layer
    invnet.params['conv5_3'][0].data[...] = net.params['conv5_3'][0].data[:,max_feature_map.max_vals_ind[0]].reshape(invnet.params['conv5_3'][0].data.shape)

    # just filter the location with weaker activation
    #feat[0][feat[0] < 100] = 0
    invnet.blobs['conv5_3'].data[...] = feat
    invnet.blobs['switches4'].data[...] = net.blobs['switches4'].data
    invnet.blobs['switches3'].data[...] = net.blobs['switches3'].data
    invnet.blobs['switches2'].data[...] = net.blobs['switches2'].data
    invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
    invnet.forward()

    feat = norm(invnet.blobs['img'].data[0], 255.0)

    plt.figure(4)
    plt.imshow(transformer.deprocess('data', feat))

    plt.figure(5)
    plt.title('label: '+label)
    top_k = 5
    out_prob_ind = out_prob.argsort()[::-1]
    out_prob = np.sort(out_prob)[::-1]
    for k in range(top_k):
        plt.text(0.4,0.8-0.1*k,str(out_prob_ind[k])+': '+str(out_prob[k]))
    
    plt.show()
    plt.waitforbuttonpress()
    plt.clf()

print('test accuracy: %f' % (float(correctNum)/image_name_list.shape[0]))