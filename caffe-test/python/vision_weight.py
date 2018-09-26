
#编写一个函数，用于显示各层的参数,padsize用于设置图片间隔空隙,padval用于调整亮度 
def show_weight(data, padsize=1, padval=0, name="conv.jpg"):
    #归一化
    data -= data.min()
    data /= data.max()
    print data.ndim
    #根据data中图片数量data.shape[0]，计算最后输出时每行每列图片数n
    n = int(np.ceil(np.sqrt(data.shape[0])))
    # padding = ((图片个数维度的padding),(图片高的padding), (图片宽的padding), ....)
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))    
    # 先将padding后的data分成n*n张图像
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))    
    # 再将（n, W, n, H）变换成(n*w, n*H)
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print data.shape
    plt.set_cmap('gray')
    plt.imshow(data)
    plt.imsave(name,data)
    plt.axis('off')
 
if __name__ == '__main__':
 
    deploy_file = CAFFE_ROOT + '/examples/cifar10/' + deploy_file_name
    model_file  = CAFFE_ROOT + '/examples/cifar10/' + model_file_name
    #初始化caffe		
    net = caffe.Net(deploy_file, model_file, caffe.TEST)
    print [(k, v[0].data.shape) for k, v in net.params.items()]
    
    #第一个卷积层，参数规模为(32,3,5,5)，即32个5*5的3通道filter
    weight = net.params["conv1"][0].data
    print weight.shape
    show_weight(weight.reshape(32*3,5,5), padsize=2, padval=0, name="conv1-cifar10.jpg")	
    #第二个卷积层，参数规模为(32,32,5,5)，即32个5*5的32通道filter
    weight = net.params["conv2"][0].data
    print weight.shape
    show_weight(weight.reshape(32*32,5,5), padsize=2, padval=0, name="conv2-cifar10.jpg")