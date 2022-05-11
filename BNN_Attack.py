import tensorflow as tf
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from setup_cifar import CIFAR, CIFARModel
import pandas as pd
import cv2
import BNN_mr1
from reactnet import reactnet
from torch.autograd import Variable
import torch.nn as nn
import resnet_bireal
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.environ['CUDA_VISIBLE_DEVICES']='0'
#log_device_placement=True 输出运行每一个运算的设备

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
def man_show(imge_std):
    # adv = (imge_std * std) + mean
    adv = imge_std * 255.0
    adv = np.clip(adv, 0, 255).astype(np.uint8)
    return adv

rgb_chanel = 0
CLASSES =  10
def box_cai1(img,k):
    imgray=img[:,:,rgb_chanel]#cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)#
    kernel = np.ones((5, 5), np.float32) / 25
    img_fil = cv2.filter2D(imgray, -1, kernel)
    # threshold 函数对图像进行二化值处理，由于处理后图像对原图像有所变化，因此img.copy()生成新的图像，cv2.THRESH_BINARY是二化值
    #ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 128, 255,cv2.THRESH_BINARY)  # 灰度值小于thresh的点置255，灰度值大于thresh的点置0
    ret, thresh = cv2.threshold(img_fil.copy(), 0.1,255, cv2.THRESH_BINARY)
    thresh = np.array(thresh,np.uint8)

    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_kuang = img.copy()
    max_ju=0
    liu=0
    for max_ite in range(len(contours)) :
        x_bei, y_bei, w_bei, h_bei = cv2.boundingRect(contours[max_ite])
        max_jubei= w_bei*h_bei
        if max_jubei>max_ju:
            max_ju=max_jubei
            liu=max_ite
    x, y, w, h = cv2.boundingRect(contours[liu])
    cv2.rectangle(img_kuang, (x, y), (x + w, y + h), (0, 255, 0), 1)

    if k % 100 == 0 :
        fig1 = plt.figure(1)
        plt.subplot(1,3,1)
        plt.imshow(img_fil,cmap='gray')
        plt.title('gray image')
        plt.subplot(1, 3, 2)
        plt.imshow(thresh,cmap='gray')
        plt.title('binary image')
        plt.subplot(1, 3, 3)
        plt.imshow(img_kuang,cmap='gray')
        plt.title('contours')
        plt.savefig('result/第%d张轮廓选取1.pdf'%k,bbox_inches="tight", pad_inches=0.0)
        # plt.draw()
        # plt.pause(2)  # 间隔的秒数： 4s
        # plt.show()
        plt.close(fig1)
    return x, y, w, h
def box_cai2(img,k):
    # imgray = img.copy()#cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((5, 5), np.float32)/25
    # img_fil = cv2.filter2D(imgray, -1, kernel)
    imgray=img[:,:,rgb_chanel]#cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)##img[:,:,0]#cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.float32) /25
    img_fil = cv2.filter2D(imgray, -1, kernel)
    # threshold 函数对图像进行二化值处理，由于处理后图像对原图像有所变化，因此img.copy()生成新的图像，cv2.THRESH_BINARY是二化值
    #ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 128, 255,cv2.THRESH_BINARY)  # 灰度值小于thresh的点置255，灰度值大于thresh的点置0
    ret, thresh = cv2.threshold(img_fil.copy(), 0.1,255, cv2.THRESH_BINARY_INV)
    thresh = np.array(thresh,np.uint8)

    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_kuang = img.copy()
    max_ju=0
    liu=0
    for max_ite in range(len(contours)) :
        x_bei, y_bei, w_bei, h_bei = cv2.boundingRect(contours[max_ite])
        max_jubei= w_bei*h_bei
        if max_jubei>max_ju:
            max_ju=max_jubei
            liu=max_ite
    x, y, w, h = cv2.boundingRect(contours[liu])
    cv2.rectangle(img_kuang, (x, y), (x + w, y + h), (0, 255, 0), 1)
    if k % 100== 0 :
        fig1 = plt.figure(1)
        plt.subplot(1,3,1)
        plt.imshow(img_fil,cmap='gray')
        plt.title('gray image')
        plt.subplot(1, 3, 2)
        plt.imshow(thresh, cmap='gray')
        plt.title('binary image')
        plt.subplot(1, 3, 3)
        plt.imshow(img_kuang, cmap='gray')
        plt.title('contours')
        plt.savefig('result/第%d张轮廓选取2.pdf'%k,bbox_inches="tight", pad_inches=0.0)
        # plt.draw()
        # plt.pause(2)  # 间隔的秒数： 4s
        # plt.show()
        plt.close(fig1)
    return x, y, w, h

def universal_perturbation2(model,theta_train, p_train, n, dataset, max_iter_df=1000):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_images = np.shape(dataset)[0]  # 图像应沿第一维堆叠
    print("X size:{}".format(num_images))
    data = CIFAR()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    state_dict = torch.load("./BNNmodel_best.pth.tar")
    model = resnet_bireal.birealnet20()
 
    model.load_state_dict(state_dict, strict=False)
   
    model.to(device)
    # 浏览数据集并顺序计算扰动增量
    num_query = []  #记录每个图像的查询次数
    l2_val=[]#记录所有的二范数结果
    linf_val=[]
    succ_num=0 #记录成功个数
    labels=[]
    results=[]
    ns=[]
    qian_fail=0
    for k in range(0, 1000):  # num_images
        theta = theta_train
        p = p_train  # 搜索方向初始值
        dorta = 5.5  # 变异强度初始值
        cur_img = dataset[k:(k + 1), :]
        plt.imshow(cur_img.reshape(32, 32, 3))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        img = cur_img.transpose(0, 3, 1, 2)   #transpose 直接调换矩阵位置,(1,32,32,3)--->(1,3,32,32):0不变,3换到1的位置,1和2向后移动(0,1,2,3)--->(0,3,1,2)
        
        img = Variable(torch.from_numpy(img).to(device).float())


        predictions = model(img).data.cpu().numpy() #预测值
        #预测值
        #predictions = model(img).data.cuda()
        label_ini = np.argmax(predictions)   #真实标签值
        box = (0, 0, 32, 32)
        n = box[2] * box[3]*3
        # box=[0,0,model.image_size,model.image_size]
        theta, iter, dorta, p, pert_v, result, pert_image, num = BNN_mr1.r1_binary(model, box, [theta], dorta, p, n, img,label_ini,
                                                                                                 max_iter=max_iter_df)



        # img_test = pert_v.reshape(3, 32, 32)
        # img_test = pert_v.reshape(3,32,32)  这个地方在原来的代码中是将扰动:pert_v的图片进行判别,这样成功率肯定变高,而且后面用的标签标题是用这里错误生成的label:k_i
        img_test = pert_image.reshape(3,32,32)
        # img_test = pert_vl.transpose(2, 0, 1)
        img_test = np.expand_dims(img_test, axis=0)  # [1,3,224,224]

        
        img_test = Variable(torch.from_numpy(img_test).to(device).float())
        predictions = model(img_test).data.cpu().numpy()
        k_i = np.argmax(predictions)
        print(k_i, label_ini)
        if k_i != label_ini:
            succ_num += 1

        num_query.append(num)
        labels.append(label_ini)
        results.append(result)
        l2_val.append(np.linalg.norm(pert_v))
        linf_val.append(np.abs(pert_v).max())
        ns.append(n)
        # if label_ini != result:
        #     succ_num+=1
        if k % 100 == 0:
            print("--------第%d个图像----------" % k)
            print('num_query', num_query)
            print('l2_val', l2_val)
            print('linf_val', linf_val)
            print('ns', ns)
            print('succ_num', succ_num)
            print('query_mean %6f, l2_val %6f,linf_val %6f, ns %6f' % (
            np.mean(num_query), np.mean(l2_val), np.mean(linf_val), np.mean(ns)))

            result_sum = pd.DataFrame(
                {'query': num_query, 'l2_val': l2_val, 'linf_val': linf_val, 'label': labels, 'result': results,
                 'ns': ns})
            pd.DataFrame(result_sum).to_csv('result/%d_nattack.csv'%k)

        

        
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
        plt.savefig("adv/%d.jpeg" % k)
        if k % 100 == 0:
            print("--------第%d个图像----------" % k)
            print("ini %d,%d iter,change %d,\nl2 %6f,l_inf %6f，query %d" % (label_ini,
                                                                            iter, result,
                                                                            torch.norm(pert_v),
                                                                            torch.norm(pert_v, float('inf')), num))

            fig2 = plt.figure(2)
            plt.subplot(1, 3, 1)
            plt.imshow(cur_img.reshape(32, 32, 3))
            plt.title('%s'%classes[label_ini])
            plt.subplot(1, 3, 2)
            plt.imshow(pert_image.reshape(32, 32, 3))
            # plt.imshow(pert_image.reshape(32, 32, 3))
            plt.title('%s'%classes[k_i])
            plt.subplot(1, 3, 3)
            plt.imshow(10*pert_v.reshape(32,32,3))
            plt.title("pert*10")
            # plt.suptitle("原始类别为：%s,经过%d代的迭代后，label变为：%s，\n二范数结果为：%6f，无穷范数结果为：%6f，共查询%d次。" % (classes[label],
            #                                                                         iter, classes[result],
            #                                                                         torch.norm(pert_v),
            #                                                                         torch.norm(pert_v,float('inf')), num))
            # plt.savefig("result/第%d张结果smr1.pdf"%k,bbox_inches='tight',pad_inches=0.0)
            plt.show()
            plt.close(fig2)

    print('重跑的个数',qian_fail)
    return theta, dorta, p, cur_img, pert_image, results, num_query,l2_val,linf_val,succ_num,labels,ns


# In[214]:

if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
    #pic = np.load('universal.npy')[:, :32, :32, :]
    pic = np.load('../universal.npy')
    #img = cv2.imread('pert.png', cv2.IMREAD_GRAYSCALE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    data =  CIFAR()
    state_dict = torch.load("./BNNmodel_best.pth.tar")
    model = resnet_bireal.birealnet20()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    test_x = data.test_data
    test_y = data.test_labels
        #X = (test_x.copy()+0.5)*255
    X = test_x.copy()
    #n = model.image_size * model.image_size * model.num_channels
    n = 32*32*3
        # theta=torch.Tensor(pic.flatten())
        # theta=torch.clamp(theta,-0.1,0.1)
        # print("tensor theta",theta)
        #theta=torch.Tensor(np.zeros(n))
        #print("theta",theta)
    theta = torch.clamp(torch.Tensor(pic.flatten()), -0.06, 0.06)
    p=torch.zeros(n)
   

    theta, dorta, p, img, pert_image, results, num_query,l2_val,linf_val,succ_num,labels,ns =\
        universal_perturbation2(model,theta, p, n, X )
    print('num_query', num_query)
    print('l2_val', l2_val)
    print('linf_val', linf_val)
    print('ns', ns)

    print('succ_num', succ_num)
    print('query_mean %6f, l2_val %6f,linf_val %6f, ns %6f' % (
    np.mean(num_query), np.mean(l2_val), np.mean(linf_val), np.mean(ns)))

    result_sum = pd.DataFrame(
        {'query': num_query, 'l2_val': l2_val, 'linf_val': linf_val, 'label': labels, 'result': results, 'ns': ns})
    pd.DataFrame(result_sum).to_csv('result/BNNattack.csv')
