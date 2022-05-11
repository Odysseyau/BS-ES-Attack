import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import resnet_attack as attack
import cv2
from torch.autograd import Variable
import torch.nn as nn
import resnet_bireal

CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

state_dict = torch.load("./BNNmodel_best.pth.tar")
model = resnet_bireal.birealnet20()
model.load_state_dict(state_dict, strict=False)
model.to(device)

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
def man_show(imge_std):
    # adv = (imge_std * std) + mean
    adv = imge_std * 255.0
    adv = np.clip(adv, 0, 255).astype(np.uint8)
    return adv
def normalize(x):
	return (x - np.min(x))/(np.max(x) - np.min(x))
#def f(resnet50, image):
def f(model, image):
    pre_dic= {}
    pre_gai = []
    #predictions = resnet50(image).data.cpu().numpy()
    predictions = model(image).data.cpu().numpy()
    predictions = np.squeeze(predictions)

    I = predictions.argsort()[-10:][::-1]
    for j in range(10):
        pre_gai.append(predictions[I[j]])
    pre_gai_nor = normalize(pre_gai)

    # print(I)
    for i in range(10):
        pre_dic[I[i]] = pre_gai_nor[i]
    # print('predictions',pre_dic)
    return pre_dic

#输入：一张图片，返回：前三个分类结果（数字）与对应概率
def label_c(session, newimg, target, label):
    score=[]
    predictions = f(session, newimg)
    #top 3
    I = list(predictions.keys())  #前3个预测类别为数字
    for node_id in I:
        #human_string = node_lookup.id_to_string(node_id)  #为英文类别
        score.append( predictions[node_id] ) #概率
    if target:
        try:
            max_p = predictions[target]
            min_p = predictions[label]
        except Exception as e:
            min_p = 0  # 防止原始类别不在新预测的前10个列表中
            max_p = 0
    else:
        max_p = -1
        min_p = -1
    # print('score',score)
    return I, score, max_p, min_p

def crossover(tensor_uni):
    CR=0.005
    len_tensor = len(tensor_uni)
    # len_cross = [224,448,672,896,1120,2240,4480,8960]
    len_cross = [32, 64, 16, 8,4,128]
    random_c = torch.randint(0, len(len_cross), (1,)).item()
    random_cc = torch.randint(0, len(len_cross), (1,)).item()
    random_ccc = torch.randint(0, len(len_cross), (1,)).item()
    random_i = torch.randint(0,len_tensor,(1,)).item()
    random_ii = torch.randint(0, len_tensor, (1,)).item()
    random_iii = torch.randint(0, len_tensor, (1,)).item()
    tensor_uni[random_i:random_i+len_cross[random_c]]=0
    tensor_uni[random_ii:random_ii + len_cross[random_cc]] = 0
    tensor_uni[random_iii:random_iii + len_cross[random_ccc]] = 0
    # for i in range(0,len_tensor):
    #     if torch.randint(0,1,(1,)).item() <=CR or (i == random_i):
    #         tensor_uni[i] = 0
    return tensor_uni


def Fun_binary(session, ite, x, label, img_t):  # 注意循环的判断条件
    #ima_t是计算得到的对抗样本，x是扰动，采样的结果
    y = []
    num = len(x)
    label_cul_mem = []
    seg = 0

    if ite != 0:
        # target_onehot = np.zeros((1, 1000))
        target_onehot = np.zeros((1, 10))
        with torch.no_grad():
            outputs = session(img_t).data.cpu().numpy()
        target_onehot[0][label] = 1.
        target_onehot = target_onehot.repeat(num, 0)
        real = (target_onehot * outputs).sum(1)  # 所有行的和,也就是每个图像原始类别的概率
        # .max(1)返回A每一行最大值组成的一维数组——种群中最大的其他类别的概率
        other = ((1. - target_onehot) * outputs - target_onehot * 10.).max(1)[0]
        # 剩余类别的最大概率
        y = np.clip(real - other, 0, 10)
    else:  # 计算初始点时，传入的x是一个值
        with torch.no_grad():
            outputs = session(img_t).data.cpu().numpy()[0]
        # print(outputs)
        score_ini = outputs[label]
        outputs[label] = 0
        # print("first score_ini", score_ini, "max_2", max(outputs))
        y.append(max(score_ini - max(outputs), 0))  # max
    y_in_sort = torch.argsort(torch.Tensor(y), -1, False)  # 排序list y从小到大的索引值
    return y_in_sort, y

def Fun(session, ite, x, p, label, img_t,target):  # 注意循环的判断条件
    #ima_t是计算得到的对抗样本，x是扰动，采样的结果
    y = []
    num = len(x)
    label_cul_mem = []
    seg = 0

    if ite != 0:
        for i in range(num):  # 计算采样后有lam个点需要计算函数值
            label_cul, labels_p ,target_p, label_p= label_c(session, img_t[i],target, label)  # 类别，概率
            label_cul_mem.append(label_cul[0])
            # print("label",label)
            # print("label_cul",label_cul)
            # print("label_p",label_p)
            # print("target_p", target_p)
            if label_cul[0] == label:
                h = max(labels_p[0]-labels_p[1], torch.tensor(-0.15))
            else:
                h = max(label_p - labels_p[0], torch.tensor(-0.15))
            #h = max(label_p - target_p, torch.tensor(-0.15))
            lamd = [0.01,1]
            if h == torch.tensor(-0.15):
                seg = 1
                f_res=-h-lamd[1]*torch.norm(x[i]) # 二范数,float('inf')
            else:
                f_res = 10*(-h - lamd[0] * torch.norm(x[i]))#,float('inf')

            y.append(f_res)  # max

    else:  # 计算初始点时，传入的x是一个值
        label_cul, labels_p, target_p, label_p = label_c(session, img_t, target, label)  # 类别，概率
        label_cul_mem.append(label_cul[0])
        # print("label", label)
        # print("label_cul", label_cul)
        # print("label_p", labels_p)
        if label_cul[0] != target:
            h = max(labels_p[0]-target_p, -0.031)  #这里的类别和概率是字典模式，所以最高的概率一定是在前面
        else:
            h = max(label_p - target_p, -0.031)
        f_res = -h - 0.01 * torch.norm(x)
        y.append(f_res)  # max

    y_in_sort = torch.argsort(torch.Tensor(y),-1,True)  # 排序list y从大到小的索引值
    # print('----------------------------------------',y)
    # print(y_in_sort)
    return y_in_sort, y, label_cul_mem[y_in_sort[0]],seg

# 根据参数，随机采样，返回一个低维的样本
def sample_xi(n, theta, dorta, ccov, p):  #后面的left+right/2=mid 传入第二个参数thehta,dorta=1
    zi = torch.normal(mean=torch.zeros(n), std=torch.ones(n)) # 每次随机采样一个点，每代产生lam个解
    ri = torch.rand(1) # 随机采1个变量
    update = dorta * torch.add(math.sqrt(1 - ccov) * zi, math.sqrt(ccov) * ri * p)
    xi = torch.add(theta,update)
    return xi

def r1_binary(session, box, theta, dorta, p, n, image_tensor, label_ini, max_iter=1000):  # overshoot 增益系数
    # n = 64*64*3
    image = image_tensor[0].cpu().numpy().transpose(1,2,0)
    node_lookup = attack.NodeLookup()
    theta[0] = theta[0][:n]
    p = p[:n]
    label_query, _, _, _ = label_c(session, image_tensor, False, False)
    label = label_query[0]
    target = label_query[1]
    num = 1
    input_shape = image.shape
    # print('input_shape',input_shape)
    # len_image = 28  #int(math.sqrt(input_shape[1]))  # 图片长度
    # print("image",input_shape)
    num_channels=3
    shape_original =(1,32,32,3)
    shape = (1, box[3], box[2], num_channels)  # 只关注轮廓以内的信息
    # v=np.zeros(shape, dtype=np.float32)
    v = theta[0].reshape(shape)  # 扰动的初始值,注意这里取的是前n个值，后需改进

    image_pic = image.copy().reshape(32, 32, 3)
    roi = torch.Tensor(image_pic[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :])
    # 用来显示截取部分的图像
    # plt.imshow(man_show(roi.numpy()).astype(np.uint8))
    # plt.title('part')
    # plt.show()

    pert_image = image_tensor
    mask = False

    # f_i = np.array(f(pert_image)).flatten()  #flatten返回一个折叠成一维的数组,只适用于array,与mat，不可list；[y for x in a for y in x]
    # k_i = int(np.argmax(f_i))

    t = 0
    ccov = 1 / (3 * math.sqrt(n) + 5)
    c = 2 / (n + 7)
    s = 0  # 累积秩率
    f0_sort, f0 = Fun_binary(session, 0, v, label, pert_image)
    num += 1

    lam = math.ceil(4 + 3 * math.log(n))  # 每代生成个体数
    # lam=10
    # print("lam is ",lam)
    mu = math.ceil(lam / 2)
    # if seg == 1:
    #     mu = 1
    # else:
    #     mu = math.ceil(lam / 2)

    F = [[f0[0]] * mu]
    cs = 0.3
    q_x = 0.3
    d_dor = 0.8
    epsi = 0.031
    alpha_k =1  # 噪声强度
    dorta_list = []


    y_plot = []
    norm_plot = []
    change_label_iter = []
    seg_iter = []
    # np.linalg.norm(v,float('inf')) > 0.2 or
    # while k_i == label and t < max_iter:
    # while (np.linalg.norm(pert_image-image)>0.005 or k_i == label) and t < max_iter:
    while (not mask) and t < max_iter:
        xt = []  # 存放每代产生的新解
        img_t = []
        pert_v = []
        for i in range(1, lam + 1):
            image_pic = torch.from_numpy(image.flatten().copy()).view(32,32, num_channels)#from_numpy(image.copy()).view(224,224, num_channels)
            # xi = torch.clamp(sample_xi(n, theta[-1], dorta, ccov, p)*0.01 ,-epsi,epsi) # 采样生成样本
            xi = sample_xi(n, torch.Tensor(theta[-1]), dorta, ccov, p)
            # xi_new = theta_to_v(xi, box[3],box[2], epsi)
            xi_new = xi * 4 / np.linalg.norm(xi)
            xt.append(xi_new)  # 用于计算后续
            # realclipdist = theta_to_v(xi,n,input_shape,newimg,epsi)        #变高维
            # v_show = np.array(xi.reshape(len_cur,len_cur))*alpha_k  #噪声图片
            v = torch.clamp(xi_new * alpha_k, -3, 3)  # 噪声图片
            # v = xi_new * alpha_k
            dst = torch.add(v.view(shape), roi.view(shape))

            # dst = torch.clamp(dst,-3,3)
            # v = dst-roi.view(shape_tu)

            # dst_new=np.pad(dst.flatten(),(box[0]*box[1],784-box[0]*box[1]-box[2]*box[3]),'constant', constant_values=(0,0))
            # dst = cv2.addWeighted(realclipdist,0.8,roi,0.2,0) #图像融合
            # dst = v_show + image.reshape(len_image,len_image)
            image_pic[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :] = dst
            image_adv = torch.clamp(torch.Tensor(image_pic).flatten(), -3, 3)
            v = image_adv - torch.from_numpy(image.flatten().copy())
            # print(torch.norm(v))

            #
            # if t % 1 == 0 and i % lam == 0:
            #     print("___________第%d代结果__________________" % t)
            #
            #     plt.subplot(1, 3, 1)
            #     plt.imshow(man_show(image).astype(np.uint8).reshape(64,64, num_channels))
            #     plt.title("%s" % node_lookup.id_to_string(label)[:17])
            #     plt.subplot(1, 3, 2)
            #     plt.imshow(v.numpy().reshape(64,64, num_channels))
            #     plt.title("%3f" % torch.norm(v))  # ,float('inf')
            #     plt.subplot(1, 3, 3)
            #     plt.imshow(man_show(image_pic.numpy()).astype(np.uint8))#
            #     plt.title("%s" % node_lookup.id_to_string(k_i)[:17])
            #     plt.show()

            # img_t.append(dst.reshape(1,input_shape[1])) #产生的新解,注意有lam个
            image_pic = image_pic.numpy().transpose(2, 0, 1)
            img_t.append(image_pic)  # 生成的对抗样本
            pert_v.append(v)  # n值的扰动

        img_pre = np.array(img_t) #np.expand_dims(image_pic, axis=0)
        img_pre = Variable(torch.from_numpy(img_pre).to(device).float())
        # print("扰动为：",pert_v)
        # print(xt) #注意生成解的形式
        y_sort, y = Fun_binary(session, 1, pert_v, label, img_pre)  # 计算目标函数值，返回对应的x的索引值，从大到小;以及随机采样的函数值
        num += lam

        # 选取前mu个优秀的解
        x_new = [xt[i] for i in y_sort[:mu]]
        y_new = [y[i] for i in y_sort[:mu]]  # 是已经从大到小排好的

        y_plot.append(y[y_sort[0]])
        norm_plot.append(np.linalg.norm(pert_v[y_sort[0]]))

        if len(F) < 3:  # 保持10代解的情况
            F.append(y_new)
        else:
            F = F[-2:]
            F.append(y_new)

        # 更新均值theta.更新搜索方向p，更新累积秩率s，更新变异强度dorta
        w_q = []  # 存放权重
        for q in range(1, mu + 1):
            w_q_v = (math.log(mu + 1) - math.log(q)) / (
                    mu * math.log(mu + 1) - sum([math.log(j) for j in range(1, mu + 1)]))
            w_q.append(w_q_v)
        w_q = torch.Tensor(w_q)
        # theta_q.append( w_q * x_new[q-1])
        # print(sum([w_q[i]*x_new[i] for i in range(mu)]))
        if len(theta) < 3:
            theta.append(sum([w_q[i] * x_new[i] for i in range(mu)]))
            # print('theta type', type(theta[0]))
        else:
            theta = theta[-2:]
            theta.append(sum([w_q[i] * x_new[i] for i in range(mu)]))

        ueff = 1 / sum(w_q.pow(2))
        p = (1 - c) * p + math.sqrt(c * (2 - c) * ueff) * (theta[-1] - theta[-2]) / dorta

        # F_z = torch.Tensor(F[-2] + F[-1])  # 合并  前一代索引为0到4，后一代为5-9
        # new_sort = torch.argsort(F_z,-1,True)
        F_z = F[-2] + F[-1]
        new_sort = np.array(-1 * np.array(F_z)).argsort()
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',new_sort) #[5 6 7 8 9 0 1 2 3 4]
        new_dict = dict(zip(new_sort, range(len(new_sort))))
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',new_dict) #{5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 0: 5, 1: 6, 2: 7, 3: 8, 4: 9}
        # for q in range(0,mu):
        # print(new_dict[q]-new_dict[q+5],"____________")
        q_diff = sum([w_q[q] * (new_dict[q] - new_dict[q + mu]) for q in range(0, mu)]) / mu
        s = (1 - cs) * s + cs * (q_diff - q_x)
        # print("s的值为：----------------------------------------",s)
        dorta_ad = dorta * math.exp(s / d_dor)
        dorta = dorta_ad
        if not mask and dorta_ad < 0.009:
            dorta = 0.5

        dorta_list.append(dorta)
        # print("dorta：-------------------------------------------------------------------------------------------------", dorta)
        # compute new perturbed image
        pert_image = img_t[y_sort[0]]
        t += 1  # 迭代次数
        if y[y_sort[0]] == 0:
            # print("成功")
            mask = True
            
    while (np.linalg.norm(left)-np.linalg.norm(right)>0.1) and t < max_iter:
        img_t = []
        pert_v = []
        mid = (left+right)/2
        # v = mid
        for i in range(1, lam+1):
            image_pic = torch.from_numpy(image.flatten().copy()).view(32,32, num_channels)#from_numpy(image.copy()).view(224,224, num_channels)
            # xi = torch.clamp(sample_xi(n, theta[-1], dorta, ccov, p)*0.01 ,-epsi,epsi) # 采样生成样本
            xi = sample_xi(n, torch.Tensor(mid), 1.0, ccov, -p)  #最大的约束为1也许就是在这里调整的!!!
            # xi_new = theta_to_v(xi, box[3],box[2], epsi)
            xi_new = xi * 1 / np.linalg.norm(xi)   #这也有可能
            v = torch.clamp(xi_new * alpha_k, -3, 3)  # 噪声图片
            # v = torch.clamp(xi_new * alpha_k, -0.031, 0.031)  # 噪声图片
            dst = torch.add(v.view(shape), roi.view(shape))
            image_pic[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :] = dst
            image_adv = torch.clamp(torch.Tensor(image_pic).flatten(), -3, 3)
            v = image_adv - torch.from_numpy(image.flatten().copy())
            image_pic = image_pic.numpy().transpose(2, 0, 1)
            img_t.append(image_pic)  # 生成的对抗样本
            pert_v.append(v)  # n值的扰动  #看pert_v生成的位置就知道constraint 的 epsilon位置

        img_pre = np.array(img_t)  # np.expand_dims(image_pic, axis=0)
        img_pre = Variable(torch.from_numpy(img_pre).to(device).float())
        y_sort, y = Fun_binary(session, 1, pert_v, label, img_pre)  # 计算目标函数值，返回对应的x的索引值，从大到小;以及随机采样的函数值
        num += lam

        if(num>8000):
            break

        if y[y_sort[0]] != 0:
            right = pert_v[y_sort[0]]
        else:
            left = pert_v[y_sort[0]]
            # print("fanshu ", np.linalg.norm(left))
        # print("zuo you ", np.linalg.norm(left), np.linalg.norm(right))
        # print("fanshu ", np.linalg.norm(v))
        t+=1

    return theta[-1], t, dorta, p, left, mask, pert_image, num

