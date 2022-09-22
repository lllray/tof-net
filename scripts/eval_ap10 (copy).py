import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import cv2
# 
def test(mode):
    if mode == 0:
        result_dir = './backup/res/itof/'
    else:
        result_dir = './backup/res/big/'
    file_dir_res = os.listdir(result_dir)
    gt_dir = './backup/data/gt/'
    file_dir_gt = os.listdir(gt_dir)
    gt = []
    res = []
    for i, file_name in enumerate(file_dir_res):
        res_image_data = scipy.io.loadmat(result_dir + file_name)
        if mode == 0:
            # res_image_mat = cv2.Mat(res_image_data['s_depth'])
            # print(res_image_mat.dtype)
            # res_image = cv2.medianBlur(res_image_mat, 3).flatten()
            res_image = res_image_data['s_depth'].flatten()
            file_name = file_name[:20] + file_name[27:]
        else:
            if mode == 1:
                res_image = 10 * res_image_data['x'].flatten()
            else:
                res_image = 9.6 * res_image_data['x'].flatten() + 0.055  # Linear Transform
            file_name = file_name[:20] + '_depth_' + file_name[-20:]
        gt_image_data = scipy.io.loadmat(gt_dir + file_name)
        gt_image = gt_image_data['depth'].flatten()


        gt = np.append(gt, 100 * gt_image)
        res = np.append(res, 100 * res_image)
        print('Processing: '+str(i+1)+'/'+str(len(file_dir_res)), end='\r')

    abe = np.fabs(gt - res)
    abe[np.isnan(gt)*np.isnan(res)] = 0
    abe = np.fabs(gt - res)/gt
    abe[np.isnan(gt)*np.isnan(res)] = 0
    rr = abe < 0.05
    x = np.linspace(0, 1000, 50)
    x = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    y = [0 for _ in x]
    for i in range(len(x)-1):
        l = x[i]
        r = x[i+1]
        mask = (l<=gt) * (gt<r)
        y[i] = np.sum(rr[mask])/np.sum(mask)
        print(str(y[i]))
    # mask = np.isnan(gt)
    # np.append(x, 1100)
    # np.append(y, np.sum(rr[mask])/np.sum(mask))
    # print(np.sum(rr[np.isnan(gt)])/np.sum(np.isnan(gt)))
    plt.plot(x, y)

test(0)
test(1)
test(2)
plt.xlabel('GT(cm)')
plt.ylabel('Accu_RE5(%)')
plt.legend(['iToF-Source', 'ToF-Net-Original', 'ToF-Net-LT'])
plt.title('Accu curve')
plt.savefig('./2.jpg')