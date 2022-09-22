import sys
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import cv2
#

def get_score(plt,abe,gt,threshold,model_name):
    rr = abe < threshold / 100
    x = np.linspace(0, 1000, 50)
    x = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    y = [0 for _ in x]
    score = 0
    for i in range(len(x) - 1):
        l = x[i]
        r = x[i + 1]
        mask = (l <= gt) * (gt < r)
        y[i] = np.sum(rr[mask]) / np.sum(mask)
        if np.isnan(y[i]):
            y[i] = 0
        score = score + y[i]
    score = score / 10
    print('{}_Average_Score_{}:{}'.format(model_name,threshold,score))
    plt.plot(x, y)

def visualizer_plot(model_name,save_path, gt, res, lt_res,source):
    abe = []
    abe.append(np.fabs(gt - source)/gt)
    abe.append(np.fabs(gt - res) / gt)
    abe.append(np.fabs(gt - lt_res) / gt)
    ap_threshold = [5,10]
    buffer = []
    model_names = ['ToF-Source','ToF-Net','ToF-Net-LT']
    for threshold in ap_threshold:
        for i, model_name in enumerate(model_names):
            get_score(plt,abe[i],gt,threshold,model_name)
        plt.xlabel('GT(cm)')
        plt.ylabel('Accu_RE')
        plt.legend(model_names)
        plt.title('Accu Curve AP {}'.format(threshold))
        plot_name = '{}_eval_{}.jpg'.format(model_name,threshold)
        plot_path = '{}/{}'.format(save_path,plot_name)
        buffer.append(plot_name)
        plt.savefig(plot_path)
        plt.clf()

def test(path,gt_path):
    file_list = os.listdir(path)
    #print(depth_list)
    total_right_5 = 0
    total_right_10 = 0
    total_num = 0
    gt = []
    res = []
    source = []
    lt_res = []
    for i, file_name in enumerate(file_list):
        #print(path+ '/' + file_name)
        #print(gt_path+ '/' + file_name)
        res_image_data = scipy.io.loadmat(path+ '/' + file_name)
        source_image_data = scipy.io.loadmat(gt_path+ '/' + file_name)
        source_image = source_image_data['im_pair']

        real_B = source_image[0, :, 240:]
        source_depth = source_image[1, :, 240:]
        fake_B = res_image_data['x'][0, :, :]

        real_B[np.isnan(real_B)] = 1.5
        fake_B[np.isnan(fake_B)] = 1.5

        re = (np.abs(fake_B - real_B) / real_B)
        total_right_5 += np.sum(re < 0.05)
        total_right_10 += np.sum(re < 0.1)
        total_num += np.sum(re > 0)
        gt = np.append(gt, 1000*real_B)
        res = np.append(res, 1000*fake_B)
        lt_res = np.append(lt_res, 1000*fake_B * 0.96 + 0.055)
        source = np.append(source, 1000*source_depth)
        print('Processing: '+str(i+1)+'/'+str(len(file_list)), end='\r')

    print('ToF-Net Accu_RE5 is: ' + str((total_right_5 / total_num).item()))
    print('ToF-Net Accu_RE10 is: ' + str((total_right_10 / total_num).item()))
    visualizer_plot('iTof',path+'/../',gt,res,lt_res,source)

def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print ('[+] usage: python eval.py gt_path tof_net_path')
        sys.exit(-1)

    gt_path = sys.argv[1]
    tof_net_path = sys.argv[2]
    test(tof_net_path,gt_path)


if __name__ == "__main__":
    main()
