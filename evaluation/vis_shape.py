from __future__ import print_function
import os
import argparse
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from data.shapenet_part import ShapeNetPart
from model.finetune_dgcnn_partseg import DGCNN_partseg
from torch.utils.data import DataLoader
from utils.util import cal_loss, IOStream
import tqdm
import matplotlib.pyplot as plt
import matplotlib


class Generate_txt_and_3d_img:
    def __init__(self, model, target_root, num_classes, testDataLoader, model_dict, color_map=None):
        # self.img_root = img_root  # 点云数据路径
        self.target_root = target_root  # 生成txt标签和预测结果路径
        self.testDataLoader = testDataLoader
        self.num_classes = num_classes
        self.color_map = color_map
        self.heat_map = False  # 控制是否输出heatmap
        self.label_path_txt = os.path.join(self.target_root, 'label_txt')  # 存放label的txt文件
        self.make_dir(self.label_path_txt)
        # print(model)
        self.model = model
        for (dict,model) in zip(model_dict,self.model):
            modelDict = torch.load(dict)
            model.load_state_dict(modelDict)
            model.eval()

        # 创建文件夹
        self.all_pred_image_path = []  # 所有预测结果的路径列表
        self.all_pred_txt_path = []  # 所有预测txt的路径列表
        for n in range(2):
            self.make_dir(os.path.join(self.target_root, str(n) + '_predict_txt'))
            self.make_dir(os.path.join(self.target_root, str(n) + '_predict_image'))
            self.all_pred_txt_path.append(os.path.join(self.target_root, str(n) + '_predict_txt'))
            self.all_pred_image_path.append(os.path.join(self.target_root, str(n) + '_predict_image'))
        "将模型对应的预测txt结果和img结果生成出来，对应几个模型就在列表中添加几个元素"

        self.generate_predict_to_txt()  # 生成预测txt
        self.draw_3d_img()  # 画图

    def generate_predict_to_txt(self):

        for batch_id, (points, label, target) in tqdm.tqdm(enumerate(self.testDataLoader),
                                                           total=len(self.testDataLoader), smoothing=0.9):
            if batch_id > 5:
                break
            # 点云数据、整个图像的标签、每个点的标签、  没有归一化的点云数据（带标签）torch.Size([1, 7, 2048])
            points = points.cuda()
            label = label.cuda()
            target = target.cuda()
            points = points.transpose(2, 1)

            # print('1',target.shape) # 1 torch.Size([1, 2048])
            xyz_feature_point = points[:, :6, :].cpu()
            # 将标签保存为txt文件
            target = target.cpu()
            points = points.cpu()
            point_set_without_normal = np.asarray(
                torch.cat([points.permute(0, 2, 1), target[:, :, None]], dim=-1)).squeeze(0)  # 代标签 没有归一化的点云数据  的numpy形式
            points = points.cuda()

            np.savetxt(os.path.join(self.label_path_txt, f'{batch_id}_label.txt'), point_set_without_normal,
                       fmt='%.04f')  # 将其存储为txt文件
            " points  torch.Size([16, 2048, 6])  label torch.Size([16, 1])  target torch.Size([16, 2048])"
            print(len(self.all_pred_txt_path))
            # assert len(self.model) == len(self.all_pred_txt_path), '路径与模型数量不匹配，请检查'

            for n,model,pred_path in zip(range(2),self.model, self.all_pred_txt_path):

                seg_pred = model(points, self.to_categorical(label, 16))
                seg_pred = seg_pred.transpose(2,1)
                seg_pred = seg_pred.cpu().data.numpy()
                # =================================================
                # seg_pred = np.argmax(seg_pred, axis=-1)  # 获得网络的预测结果 b n c
                if self.heat_map:
                    out = np.asarray(np.sum(seg_pred, axis=2))
                    seg_pred = ((out - np.min(out) / (np.max(out) - np.min(out))))
                else:
                    seg_pred = np.argmax(seg_pred, axis=-1)  # 获得网络的预测结果 b n c
                # =================================================
                # print(seg_pred.shape,xyz_feature_point.shape)
                seg_pred = np.concatenate([np.asarray(xyz_feature_point), seg_pred[:, None, :]],
                                          axis=1).transpose((0, 2, 1)).squeeze(0)  # 将点云与预测结果进行拼接，准备生成txt文件
                svae_path = os.path.join(pred_path, f'{n}_{batch_id}.txt')
                np.savetxt(svae_path, seg_pred, fmt='%.04f')
        each_label0 = os.listdir(self.all_pred_txt_path[0])
        each_label1 = os.listdir(self.all_pred_txt_path[1])
        print(each_label0,each_label1)
        for num in range(50):
            num = str(num)
            svae_path = os.path.join('./outputs/vis_result/con_predict_txt/'+f'2_{num}.txt')
            p0 = np.loadtxt('./outputs/vis_result/0_predict_txt/'+f'0_{num}.txt')
            p1 = np.loadtxt('./outputs/vis_result/1_predict_txt/'+f'1_{num}.txt')

            comparison_matrix = np.where(p0[:,3] == p1[:,3],0,1)
            p0[:,3] = comparison_matrix
            # svae_path = os.path.join
            np.savetxt(svae_path,p0, fmt='%.04f')

    def draw_3d_img(self):
        #   调用matpltlib 画3d图像
        print(self.label_path_txt,'self.label_path_txt')
        each_label = os.listdir(self.label_path_txt)  # 所有标签txt路径
        self.label_path_3d_img = os.path.join(self.target_root, 'label_3d_img')
        self.make_dir(self.label_path_3d_img)
        assert len(self.all_pred_txt_path) == len(self.all_pred_image_path)

        for i, (pre_txt_path, save_img_path,name) in enumerate(
                zip(self.all_pred_txt_path, self.all_pred_image_path,['model,model_weak'])):
            each_txt_path = os.listdir(pre_txt_path)  # 拿到txt文件的全部名字

            for idx, (txt, lab) in tqdm.tqdm(enumerate(zip(each_txt_path, each_label)), total=len(each_txt_path)):
                if i == 0:
                    self.draw_each_img(os.path.join(self.label_path_txt, lab), idx, heat_maps=False)
                self.draw_each_img(os.path.join(pre_txt_path, txt), idx, name=name, save_path=save_img_path,
                                   heat_maps=self.heat_map)
        new_path = './outputs/vis_result/con_predict_txt'
        new_each_txt_path = os.listdir(new_path)
        print(new_each_txt_path, '----------------------')
        for idx, (txt, lab) in tqdm.tqdm(enumerate(zip(new_each_txt_path, each_label)), total=len(new_each_txt_path)):
            self.draw_each_img(os.path.join(new_path, txt), idx, name='2', save_path='./outputs/vis_result/con_predict_image',
                               heat_maps=self.heat_map)

        print(f'所有预测图片已生成完毕，请前往：{self.all_pred_image_path} 查看')

    def draw_each_img(self, root, idx, name=None, skip=1, save_path=None, heat_maps=False):
        "root：每个txt文件的路径"
        points = np.loadtxt(root)[:, :3]  # 点云的xyz坐标
        points_all = np.loadtxt(root)  # 点云的所有坐标
        points = self.pc_normalize(points)
        skip = skip  # Skip every n points

        fig = plt.figure()
        # ax = plt.axes(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        point_range = range(0, points.shape[0], skip)  # skip points to prevent crash
        x = points[point_range, 0]
        z = points[point_range, 1]
        y = points[point_range, 2]

        "根据传入的类别数 自定义生成染色板  标签 0对应 随机颜色1  标签1 对应随机颜色2"
        if self.color_map is not None:
            color_map = self.color_map
        else:
            color_map = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, num_classes))}
        if name == '2':
            print('yan se xiugai')
            color_map[0] = 'gray'
            color_map[1] = 'red'

        Label = points_all[point_range, -1]  # 拿到标签
        # 将标签传入前面的字典，找到对应的颜色 并放入列表

        Color = list(map(lambda x: color_map[x], Label))
        # if name == '2':
        #     Color[2] = 'green'
        #     Color[1] = 'red'
        ax.scatter(x,  # x
                   y,  # y
                   z,  # z
                   c=Color,  # Color,  # height data for color
                   s=25,
                   marker=".")
        ax.axis('auto')  # {equal, scaled}
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('off')  # 设置坐标轴不可见
        ax.grid(False)  # 设置背景网格不可见
        ax.view_init(elev=45, azim=45)
        # plt.show()
        if save_path is None:
            plt.savefig(os.path.join(self.label_path_3d_img, f'{idx}_label_img.png'), dpi=300, bbox_inches='tight',
                        transparent=True)
        else:
            plt.savefig(os.path.join(save_path, f'{idx}_{name}_img.png'), dpi=300, bbox_inches='tight',
                        transparent=True)

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def make_dir(self, root):
        if os.path.exists(root):
            print(f'{root} 路径已存在 无需创建')
        else:
            os.mkdir(root)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
        if (y.is_cuda):
            return new_y.cuda()
        return new_y

    def load_cheackpoint_for_models(self, name, model, cheackpoints):

        assert cheackpoints is not None, '请填写权重文件'
        assert model is not None, '请实例化模型'

        for n, m, c in zip(name, model, cheackpoints):
            print(f'正在加载{n}的权重.....')
            weight_dict = torch.load(os.path.join(c, 'best_model.pth'))
            m.load_state_dict(weight_dict['model_state_dict'])
            print(f'{n}权重加载完毕')


if __name__ == '__main__':
    import copy
    import argparse

    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    args = parser.parse_args()
    # img_root = r'你的数据集路径'  # 数据集路径
    target_root = './outputs/vis_result'  # 输出结果路径

    num_classes = 50  # 填写数据集的类别数 如果是s3dis这里就填13   shapenet这里就填50
    choice_dataset = 'ShapeNet'  # 预测ShapNet数据集
    # 导入模型  部分
    "所有的模型以PointNet++为标准  输入两个参数 输出两个参数，如果模型仅输出一个，可以将其修改为多输出一个None！！！！"
    # ==============================================

    class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                     'motorbike',
                     'mug', 'pistol', 'rocket', 'skateboard', 'table']



    # ============================================
    # 实例化数据集
    "Dataset同理，都按ShapeNet格式输出三个变量 point_set, cls, seg # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别"
    "不是这个格式的话就手动添加一个"

    if choice_dataset == 'ShapeNet':
        print('实例化ShapeNet')
        test_loader = DataLoader(
            ShapeNetPart(partition='test', num_points=2048, class_choice=args.class_choice),
            num_workers=1, shuffle=False, batch_size=1, drop_last=False)
        # TEST_DATASET = ShapeNetPart(root=img_root, npoints=2048, split='test', normal_channel=True)
        # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0,
        #                                              drop_last=True)
        color_map = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, num_classes))}
    else:
        TEST_DATASET = S3DISDataset(split='test', data_root=img_root, num_point=4096, test_area=5,
                                    block_size=1.0, sample_rate=1.0, transform=None)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0,
                                                     pin_memory=True, drop_last=True)
        color_maps = [(152, 223, 138), (174, 199, 232), (255, 127, 14), (91, 163, 138), (255, 187, 120), (188, 189, 34),
                      (140, 86, 75)
            , (255, 152, 150), (214, 39, 40), (197, 176, 213), (196, 156, 148), (23, 190, 207), (112, 128, 144)]

        color_map = []
        for i in color_maps:
            tem = ()
            for j in i:
                j = j / 255
                tem += (j,)
            color_map.append(tem)
        print('实例化S3DIS')
    seg_num_all = test_loader.dataset.seg_num_all
    # 将模型和权重路径填写到字典中，以下面这个格式填写就可以了
    # 如果加载权重报错，可以查看类里面的加载权重部分，进行对应修改即可
    model_dict = ['./outputs/exp_seg_down_1/models/model.t7','./outputs/exp_seg_down_1/models/model_weak.t7']
    model1 = DGCNN_partseg(args, seg_num_all).cuda()
    model2 = DGCNN_partseg(args, seg_num_all).cuda()
    c = Generate_txt_and_3d_img([model1,model2], target_root, num_classes, test_loader, model_dict, color_map)

