from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch
import torch.nn.functional as F
import numpy as np
import math


class ClusterContrastTrainerSupport(object):
    def __init__(self, args, encoder, memory=None):
        super(ClusterContrastTrainerSupport, self).__init__()
        self.args = args
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for i in range(train_iters):
            ### define the degree direction of PLI
            cur_lambda = np.log((math.e - 1) * (epoch * train_iters + i) / (train_iters * self.args.epochs) + 1) * self.args.support_base_lambda * 0.5
            # input data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)

            ###### generate support samples
            f_out_support, _, dynamic_k = self.generate_hard_samples(f_out, labels, cur_lambda)
            support_labels = labels.repeat_interleave(torch.tensor(dynamic_k, device=labels.device))

            if self.args.use_support:
                all_feats = torch.cat([f_out, f_out_support])
                all_labels = torch.cat([labels, support_labels])
                loss = self.memory(all_feats, all_labels)
            else:
                loss = self.memory(f_out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


    def determine_k_dynamically(self, sorted_sims):
        """向量化动态K值计算"""
        # sorted_sims: [B, M] 已排序的相似度矩阵（降序排列）
        diffs = sorted_sims[:, :-1] - sorted_sims[:, 1:]    # [B, M-1]
        cum_mask = (diffs < self.args.dynamic_threshold).cumsum(dim=1)  # 累积满足条件的索引
        k_indices = (cum_mask == 0).sum(dim=1)+1    # 找到第一个不满足条件的点
        return torch.clamp(k_indices, max=self.args.max_k)  # 限制最大K值


    def generate_hard_samples(self, feats, labels, cur_lambda):
        B, C = feats.shape[0], feats.shape[-1]
        indexes, sims, mask, dynamic_k = self.find_nearest_center(feats, labels)
        max_k = indexes.size(1)
        # 计算目标中心
        target_center = self.memory.features[labels]  # [B, C]
        # 动态lambda系数 (不同样本不同系数)
        cur_lambda = cur_lambda * (dynamic_k.float() / self.args.max_k)  # 根据K值缩放
        cur_lambda = cur_lambda.view(-1, 1, 1)  # [B, 1, 1]
        # 批量获取最近中心 [B, max_k, C]
        nearest_center = self.memory.features[indexes]
        # 计算支持样本  ~f = f + λ∆f , ∆f = 1/2(c* - c)
        delta = (nearest_center - target_center.unsqueeze(1)) * cur_lambda * 0.5
        supports = feats.unsqueeze(1) + delta
        # 应用掩码并展平
        valid_supports = supports[mask].view(-1, C)
        return F.normalize(valid_supports, dim=1), sims, dynamic_k


    def find_nearest_center(self, feats, labels):
        B = feats.size(0)
        all_sim = feats.detach() @ self.memory.features.T.detach()  # [B, N]
        # 排除自身类别
        mask = (torch.arange(all_sim.size(1), device=feats.device) != labels.view(-1, 1))
        masked_sim = all_sim * mask.float() - (1 - mask.float()) * 1e6
        # 获取top(max_k+1)结果
        max_k = self.args.max_k
        sim_sorted, indices = torch.topk(masked_sim, k=max_k + 1, dim=1)  # [B, max_k+1]
        # 动态确定K值
        dynamic_k = self.determine_k_dynamically(sim_sorted)
        # 生成掩码矩阵
        k_mask = (torch.arange(max_k, device=feats.device) < dynamic_k.view(-1, 1))  # [B, max_k]
        return indices[:, :max_k], sim_sorted[:, :max_k], k_mask, dynamic_k
