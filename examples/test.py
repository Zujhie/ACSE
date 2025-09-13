from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from acse import datasets
from acse import models
from acse.models.dsbn import convert_dsbn, convert_bn
from acse.evaluators import Evaluator, extract_features
from acse.utils.data import transforms as T
from acse.utils.data.preprocessor import Preprocessor
from acse.utils.logging import Logger
from acse.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter


def visualize_tsne(features, labels, title="t-SNE Visualization", save_path="tsne_visualization.png"):
    print("Selecting top 10 most frequent identities...")
    # counter = Counter(labels)
    # top_10_ids = [pid for pid, _ in counter.most_common(10)]
    # print(f"Selected PIDs: {top_10_ids}")  # 打印选中的 PID
    # mask = np.isin(labels, top_10_ids)

    # 固定pid
    selected_pids = [67, 46, 136, 104, 6, 167, 63, 72, 115, 131]
    mask = np.isin(labels, selected_pids)

    selected_features = features[mask]
    selected_labels = labels[mask]

    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedded_features = tsne.fit_transform(selected_features)

    unique_pids = np.unique(selected_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_pids))  # Use a color map with enough distinct colors
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H']  # Different markers

    plt.figure(figsize=(10, 8))

    for i, pid in enumerate(unique_pids):
        pid_mask = selected_labels == pid
        plt.scatter(embedded_features[pid_mask, 0], embedded_features[pid_mask, 1],
                    c=[colors(i)], marker=markers[i % len(markers)], label=f'PID {pid}', alpha=0.7)

    # plt.colorbar()  # Display color bar
    # plt.title(title)
    # plt.xlabel("t-SNE Component 1")
    # plt.ylabel("t-SNE Component 2")
    # plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title="PIDs")

    plt.xticks([])  # 不显示横轴刻度
    plt.yticks([])  # 不显示纵轴刻度
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.savefig(save_path, bbox_inches='tight')
    print(f"t-SNE visualization saved to {save_path}")
    plt.close()


def get_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)   # (fname,pid,cid)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return dataset, test_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    cudnn.benchmark = True

    log_dir = osp.dirname(args.resume)
    sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset, test_loader = get_data(args.dataset, args.data_dir, args.height,
                                    args.width, args.batch_size, args.workers)

    # Create model
    model = models.create(args.arch, pretrained=False, num_features=args.features, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    if args.dsbn:
        print("==> Load the model with domain-specific BNs")
        convert_dsbn(model)

    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint['state_dict'], model, strip='module.')

    if args.dsbn:
        print("==> Test with {}-domain BNs".format("source" if args.test_source else "target"))
        convert_bn(model, use_target=(not args.test_source))

    model.cuda()
    model = nn.DataParallel(model)

    # save cids and pids
    # save_dir = "/data7/jiezhu/ReID/cc_refine/logs/file"
    #
    # cids = np.array([cid for _, _, cid in sorted(dataset.train)])
    # pids = np.array([pid for _, pid, _ in sorted(dataset.train)])
    # np.save(save_dir, args.dataset + '_cids.npy', cids)
    # np.save(save_dir, args.dataset + '_pids.npy', pids)

    # Evaluator
    model.eval()
    evaluator = Evaluator(model)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, rerank=args.rerank)

    # 保存训练样本提取出来的特征
    if args.save_features:
        cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers,
                                         testset=sorted(dataset.train))
        features, _ = extract_features(model, cluster_loader)  # {fname:(2048)}
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)  # (N,2048)
        labels = np.array([pid for _, pid, _ in sorted(dataset.train)])
        features = features.cpu().numpy()
        np.save('/root/autodl-tmp/code/cc_refine/logs/file/features.npy', features)
        np.save('/root/autodl-tmp/code/cc_refine/logs/file/labels.npy', labels)

        visualize_tsne(features, labels, title=f"baseline",
                       save_path="/root/autodl-tmp/code/cc_refine/logs/file/tsne_visualization.png")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)

    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    # testing configs
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--save-features', action='store_true')
    parser.add_argument('--dsbn', action='store_true',
                        help="test on the model with domain-specific BN")
    parser.add_argument('--test-source', action='store_true',
                        help="test on the source domain")
    parser.add_argument('--seed', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',default=osp.join(working_dir, 'data'))
    parser.add_argument('--log-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'test'))
    parser.add_argument('--pooling-type', type=str, default='avg')
    parser.add_argument('--embedding_features_path', type=str,
                        default='/media/yixuan/Project/guangyuan/workpalces/SpCL/embedding_features/mark1501_res50_ibn/')
    main()
