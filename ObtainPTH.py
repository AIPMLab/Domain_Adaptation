import configargparse
import data_loader
import os
import models
import utils
from utils import str2bool
import numpy as np
import random
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize, LogNorm, SymLogNorm


def get_parser(source, target,model):
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    # parser.add("--config", is_config_file=True, help='DSAN/DSAN.yaml',default="BNM/BNM.yaml")
    # parser.add("--config", is_config_file=True, help='DSAN/DSAN.yaml', default="DeepCoral/DeepCoral.yaml")
    # parser.add("--config", is_config_file=True, help='DSAN/DSAN.yaml', default="DANN/dann.yaml")
    parser.add("--config", is_config_file=True, help='DSAN/DSAN.yaml', default=model)
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, default='./prostate')
    parser.add_argument('--src_domain', type=str, default=source)
    parser.add_argument('--tgt_domain', type=str, default=target)

    # training related
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    return parser


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True,
        num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True,
        num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class


def get_model(args):
    model = models.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter,
        use_bottleneck=args.use_bottleneck).to(args.device)
    return model


def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                                nesterov=False
                                )
    # optimizer = torch.optim.Adam(params,lr= args.lr,weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay,momentum=args.momentum,nesterov=True
    #                             )
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
        -args.lr_decay))
    return scheduler


import numpy as np
import matplotlib.pyplot as plt


def test(model, target_test_loader, args, count):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)

    with torch.no_grad():
        target_features = []
        target_labels = []
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

            if count == args.n_epoch:
                target_features_batch = model.bottleneck_layer(model.base_network(data))
                target_features.append(target_features_batch.cpu().numpy())
                target_labels.append(target.cpu().numpy())

        acc = 100. * correct / len_target_dataset
        #
        if count == args.n_epoch:
            target_features = np.concatenate(target_features)
            target_labels = np.concatenate(target_labels)


    return acc, test_loss.avg, target_features, target_labels


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args, name, location):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    print(len_source_loader,len_target_loader)
    n_batch = min(len_source_loader, len_target_loader)
    print(n_batch)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch
        print(n_batch)

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []
    count = 1
    all_outputs_train = []
    for e in range(1, args.n_epoch + 1):
        all_outputs_train = []
        source_features = []
        source_labels = []
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        for _ in range(n_batch):
            data_source, label_source = next(iter_source)
            data_target, _ = next(iter_target)
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)
            clf_loss, transfer_loss = model(data_source, data_target, label_source)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss
            if count == args.n_epoch:
                source_features_batch = model.bottleneck_layer(model.base_network(data_source))
                source_features.append(source_features_batch.cpu().detach().numpy())
                source_labels.append(label_source.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())

            s_output_train = model.predict(data_source)
            all_outputs_train.append(s_output_train.cpu().detach().numpy())

        # log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        #     all_outputs = all_outputs_train
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
            e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Test
        stop += 1
        test_acc, test_loss, target_features, target_labels= test(model, target_test_loader, args, count)
        log.append([test_acc.item()])
        info += ', test_loss {:.4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt(name, np_log, delimiter=',', fmt='%.6f')
        if count == args.n_epoch:
            source_features = np.concatenate(source_features)
            source_labels = np.concatenate(source_labels)
            merged_features = np.concatenate((source_features, target_features), axis=0)
            # source_features = np.append(target_features)
            print(len(merged_features))
            pca = PCA(n_components=50,random_state=0)
            # pca_features = pca.fit_transform(source_features)
            # target_features = pca.fit_transform(target_features)
            merged_features = pca.fit_transform(merged_features)

            tsne = TSNE(n_components=2, random_state=0)
            # tsne_features = tsne.fit_transform(pca_features)
            # target_features = tsne.fit_transform(target_features)
            merged_features = tsne.fit_transform(merged_features)
            plt.figure(figsize=(10, 5))
            print(args.n_iter_per_epoch * n_batch)
            # plt.scatter(tsne_features[0:30 :, 0], tsne_features[0:30 :, 1], c='red',norm=Normalize(vmin=0, vmax=1))
            plt.scatter(merged_features[0:args.n_iter_per_epoch * args.batch_size:, 0], merged_features[0:args.n_iter_per_epoch * args.batch_size:, 1], c='red', norm=Normalize(vmin=0, vmax=1),marker='o',s=10,label='source')
            plt.scatter(merged_features[args.n_iter_per_epoch * args.batch_size:args.n_iter_per_epoch * args.batch_size + len(target_test_loader.dataset), 0], merged_features[args.n_iter_per_epoch * args.batch_size:args.n_iter_per_epoch * args.batch_size + len(target_test_loader.dataset), 1], c='blue', norm=Normalize(vmin=0, vmax=1),marker='D',s=10,label='target')
            plt.legend()
            # plt.title('t-SNE Visualization of Target Features')
            plt.savefig(location)
        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)

        count += 1
    print('Transfer result: {:.4f}'.format(best_acc))
    name = ["conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "layer1.0.conv1.weight", "layer1.0.bn1.weight", "layer1.0.bn1.bias", "layer1.0.bn1.running_mean", "layer1.0.bn1.running_var", "layer1.0.conv2.weight", "layer1.0.bn2.weight", "layer1.0.bn2.bias", "layer1.0.bn2.running_mean", "layer1.0.bn2.running_var", "layer1.1.conv1.weight", "layer1.1.bn1.weight", "layer1.1.bn1.bias", "layer1.1.bn1.running_mean", "layer1.1.bn1.running_var", "layer1.1.conv2.weight", "layer1.1.bn2.weight", "layer1.1.bn2.bias", "layer1.1.bn2.running_mean", "layer1.1.bn2.running_var", "layer1.2.conv1.weight", "layer1.2.bn1.weight", "layer1.2.bn1.bias", "layer1.2.bn1.running_mean", "layer1.2.bn1.running_var", "layer1.2.conv2.weight", "layer1.2.bn2.weight", "layer1.2.bn2.bias", "layer1.2.bn2.running_mean", "layer1.2.bn2.running_var", "layer2.0.conv1.weight", "layer2.0.bn1.weight", "layer2.0.bn1.bias", "layer2.0.bn1.running_mean", "layer2.0.bn1.running_var", "layer2.0.conv2.weight", "layer2.0.bn2.weight", "layer2.0.bn2.bias", "layer2.0.bn2.running_mean", "layer2.0.bn2.running_var", "layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight", "layer2.0.downsample.1.bias", "layer2.0.downsample.1.running_mean", "layer2.0.downsample.1.running_var", "layer2.1.conv1.weight", "layer2.1.bn1.weight", "layer2.1.bn1.bias", "layer2.1.bn1.running_mean", "layer2.1.bn1.running_var", "layer2.1.conv2.weight", "layer2.1.bn2.weight", "layer2.1.bn2.bias", "layer2.1.bn2.running_mean", "layer2.1.bn2.running_var", "layer2.2.conv1.weight", "layer2.2.bn1.weight", "layer2.2.bn1.bias", "layer2.2.bn1.running_mean", "layer2.2.bn1.running_var", "layer2.2.conv2.weight", "layer2.2.bn2.weight", "layer2.2.bn2.bias", "layer2.2.bn2.running_mean", "layer2.2.bn2.running_var", "layer2.3.conv1.weight", "layer2.3.bn1.weight", "layer2.3.bn1.bias", "layer2.3.bn1.running_mean", "layer2.3.bn1.running_var", "layer2.3.conv2.weight", "layer2.3.bn2.weight", "layer2.3.bn2.bias", "layer2.3.bn2.running_mean", "layer2.3.bn2.running_var", "layer3.0.conv1.weight", "layer3.0.bn1.weight", "layer3.0.bn1.bias", "layer3.0.bn1.running_mean", "layer3.0.bn1.running_var", "layer3.0.conv2.weight", "layer3.0.bn2.weight", "layer3.0.bn2.bias", "layer3.0.bn2.running_mean", "layer3.0.bn2.running_var", "layer3.0.downsample.0.weight", "layer3.0.downsample.1.weight", "layer3.0.downsample.1.bias", "layer3.0.downsample.1.running_mean", "layer3.0.downsample.1.running_var", "layer3.1.conv1.weight", "layer3.1.bn1.weight", "layer3.1.bn1.bias", "layer3.1.bn1.running_mean", "layer3.1.bn1.running_var", "layer3.1.conv2.weight", "layer3.1.bn2.weight", "layer3.1.bn2.bias", "layer3.1.bn2.running_mean", "layer3.1.bn2.running_var", "layer3.2.conv1.weight", "layer3.2.bn1.weight", "layer3.2.bn1.bias", "layer3.2.bn1.running_mean", "layer3.2.bn1.running_var", "layer3.2.conv2.weight", "layer3.2.bn2.weight", "layer3.2.bn2.bias", "layer3.2.bn2.running_mean", "layer3.2.bn2.running_var", "layer3.3.conv1.weight", "layer3.3.bn1.weight", "layer3.3.bn1.bias", "layer3.3.bn1.running_mean", "layer3.3.bn1.running_var", "layer3.3.conv2.weight", "layer3.3.bn2.weight", "layer3.3.bn2.bias", "layer3.3.bn2.running_mean", "layer3.3.bn2.running_var", "layer3.4.conv1.weight", "layer3.4.bn1.weight", "layer3.4.bn1.bias", "layer3.4.bn1.running_mean", "layer3.4.bn1.running_var", "layer3.4.conv2.weight", "layer3.4.bn2.weight", "layer3.4.bn2.bias", "layer3.4.bn2.running_mean", "layer3.4.bn2.running_var", "layer3.5.conv1.weight", "layer3.5.bn1.weight", "layer3.5.bn1.bias", "layer3.5.bn1.running_mean", "layer3.5.bn1.running_var", "layer3.5.conv2.weight", "layer3.5.bn2.weight", "layer3.5.bn2.bias", "layer3.5.bn2.running_mean", "layer3.5.bn2.running_var", "layer4.0.conv1.weight", "layer4.0.bn1.weight", "layer4.0.bn1.bias", "layer4.0.bn1.running_mean", "layer4.0.bn1.running_var", "layer4.0.conv2.weight", "layer4.0.bn2.weight", "layer4.0.bn2.bias", "layer4.0.bn2.running_mean", "layer4.0.bn2.running_var", "layer4.0.downsample.0.weight", "layer4.0.downsample.1.weight", "layer4.0.downsample.1.bias", "layer4.0.downsample.1.running_mean", "layer4.0.downsample.1.running_var", "layer4.1.conv1.weight", "layer4.1.bn1.weight", "layer4.1.bn1.bias", "layer4.1.bn1.running_mean", "layer4.1.bn1.running_var", "layer4.1.conv2.weight", "layer4.1.bn2.weight", "layer4.1.bn2.bias", "layer4.1.bn2.running_mean", "layer4.1.bn2.running_var", "layer4.2.conv1.weight", "layer4.2.bn1.weight", "layer4.2.bn1.bias", "layer4.2.bn1.running_mean", "layer4.2.bn1.running_var", "layer4.2.conv2.weight", "layer4.2.bn2.weight", "layer4.2.bn2.bias", "layer4.2.bn2.running_mean", "layer4.2.bn2.running_var", "fc.weight", "fc.bias"]
    adjust = ["base_network.conv1.weight", "base_network.bn1.weight", "base_network.bn1.bias",
               "base_network.bn1.running_mean", "base_network.bn1.running_var", "base_network.layer1.0.conv1.weight",
               "base_network.layer1.0.bn1.weight", "base_network.layer1.0.bn1.bias",
               "base_network.layer1.0.bn1.running_mean", "base_network.layer1.0.bn1.running_var",
               "base_network.layer1.0.conv2.weight", "base_network.layer1.0.bn2.weight",
               "base_network.layer1.0.bn2.bias", "base_network.layer1.0.bn2.running_mean",
               "base_network.layer1.0.bn2.running_var", "base_network.layer1.1.conv1.weight",
               "base_network.layer1.1.bn1.weight", "base_network.layer1.1.bn1.bias",
               "base_network.layer1.1.bn1.running_mean", "base_network.layer1.1.bn1.running_var",
               "base_network.layer1.1.conv2.weight", "base_network.layer1.1.bn2.weight",
               "base_network.layer1.1.bn2.bias", "base_network.layer1.1.bn2.running_mean",
               "base_network.layer1.1.bn2.running_var", "base_network.layer1.2.conv1.weight",
               "base_network.layer1.2.bn1.weight", "base_network.layer1.2.bn1.bias",
               "base_network.layer1.2.bn1.running_mean", "base_network.layer1.2.bn1.running_var",
               "base_network.layer1.2.conv2.weight", "base_network.layer1.2.bn2.weight",
               "base_network.layer1.2.bn2.bias", "base_network.layer1.2.bn2.running_mean",
               "base_network.layer1.2.bn2.running_var", "base_network.layer2.0.conv1.weight",
               "base_network.layer2.0.bn1.weight", "base_network.layer2.0.bn1.bias",
               "base_network.layer2.0.bn1.running_mean", "base_network.layer2.0.bn1.running_var",
               "base_network.layer2.0.conv2.weight", "base_network.layer2.0.bn2.weight",
               "base_network.layer2.0.bn2.bias", "base_network.layer2.0.bn2.running_mean",
               "base_network.layer2.0.bn2.running_var", "base_network.layer2.0.downsample.0.weight",
               "base_network.layer2.0.downsample.1.weight", "base_network.layer2.0.downsample.1.bias",
               "base_network.layer2.0.downsample.1.running_mean", "base_network.layer2.0.downsample.1.running_var",
               "base_network.layer2.1.conv1.weight", "base_network.layer2.1.bn1.weight",
               "base_network.layer2.1.bn1.bias", "base_network.layer2.1.bn1.running_mean",
               "base_network.layer2.1.bn1.running_var", "base_network.layer2.1.conv2.weight",
               "base_network.layer2.1.bn2.weight", "base_network.layer2.1.bn2.bias",
               "base_network.layer2.1.bn2.running_mean", "base_network.layer2.1.bn2.running_var",
               "base_network.layer2.2.conv1.weight", "base_network.layer2.2.bn1.weight",
               "base_network.layer2.2.bn1.bias", "base_network.layer2.2.bn1.running_mean",
               "base_network.layer2.2.bn1.running_var", "base_network.layer2.2.conv2.weight",
               "base_network.layer2.2.bn2.weight", "base_network.layer2.2.bn2.bias",
               "base_network.layer2.2.bn2.running_mean", "base_network.layer2.2.bn2.running_var",
               "base_network.layer2.3.conv1.weight", "base_network.layer2.3.bn1.weight",
               "base_network.layer2.3.bn1.bias", "base_network.layer2.3.bn1.running_mean",
               "base_network.layer2.3.bn1.running_var", "base_network.layer2.3.conv2.weight",
               "base_network.layer2.3.bn2.weight", "base_network.layer2.3.bn2.bias",
               "base_network.layer2.3.bn2.running_mean", "base_network.layer2.3.bn2.running_var",
               "base_network.layer3.0.conv1.weight", "base_network.layer3.0.bn1.weight",
               "base_network.layer3.0.bn1.bias", "base_network.layer3.0.bn1.running_mean",
               "base_network.layer3.0.bn1.running_var", "base_network.layer3.0.conv2.weight",
               "base_network.layer3.0.bn2.weight", "base_network.layer3.0.bn2.bias",
               "base_network.layer3.0.bn2.running_mean", "base_network.layer3.0.bn2.running_var",
               "base_network.layer3.0.downsample.0.weight", "base_network.layer3.0.downsample.1.weight",
               "base_network.layer3.0.downsample.1.bias", "base_network.layer3.0.downsample.1.running_mean",
               "base_network.layer3.0.downsample.1.running_var", "base_network.layer3.1.conv1.weight",
               "base_network.layer3.1.bn1.weight", "base_network.layer3.1.bn1.bias",
               "base_network.layer3.1.bn1.running_mean", "base_network.layer3.1.bn1.running_var",
               "base_network.layer3.1.conv2.weight", "base_network.layer3.1.bn2.weight",
               "base_network.layer3.1.bn2.bias", "base_network.layer3.1.bn2.running_mean",
               "base_network.layer3.1.bn2.running_var", "base_network.layer3.2.conv1.weight",
               "base_network.layer3.2.bn1.weight", "base_network.layer3.2.bn1.bias",
               "base_network.layer3.2.bn1.running_mean", "base_network.layer3.2.bn1.running_var",
               "base_network.layer3.2.conv2.weight", "base_network.layer3.2.bn2.weight",
               "base_network.layer3.2.bn2.bias", "base_network.layer3.2.bn2.running_mean",
               "base_network.layer3.2.bn2.running_var", "base_network.layer3.3.conv1.weight",
               "base_network.layer3.3.bn1.weight", "base_network.layer3.3.bn1.bias",
               "base_network.layer3.3.bn1.running_mean", "base_network.layer3.3.bn1.running_var",
               "base_network.layer3.3.conv2.weight", "base_network.layer3.3.bn2.weight",
               "base_network.layer3.3.bn2.bias", "base_network.layer3.3.bn2.running_mean",
               "base_network.layer3.3.bn2.running_var", "base_network.layer3.4.conv1.weight",
               "base_network.layer3.4.bn1.weight", "base_network.layer3.4.bn1.bias",
               "base_network.layer3.4.bn1.running_mean", "base_network.layer3.4.bn1.running_var",
               "base_network.layer3.4.conv2.weight", "base_network.layer3.4.bn2.weight",
               "base_network.layer3.4.bn2.bias", "base_network.layer3.4.bn2.running_mean",
               "base_network.layer3.4.bn2.running_var", "base_network.layer3.5.conv1.weight",
               "base_network.layer3.5.bn1.weight", "base_network.layer3.5.bn1.bias",
               "base_network.layer3.5.bn1.running_mean", "base_network.layer3.5.bn1.running_var",
               "base_network.layer3.5.conv2.weight", "base_network.layer3.5.bn2.weight",
               "base_network.layer3.5.bn2.bias", "base_network.layer3.5.bn2.running_mean",
               "base_network.layer3.5.bn2.running_var", "base_network.layer4.0.conv1.weight",
               "base_network.layer4.0.bn1.weight", "base_network.layer4.0.bn1.bias",
               "base_network.layer4.0.bn1.running_mean", "base_network.layer4.0.bn1.running_var",
               "base_network.layer4.0.conv2.weight", "base_network.layer4.0.bn2.weight",
               "base_network.layer4.0.bn2.bias", "base_network.layer4.0.bn2.running_mean",
               "base_network.layer4.0.bn2.running_var", "base_network.layer4.0.downsample.0.weight",
               "base_network.layer4.0.downsample.1.weight", "base_network.layer4.0.downsample.1.bias",
               "base_network.layer4.0.downsample.1.running_mean", "base_network.layer4.0.downsample.1.running_var",
               "base_network.layer4.1.conv1.weight", "base_network.layer4.1.bn1.weight",
               "base_network.layer4.1.bn1.bias", "base_network.layer4.1.bn1.running_mean",
               "base_network.layer4.1.bn1.running_var", "base_network.layer4.1.conv2.weight",
               "base_network.layer4.1.bn2.weight", "base_network.layer4.1.bn2.bias",
               "base_network.layer4.1.bn2.running_mean", "base_network.layer4.1.bn2.running_var",
               "base_network.layer4.2.conv1.weight", "base_network.layer4.2.bn1.weight",
               "base_network.layer4.2.bn1.bias", "base_network.layer4.2.bn1.running_mean",
               "base_network.layer4.2.bn1.running_var", "base_network.layer4.2.conv2.weight",
               "base_network.layer4.2.bn2.weight", "base_network.layer4.2.bn2.bias",
               "base_network.layer4.2.bn2.running_mean", "base_network.layer4.2.bn2.running_var",
               "classifier_layer.weight", "classifier_layer.bias" ]
    state_dict = model.state_dict().copy()
    # print(state_dict)
    delete = ["base_network.bn1.num_batches_tracked", "base_network.layer1.0.bn1.num_batches_tracked", "base_network.layer1.0.bn2.num_batches_tracked", "base_network.layer1.1.bn1.num_batches_tracked", "base_network.layer1.1.bn2.num_batches_tracked", "base_network.layer1.2.bn1.num_batches_tracked", "base_network.layer1.2.bn2.num_batches_tracked", "base_network.layer2.0.bn1.num_batches_tracked", "base_network.layer2.0.bn2.num_batches_tracked", "base_network.layer2.0.downsample.1.num_batches_tracked", "base_network.layer2.1.bn1.num_batches_tracked", "base_network.layer2.1.bn2.num_batches_tracked", "base_network.layer2.2.bn1.num_batches_tracked", "base_network.layer2.2.bn2.num_batches_tracked", "base_network.layer2.3.bn1.num_batches_tracked", "base_network.layer2.3.bn2.num_batches_tracked", "base_network.layer3.0.bn1.num_batches_tracked", "base_network.layer3.0.bn2.num_batches_tracked", "base_network.layer3.0.downsample.1.num_batches_tracked", "base_network.layer3.1.bn1.num_batches_tracked", "base_network.layer3.1.bn2.num_batches_tracked", "base_network.layer3.2.bn1.num_batches_tracked", "base_network.layer3.2.bn2.num_batches_tracked", "base_network.layer3.3.bn1.num_batches_tracked", "base_network.layer3.3.bn2.num_batches_tracked", "base_network.layer3.4.bn1.num_batches_tracked", "base_network.layer3.4.bn2.num_batches_tracked", "base_network.layer3.5.bn1.num_batches_tracked", "base_network.layer3.5.bn2.num_batches_tracked", "base_network.layer4.0.bn1.num_batches_tracked", "base_network.layer4.0.bn2.num_batches_tracked", "base_network.layer4.0.downsample.1.num_batches_tracked", "base_network.layer4.1.bn1.num_batches_tracked", "base_network.layer4.1.bn2.num_batches_tracked", "base_network.layer4.2.bn1.num_batches_tracked", "base_network.layer4.2.bn2.num_batches_tracked", "bottleneck_layer.0.weight", "bottleneck_layer.0.bias", "bottleneck_layer.1.weight", "bottleneck_layer.1.bias", "bottleneck_layer.1.running_mean", "bottleneck_layer.1.running_var", "bottleneck_layer.1.num_batches_tracked"]
    count = 0
    print(len(adjust))
    print(len(name))
    for i in delete:
        if i in state_dict:
            del state_dict[i]
        else:
            print(f"Key '{i}' not found in the state_dict.")
    for i in adjust:
        state_dict[name[count]] = state_dict.pop(i)
        count += 1
    print(count)
    torch.save(state_dict, './Output/resnet34_DA.pth')



def main(source, target,name, location,model):
    parser = get_parser(source, target,model)
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    optimizer = get_optimizer(model, args)

    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args, name, location)


if __name__ == "__main__":
    main('train', 'test', name='AtoW.csv', location='./TSNE/withoutDA.png', model='DAN/DAN.yaml')



