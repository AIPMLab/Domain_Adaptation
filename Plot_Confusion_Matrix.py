import configargparse
import data_loader
import os
import models
import utils
from utils import str2bool
import numpy as np
import random
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def get_parser(source,target,backbone):
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help='DSAN/DSAN.yaml',default="DSAN/DSAN.yaml")
    add = parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    # network related
    parser.add_argument('--backbone', type=str, default=backbone)
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, default='./ChestXray')
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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

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
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay,momentum=args.momentum,nesterov=False
                                )
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
        -args.lr_decay))
    return scheduler

count = 1

def test(model, target_test_loader, args, count,location):
    model.eval()
    test_loss = utils.AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    acc = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / len_target_dataset


    if count == args.n_epoch:
        conf_mat = confusion_matrix(all_targets, all_preds)
        num_classes = len(target_test_loader.dataset.classes)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(conf_mat, cmap='Blues')


        cbar = ax.figure.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(target_test_loader.dataset.classes, rotation=90, ha='right', fontsize=18)
        ax.set_yticklabels(target_test_loader.dataset.classes, fontsize=18)
        ax.tick_params(axis='both', which='both', length=0, pad=2)
        ax.grid(visible=False)

        thresh = conf_mat.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, format(conf_mat[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_mat[i, j] > thresh else "black", fontsize=18)

        plt.tight_layout()

        # plt.show()
        plt.savefig(location)
    return acc, test_loss.avg




def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args,location):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []
    count = 1
    for e in range(1, args.n_epoch + 1):
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())


        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
            e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        
        stop += 1
        test_acc, test_loss = test(model, target_test_loader, args, count=count,location=location)
        log.append([test_acc.item()])
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
        count += 1
    print('Transfer result: {:.4f}'.format(best_acc))

def main(source,target,backbone,location):
    parser = get_parser(source,target,backbone)
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
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args,location=location)


if __name__ == "__main__":
    source = ['train']
    target = ['test']
    # backbone = ['alexnet','resnet34','resnet50','densenet']
    backbone = ['resnet50','resnet101','densenet']
    for b in backbone:
        for s in source:
            for t in target:
                if not os.path.exists('./Confusion_Matrix/Chest/'+'DSAN/'+str(b)+'/'):
                    os.makedirs('./Confusion_Matrix/Chest/'+'DSAN/'+str(b)+'/')
                main(s,t,b,'./Confusion_Matrix/Chest/'+'DSAN/'+str(b)+'/'+str(s)+'to'+str(t)+'.png')
                # main(t,s,b, './Confusion_Matrix/' + 'DSAN/' + str(b) + '/' + str(t) + 'to' + str(s) + '.png')
