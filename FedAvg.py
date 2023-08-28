import copy
import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random


def get_parser(name, technique):
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="DAN/DAN.yaml",default=technique)
    parser.add("--seed", type=int,default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    # network related
    parser.add_argument('--backbone', type=str, default='resnet34')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, default='./office31')
    parser.add_argument('--src_domain', type=str, default=name[0])
    parser.add_argument('--tgt_domain', type=str, default=name[1])

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
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=False)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
        -args.lr_decay))
    return scheduler


m = ['1','2','3'] #clients
# client = ['amazon','webcam'] #domain
target = ['amazon', 'target']
client = [['client1','webcam'],['client2','webcam'],['client3','webcam']]
technique = ['DSAN/DSAN.yaml','DeepCoral/DeepCoral.yaml','BNM/BNM.yaml'] # algorithm

loc = []
test_a = []
parser1 = get_parser(target, technique[0])
args1 = parser1.parse_args()
setattr(args1, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# print(args1)
print(f'Global model setted up, {args1}')
set_random_seed(args1.seed)
source_loader1, target_train_loader1, target_test_loader1, n_class = load_data(args1)
setattr(args1, "n_class", n_class)
if args1.epoch_based_training:
    setattr(args1, "max_iter", args1.n_epoch * min(len(source_loader1), len(target_train_loader1)))
else:
    setattr(args1, "max_iter", args1.n_epoch * args1.n_iter_per_epoch)
glo_model = get_model(args1)
optimizer1 = get_optimizer(glo_model, args1)
if args1.lr_scheduler:
    scheduler1 = get_scheduler(optimizer1, args1)
else:
    scheduler = None


def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return test_loss.avg, acc


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []
    for e in range(1, args.n_epoch + 1):
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)

        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        for _ in range(n_batch):
            data_source, label_source = next(iter_source)  # .next()
            data_target, _ = next(iter_target)  # .next()
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

        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])

        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
            e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Test
        stop += 1

        test_loss, test_acc = test(model, target_test_loader1, args)
        test_a.append(test_acc)
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        print(info)
    return model.state_dict()


def LocalTraining(loc):
    for i in range(0, len(m)):
        parser = get_parser(client[i], technique[i])
        args = parser.parse_args()
        setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f'Local model setted up, {args}')
        set_random_seed(args.seed)
        source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
        setattr(args, "n_class", n_class)
        if args.epoch_based_training:
            setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
        else:
            setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
        model = get_model(args)
        model.load_state_dict(copy.deepcopy(loc[i]))
        optimizer = get_optimizer(model, args)
        if args.lr_scheduler:
            scheduler = get_scheduler(optimizer, args)
        else:
            scheduler = None
        loc[i] = (train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args))


def Aggregation():
    weights = [0, 0, 0]  # weights
    ans = 0
    for i in test_a:
        ans += i
    for i in range(0, 3):
        weights[i] = test_a[i] / ans
    test_a.clear()
    averaged_state_dict = {}
    for model, weight in zip(loc, weights):
        for key in model.keys():
            if key in averaged_state_dict:
                averaged_state_dict[key] += weight * model[key]
            else:
                averaged_state_dict[key] = weight * model[key]
    glo_model.load_state_dict(copy.deepcopy(averaged_state_dict))


def main():
    for i in range(0, 3):
        loc.append(glo_model.state_dict())
    for k in range(0, 3): #communication rounds
        LocalTraining(loc)
        Aggregation()
        for i in range(0, 3):
            loc[i] = (glo_model.state_dict())
        test_loss, test_acc = test(glo_model, target_test_loader1, args1)
        print('test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc))


if __name__ == "__main__":
    main()
