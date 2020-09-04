import argparse
import os
import time

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from model.dataset import ShadowPairedDataset
from model.model import TeachingTeleModel, NaiveTeleModel, NewTeachingTeleModel

parser = argparse.ArgumentParser(description='deepShadowTeleop')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--load-model', type=str, default='new_early_teach_teleop.model')
parser.add_argument('--load-epoch', type=int, default=-1)
parser.add_argument('--model-path', type=str, default='./assets/learned_models',
                   help='pre-trained model path')
parser.add_argument('--data-path', type=str, default='../shadow_translation_teleop/data/human_shadow_com_nobase', help='data path')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=5)

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.cuda:
    torch.cuda.manual_seed(1)

logger = SummaryWriter(os.path.join('./assets/log/', args.tag))
np.random.seed(int(time.time()))

def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2**31-1))

def my_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

input_viewpoint=[0,1,2,3,4,5,6,7,8]
input_size=100
embedding_size=128
joint_size=22
thresh_acc=[0.2, 0.25, 0.3]
joint_upper_range = torch.tensor([0.349, 1.571, 1.571, 1.571, 0.785, 0.349, 1.571, 1.571,
                                  1.571, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571,
                                  1.571, 1.047, 1.222, 0.209, 0.524, 1.571])
joint_lower_range = torch.tensor([-0.349, 0, 0, 0, 0, -0.349, 0, 0, 0, -0.349, 0, 0, 0,
                                  -0.349, 0, 0, 0, -1.047, 0, -0.209, -0.524, 0])

train_loader = torch.utils.data.DataLoader(
    ShadowPairedDataset(
        path=args.data_path,
        input_size=input_size,
        input_viewpoint=input_viewpoint,
        is_train=True,
    ),
    batch_size=args.batch_size,
    num_workers=32,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
)

test_loader = torch.utils.data.DataLoader(
    ShadowPairedDataset(
        path=args.data_path,
        input_size=input_size,
        input_viewpoint=input_viewpoint,
        is_train=False,
        with_name=True,
    ),
    batch_size=args.batch_size,
    num_workers=32,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
)

is_resume = 0
if args.load_model and args.load_epoch != -1:
    is_resume = 1

if is_resume or args.mode == 'test':
    model = torch.load(args.load_model, map_location='cuda:{}'.format(args.gpu))
    model.device_ids = [args.gpu]
    print('load model {}'.format(args.load_model))
else:
    model = NewTeachingTeleModel(input_size=input_size, embedding_size=embedding_size, joint_size=joint_size)
    # model = TeachingRENTeleModel(input_size=input_size, embedding_size=embedding_size, joint_size=joint_size)

if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [1,2]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
    joint_upper_range = joint_upper_range.cuda()
    joint_lower_range = joint_lower_range.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=80, gamma=0.5)

def train(model, loader, epoch):
    scheduler.step()
    model.train()
    torch.set_grad_enabled(True)
    train_error_shadow = 0
    train_error_human = 0
    correct_shadow, correct_human = [0,0,0], [0,0,0]
    for batch_idx, (shadow, human, target) in enumerate(loader):
        if args.cuda:
            shadow, human, target = shadow.cuda(), human.cuda(), target.cuda()

        # shadow part
        optimizer.zero_grad()
        embedding_shadow, joint_shadow = model(shadow, is_human=False)
        #TODO: check if we need to reshape joint_upper_ranges, joint_lower_ranges
        joint_shadow = joint_shadow * (joint_upper_range - joint_lower_range) + joint_lower_range
        loss_shadow_reg = F.mse_loss(joint_shadow, target)
        loss_shadow_cons = constraints_loss(joint_shadow)/target.shape[0]
        loss_shadow = loss_shadow_reg + loss_shadow_cons

        # human part
        embedding_human, joint_human = model(human, is_human=True)
        joint_human = joint_human * (joint_upper_range - joint_lower_range) + joint_lower_range
        loss_human_reg = F.mse_loss(joint_human, target)
        loss_align = F.mse_loss(embedding_human, embedding_shadow.detach())
        loss_human_cons = constraints_loss(joint_human)/target.shape[0]
        loss_human = loss_human_reg + loss_align + loss_human_cons
        loss = loss_shadow + loss_human
        loss.backward()
        optimizer.step()

        loss = loss_shadow + loss_human

        # compute acc
        res_shadow = [np.sum(np.sum(abs(joint_shadow.cpu().data.numpy() - target.cpu().data.numpy()) < thresh,
                             axis=-1) == joint_size) for thresh in thresh_acc]
        res_human = [np.sum(np.sum(abs(joint_human.cpu().data.numpy() - target.cpu().data.numpy()) < thresh,
                            axis=-1) == joint_size) for thresh in thresh_acc]
        correct_shadow = [c + r for c, r in zip(correct_shadow, res_shadow)]
        correct_human = [c + r for c, r in zip(correct_human, res_human)]

        # compute average angle error
        train_error_shadow += F.l1_loss(joint_shadow, target, size_average=False)/joint_size
        train_error_human += F.l1_loss(joint_human, target, size_average=False)/joint_size

        if batch_idx % args.log_interval == 0:
            if isinstance(loss_shadow_cons, float):
                loss_shadow_cons = torch.zeros(1)
            if isinstance(loss_human_cons, float):
                loss_human_cons = torch.zeros(1)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss_reg_shadow: {:.6f}\t'
                    'Loss_cons_shadow: {:.6f}\tLoss_reg_human: {:.6f}\t'
                    'Loss_cons_human: {:.6f}\tLoss_align: {:.6f}\t{}'.format(
                    epoch, batch_idx * args.batch_size, len(loader.dataset),
                    100. * batch_idx * args.batch_size / len(loader.dataset),
                    loss.item(), loss_shadow_reg.item(), loss_shadow_cons.item(),
                    loss_human_reg.item(), loss_human_cons.item(), loss_align.item(), args.tag))

            logger.add_scalar('train_loss', loss.item(),
                    batch_idx + epoch * len(loader))
            logger.add_scalar('train_loss_shadow_reg', loss_shadow_reg.item(),
                    batch_idx + epoch * len(loader))
            logger.add_scalar('train_loss_shadow_cons', loss_shadow_cons.item(),
                    batch_idx + epoch * len(loader))
            logger.add_scalar('train_loss_human_reg', loss_human_reg.item(),
                    batch_idx + epoch * len(loader))
            logger.add_scalar('train_loss_human_cons', loss_human_cons.item(),
                    batch_idx + epoch * len(loader))
            logger.add_scalar('train_loss_align', loss_align.item(),
                    batch_idx + epoch * len(loader))

    train_error_shadow /= len(loader.dataset)
    train_error_human /= len(loader.dataset)
    acc_shadow = [float(c) / float(len(loader.dataset)) for c in correct_shadow]
    acc_human = [float(c) / float(len(loader.dataset)) for c in correct_human]

    return acc_shadow, acc_human, train_error_shadow, train_error_human


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss_shadow_reg = 0
    test_loss_shadow_cons = 0
    test_loss_human_reg = 0
    test_loss_human_cons = 0
    test_loss_align = 0
    test_error_shadow = 0
    test_error_human = 0
    res = []
    correct_shadow, correct_human = [0,0,0], [0,0,0]
    for shadow, human, target, name in loader:
        if args.cuda:
            shadow, human, target = shadow.cuda(), human.cuda(), target.cuda()
        
        # shadow part
        embedding_shadow, joint_shadow = model(shadow, is_human=False)
        joint_shadow = joint_shadow * (joint_upper_range - joint_lower_range) + joint_lower_range
        test_loss_shadow_reg += F.mse_loss(joint_shadow, target, size_average=False).item()
        cons = constraints_loss(joint_shadow)
        if not isinstance(cons, float):
            test_loss_shadow_cons += cons 

        # human part
        embedding_human, joint_human = model(human, is_human=True)
        joint_human = joint_human * (joint_upper_range - joint_lower_range) + joint_lower_range
        test_loss_human_reg += F.mse_loss(joint_human, target, size_average=False).item()
        test_loss_align += F.mse_loss(embedding_human, embedding_shadow.detach(), size_average=False).item()
        cons = constraints_loss(joint_human)
        if not isinstance(cons, float):
            test_loss_human_cons += cons 

        # compute acc
        res_shadow = [np.sum(np.sum(abs(joint_shadow.cpu().data.numpy() - target.cpu().data.numpy()) < thresh,
                      axis=-1) == joint_size) for thresh in thresh_acc]
        res_human = [np.sum(np.sum(abs(joint_human.cpu().data.numpy() - target.cpu().data.numpy()) < thresh,
                      axis=-1) == joint_size) for thresh in thresh_acc]
        correct_shadow = [c + r for c, r in zip(correct_shadow, res_shadow)]
        correct_human = [c + r for c, r in zip(correct_human, res_human)]

        # compute average angle error
        test_error_shadow += F.l1_loss(joint_shadow, target, size_average=False)/joint_size
        test_error_human += F.l1_loss(joint_human, target, size_average=False)/joint_size
        res.append((name, joint_human))
        # res.append(joint_shadow - target)

    test_loss_shadow_reg /= len(loader.dataset)
    test_loss_shadow_cons /= len(loader.dataset)
    test_loss_human_reg /= len(loader.dataset)
    test_loss_align /= len(loader.dataset)
    test_loss_human_cons /= len(loader.dataset)
    test_loss = test_loss_shadow_reg + test_loss_human_reg + test_loss_align + test_loss_shadow_cons +\
                test_loss_human_cons
    test_error_shadow /= len(loader.dataset)
    test_error_human /= len(loader.dataset)

    acc_shadow = [float(c)/float(len(loader.dataset)) for c in correct_shadow]
    acc_human = [float(c)/float(len(loader.dataset)) for c in correct_human]

    # f = open('input.csv', 'w')
    # for batch in res:
    #     for name, joint in zip(batch[0], batch[1]):
    #         buf = [name, '0.0', '0.0'] + [str(i) for i in joint.cpu().data.numpy()]

    #     for error in batch:
    #         buf = [str(i) for i in error.cpu().data.numpy()]
    #         f.write(','.join(buf) + '\n')

    return acc_shadow, acc_human, test_error_shadow, test_error_human, test_loss, test_loss_shadow_reg,\
           test_loss_shadow_cons, test_loss_human_reg, test_loss_human_cons, test_loss_align

def constraints_loss(joint_angle):
    F4 = [joint_angle[:, 0], joint_angle[:, 5], joint_angle[:, 9], joint_angle[:, 13]]
    F1_3 = [joint_angle[:, 1], joint_angle[:, 6], joint_angle[:, 10], joint_angle[:, 14],
            joint_angle[:, 2], joint_angle[:, 7], joint_angle[:, 11], joint_angle[:, 15],
            joint_angle[:, 3], joint_angle[:, 8], joint_angle[:, 12], joint_angle[:, 16],
            joint_angle[:, 21]]
    loss_cons = 0.0

    for pos in F1_3:
        for f in pos:
            loss_cons = loss_cons + max(0 - f, 0) + max(f - 1.57, 0)
    for pos in F4:
        for f in pos:
            loss_cons = loss_cons + max(-0.349 - f, 0) + max(f - 0.349, 0)
    for f in joint_angle[:, 4]:
        loss_cons = loss_cons + max(0 - f, 0) + max(f - 0.785, 0)
    for f in joint_angle[:, 17]:
        loss_cons = loss_cons + max(-1.047 - f, 0) + max(f - 1.047, 0)
    for f in joint_angle[:, 18]:
        loss_cons = loss_cons + max(0 - f, 0) + max(f - 1.222, 0)
    for f in joint_angle[:, 19]:
        loss_cons = loss_cons + max(-0.209 - f, 0) + max(f - 0.209, 0)
    for f in joint_angle[:, 20]:
        loss_cons = loss_cons + max(-0.524 - f, 0) + max(f - 0.524, 0)
    return loss_cons


def main():
    if args.mode == 'train':
        for epoch in range(is_resume*args.load_epoch, args.epoch):
            acc_train_shadow, acc_train_human, train_error_shadow, train_error_human = train(model, train_loader, epoch)
            print('Train done, acc_shdow={}, acc_human={}, train_error_shadow={}, train_error_human={}'.format(acc_train_shadow, acc_train_human, train_error_shadow, train_error_human))
            acc_test_shadow, acc_test_human, test_error_shadow, test_error_human, loss, loss_shadow_reg, loss_shadow_cons, loss_human_reg,\
                    loss_human_cons, loss_align = test(model, test_loader)
            print('Test done, acc_shadow={}, acc_human={}, error_shadow ={}, error_human ={}, loss={}, loss_shadow_reg={}, loss_shadow_cons={}, '\
                  'loss_human_reg={}, loss_human_cons={}, loss_align={}'.format(acc_test_shadow, acc_test_human,
                                                                                 test_error_shadow, test_error_human,
                                                                                 loss, loss_shadow_reg, 
                                                                                 loss_shadow_cons, loss_human_reg, 
                                                                                 loss_human_cons, loss_align))
            logger.add_scalar('train_acc_shadow0.2', acc_train_shadow[0], epoch)
            logger.add_scalar('train_acc_shadow0.25', acc_train_shadow[1], epoch)
            logger.add_scalar('train_acc_shadow0.3', acc_train_shadow[2], epoch)
            logger.add_scalar('train_acc_human0.2', acc_train_human[0], epoch)
            logger.add_scalar('train_acc_human0.25', acc_train_human[1], epoch)
            logger.add_scalar('train_acc_human0.3', acc_train_human[2], epoch)

            logger.add_scalar('test_acc_shadow0.2', acc_test_shadow[0], epoch)
            logger.add_scalar('test_acc_shadow0.25', acc_test_shadow[1], epoch)
            logger.add_scalar('test_acc_shadow0.3', acc_test_shadow[2], epoch)
            logger.add_scalar('test_acc_human0.2', acc_test_human[0], epoch)
            logger.add_scalar('test_acc_human0.25', acc_test_human[1], epoch)
            logger.add_scalar('test_acc_human0.3', acc_test_human[2], epoch)

            logger.add_scalar('test_error_shadow', test_error_shadow, epoch)
            logger.add_scalar('test_error_human', test_error_human, epoch)

            logger.add_scalar('test_loss', loss, epoch)
            logger.add_scalar('test_loss_shadow_reg', loss_shadow_reg, epoch)
            logger.add_scalar('test_loss_shadow_cons', loss_shadow_cons, epoch)
            logger.add_scalar('test_loss_human_reg', loss_human_reg, epoch)
            logger.add_scalar('test_loss_align', loss_align, epoch)
            logger.add_scalar('test_loss_human_cons', loss_human_cons, epoch)

            if epoch % args.save_interval == 0:
                path = os.path.join(args.model_path, args.tag + '_{}.model'.format(epoch))
                torch.save(model, path)
                print('Save model @ {}'.format(path))
    else:
        print('testing...')
        acc_test_shadow, acc_test_human, test_error_shadow, test_error_human, loss, loss_shadow_reg, loss_shadow_cons, loss_human_reg, \
                loss_human_cons, loss_align = test(model, test_loader)
        print('Test done, acc_shadow={}, acc_human={}, error_shadow ={}, error_human ={}, loss={}, loss_shadow_reg={}, loss_shadow_cons={}, ' \
              'loss_human_reg={}, loss_human_cons={}, loss_align={},'.format(acc_test_shadow, acc_test_human,
                                                                           test_error_shadow, test_error_human,
                                                                           loss, loss_shadow_reg,
                                                                           loss_shadow_cons, loss_human_reg,
                                                                           loss_human_cons, loss_align))
if __name__ == "__main__":
    main()
