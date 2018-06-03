from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
from dataset_loader import ImageDataset
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, TripletLoss, DeepSupervision, TripletLossDoneRight
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from optimizers import init_optim

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='sgd', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=256, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=8, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=32, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.5, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-05, type=float,
                    help="weight decay (default: 5e-05)")
parser.add_argument('--margin', type=float, default=0.1, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Hard Triplet Mining Options
parser.add_argument('--htmn', default=5000, type=int,
					help="Number of randomly selected example images for hard triplet mining")
parser.add_argument('--htmk', default=16, type=int,
					help="Number of iterations with same example image set")
parser.add_argument('--done-right', action='store_true', help="person re-id done right")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=16,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=128, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # These are for ImageNet Dataset
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # These are for ImageNet Dataset
    ])

    pin_memory = True if use_gpu else False

    if args.done_right:
        trainloader = DataLoader(
            ImageDataset(dataset.train, transform=transform_train),
            shuffle=True,
            batch_size=args.htmn, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )
    else:
        trainloader = DataLoader(
            ImageDataset(dataset.train, transform=transform_train),
            sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    if args.done_right:
        criterion_htri = TripletLossDoneRight(margin=args.margin, bs=args.train_batch)
    else:
        criterion_htri = TripletLoss(margin=args.margin)
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        if args.done_right:
            train_done_right(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)
        else:
            train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        if args.stepsize > 0: scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        outputs, features = model(imgs)
        if args.htri_only:
            if isinstance(features, tuple):
                loss = DeepSupervision(criterion_htri, features, pids)
            else:
                loss = criterion_htri(features, pids)
        else:
            if isinstance(outputs, tuple):
                xent_loss = DeepSupervision(criterion_xent, outputs, pids)
            else:
                xent_loss = criterion_xent(outputs, pids)

            if isinstance(features, tuple):
                htri_loss = DeepSupervision(criterion_htri, features, pids)
            else:
                htri_loss = criterion_htri(features, pids)

            loss = xent_loss + htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def train_done_right(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()

    n = args.htmn

    for _, (imgs, pids, _) in enumerate(trainloader):

        # No need for grad calculation in hard triplet mining steps
        with torch.no_grad():
            if use_gpu:
                imgs, pids = imgs.cuda(), pids.cuda()

            features = compute_features(model, imgs)
            #features = torch.rand(args.htmn, 2048).cuda()

            mask, dist = compute_mask_dist(features, pids)

        for batch_idx in range(args.htmk):
            # measure data loading time
            data_time.update(time.time() - end)

            # No need for grad calculation in hard triplet mining steps
            with torch.no_grad():
                # Randomly select batch number of query images
                batch_queries = torch.randperm(n)[:args.train_batch].cuda()
                #starttime = time.time()
                batch_positives, batch_negatives = sample_triplet(mask, dist, batch_queries)
                #endtime = time.time()
                #print(endtime - starttime)
                batch_query_imgs = torch.index_select(imgs, 0,  torch.cuda.LongTensor(batch_queries))
                batch_pos_imgs = torch.index_select(imgs, 0, batch_positives)
                batch_neg_imgs = torch.index_select(imgs, 0, batch_negatives)
                batch_imgs = torch.cat((batch_query_imgs, batch_pos_imgs, batch_neg_imgs), 0)

            _, batch_features = model(batch_imgs)

            loss = criterion_htri(batch_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), pids.size(0))

            if (batch_idx+1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch+1, batch_idx+1, args.htmk, batch_time=batch_time,
                       data_time=data_time, loss=losses))
        break


def compute_features(model, imgs):
    n = args.htmn
    chunks = n // args.train_batch
    features_list = []
    for c in range(chunks):
        img_chunk = imgs.narrow(0, c * args.train_batch, args.train_batch)
        _, feat_chunk = model(img_chunk)
        features_list.append(feat_chunk)
        #print('Feature calculation, chunk: {0}/{1}\t'.format(c, chunks))
    if chunks * args.train_batch < n:
        img_chunk = imgs.narrow(0, chunks * args.train_batch, n - chunks * args.train_batch)
        _, feat_chunk = model(img_chunk)
        features_list.append(feat_chunk)

    features = torch.cat(features_list)
    return features


def compute_mask_dist(features, pids):
    n = args.htmn
    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n).cuda()
    dist = dist + dist.t()
    dist.addmm_(1, -2, features, features.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    mask = pids.expand(n, n).eq(pids.expand(n, n).t()).cuda()
    return mask, dist


def sample_triplet(mask, dist, queries):
    n = mask.size(0)
    b = len(queries)
    maskp = mask[queries, :].type(torch.cuda.FloatTensor).unsqueeze(2).expand(b, n, n)
    maskn = (1 - mask[queries, :].type(torch.cuda.FloatTensor)).unsqueeze(1).expand(b, n, n)
    maskLoss = maskp * maskn
    ap = dist[queries, :].unsqueeze(2).expand(b, n, n)
    an = dist[queries, :].unsqueeze(1).expand(b, n, n)
    loss_like = (ap - an)
    loss_like = loss_like * maskLoss

    """
    loss_like2 = torch.zeros([b, n, n])
    for anci, anc in enumerate(queries):
        for pos in range(n):
            if mask[anc][pos] == 1 and anc != pos:
                for neg in range(n):
                    if mask[anc, neg] == 0:
                        loss_like2[anci, pos, neg] = dist[anc][pos] - dist[anc][neg]

    """

    # Determine 25 triplets with highest loss for each query
    vals, inds = torch.topk(loss_like.view(b, n * n), 25)
    pos_inds, neg_inds = np.unravel_index(inds.view(b*25), (n, n))
    pos_inds = torch.reshape(torch.tensor(pos_inds), (b, 25))
    neg_inds = torch.reshape(torch.tensor(neg_inds), (b, 25))

    # Randomly pick one triplet among 25 triplets for each query
    selected_idx = torch.Tensor(np.random.choice(range(25), size=b)).long()
    pos_inds = pos_inds.gather(1, selected_idx.view(-1, 1)).cuda().squeeze()
    neg_inds = neg_inds.gather(1, selected_idx.view(-1, 1)).cuda().squeeze()

    return pos_inds, neg_inds


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]

if __name__ == '__main__':
    main()