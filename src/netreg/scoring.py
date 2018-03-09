from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from utils import make_graph_image, dropout_train
from dataset import MNISTbyClass
import numpy as np

def validation_metrics(net, val_dataloader, writer, global_step):

    # save computation
    for p in net.parameters():
        p.requires_grad = False
    
    net.eval()
    net.apply(dropout_train)

    val_l2, val_hinge = 0, 0
    for val_sample in val_dataloader:

        w0_val = [Variable(x.float()) for x in val_sample['w0']]
        w1_val = [Variable(x.float()) for x in val_sample['w1']]

        train_val = []
        labels_val = []

        for t, l in val_sample['train']:
            train_val.append(Variable(t.float()))
            labels_val.append(Variable(l.float()))

        if torch.cuda.is_available():
            w0_val = [x.cuda() for x in w0_val]
            w1_val = [x.cuda() for x in w1_val]
            train_val = [t.cuda() for t in train_val]
            labels_val = [l.cuda() for l in labels_val]

        regressed_val = net(w0_val)

        val_l2_loss = net.l2_loss(regressed_val, w1_val)
        val_hinge_loss = net.perf_loss(regressed_val, train_val, labels_val)

        val_l2 += val_l2_loss.data[0]
        val_hinge += val_hinge_loss.data[0]

    writer.add_scalar('l2_loss/val', val_l2/len(val_dataloader),
            global_step)
    writer.add_scalar('hinge_loss/val', val_hinge/len(val_dataloader),
            global_step)


    # reset params
    for p in net.parameters():
        p.requires_grad = True
    net.train()

def check_performance(net, val_dataloader, writer, args, global_step):
    # save computation
    for p in net.parameters():
        p.requires_grad = False
    
    net.eval()
    net.apply(dropout_train)

    correct = defaultdict(int)
    total = defaultdict(int)

    for val_sample in val_dataloader:

        w0_val = [Variable(x.float()) for x in val_sample['w0']]

        if torch.cuda.is_available():
            w0_val = [x.cuda() for x in w0_val]

        regressed_val = net(w0_val)

        l = net.tensor_dict(regressed_val)


        for b in range(regressed_val[0].size(0)):
            mnist = MNISTbyClass(args.mnist, args.index, 
                    int(val_sample['label'][b]), 400, 
                    relevant_labels=args.val_labels, train_split=False,
                    extended=args.extended)
            loader = DataLoader(mnist, batch_size=200, num_workers=0)
            n = val_sample['train'][b][1].size(0) // 2


            for ipt, labels in loader:
                ipt = Variable(ipt, volatile=True)
                labels = labels.view(-1,1)

                if torch.cuda.is_available():
                    ipt = ipt.cuda()
                    labels = labels.cuda()

                out = net.fprop(l, ipt, b)
                pred = (out.data > 0.5).float()

                assert labels.size(0) == pred.size(0)

                correct[n] += (pred==labels.float()).sum()
                total[n] += labels.size(0)

    accuracies = [correct[n]/total[n] for num in sorted(correct.keys())]
    # print(accuracies)
    im = make_graph_image(np.arange(1,len(accuracies)+1), accuracies)
    latex = ''.join(['({:.0f},{:.4f})'.format(n+1, acc) for n, acc in enumerate(accuracies)])

    writer.add_image('classification', im, global_step)
    writer.add_text('latex', latex, global_step)

    # reset params
    for p in net.parameters():
        p.requires_grad = True
    net.train()
