from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from utils import make_graph_image, dropout_train
from dataset import MNISTbyClass

def validation_metrics(net, val_dataloader, writer, global_step):

    # save computation
    for p in net.parameters():
        p.requires_grad = False
    
    net.eval()
    net.apply(dropout_train)

    val_l2, val_hinge = 0, 0
    for val_sample in val_dataloader:

        w0_val = [Variable(x) for x in val_sample['w0'].float()]
        w1_val = [Variable(x) for x in val_sample['w1'].float()]

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

    for val_sample in val_dataloader:

        w0_val = [Variable(x) for x in val_sample['w0'].float()]

        if torch.cuda.is_available():
            w0_val = [x.cuda() for x in w0_val]

        regressed_val = net(w0_val)

        l = net.tensor_dict(regressed_val)

        correct = defaultdict(int)
        total = defaultdict(int)

        for b in range(regressed_val[0].size(0)):
            mnist = MNISTbyClass(args.mnist, args.index, int(val_sample['label'][b]), train_labels=False, train_split=False)
            loader = DataLoader(mnist, batch_size=256, num_workers=0)
            n = val_sample['train'][b][1].size(0)

            for ipt, labels in loader:
                out = net.fprop(l, ipt, b)
                pred = (out.data > 0.5).float()
                correct[n] += (pred==labels.data).sum()
                total[n] += labels.size(0)

    accuracies = [correct[n]/total[n] for num in sorted(correct.keys())]
    im = make_graph_image(np.arange(1,len(accuracies)+1), accuracies)
    latex = ''.join(['({:.0f},{:.4f})'.format(n+1, acc) for n, acc in enumerate(accuracies)])

    writer.add_image('classification', im, global_step)
    writer.add_text('latex', latex, global_step)

    # reset params
    for p in net.parameters():
        p.requires_grad = True
    net.train()
