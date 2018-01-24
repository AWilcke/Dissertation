import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F

# WGAN stuff

def wgan_gradient_penalty(critic, real, fake):

    epsilon = torch.rand(real.shape[0], 1)
    # expand to size of real
    epsilon = epsilon.expand(real.shape[0], int(real.nelement()/real.shape[0])).contiguous().view(real.size())
    if isinstance(real, torch.cuda.FloatTensor):
        epsilon = epsilon.cuda()

    xhat = Variable(epsilon * real + (1-epsilon) * fake, requires_grad=True)

    critic_out = critic(xhat)

    ones = torch.ones(critic_out.size())
    if isinstance(real, torch.cuda.FloatTensor):
        ones = ones.cuda()

    grads = autograd.grad(outputs=critic_out, inputs=xhat, grad_outputs=ones,
            create_graph=True, only_inputs=True)[0]

    return (torch.norm(grads) - 1).pow(2)

def wgan_gen_loss(critic_out, *args, **kwargs):
    err_G = torch.mean(critic_out)
    return err_G

def wgan_critic_loss(critic, w1, fake_w1, one, mone, arg, *args, **kwargs):
    if not arg.gp:
        for p in critic.parameters():
            p.data.clamp_(-0.01, 0.01)

    # train with real (w1)
    errC_real = torch.mean(critic(w1))
    errC_real.backward(mone)

    # train with fake
    errC_fake = torch.mean(critic(fake_w1))
    errC_fake.backward(one)

    errC = errC_real - errC_fake
    return errC

# DRAGAN stuff

def dragan_gradient_penalty(critic, real, fake):

    delta = Variable(torch.normal(torch.zeros(real.size()), .1*torch.std(real)))
    if isinstance(real, torch.cuda.FloatTensor):
        delta = delta.cuda()
    x = Variable(real, requires_grad=True)
    xhat = x + delta

    critic_out = critic(xhat)

    ones = torch.ones(critic_out.size())
    if isinstance(real, torch.cuda.FloatTensor):
        ones = ones.cuda()

    grads = autograd.grad(outputs=critic_out, inputs=x, grad_outputs=ones,
            create_graph=True, only_inputs=True)[0]

    return (torch.norm(grads) - 1).pow(2)

def dragan_gen_loss(critic_out, labels_):
    labels_.data.fill(1.)
    return F.binary_cross_entropy(critic_out, labels_)

def dragan_critic_loss(critic, w1, fake_w1, one, mone, args, labels_):
    # train with real (w1)
    labels_.data.fill_(1.)
    errC_real = F.binary_cross_entropy(critic(w1), labels_)
    errC_real.backward()

    # train with fake
    labels_.data.fill_(0.)
    errC_fake = F.binary_cross_entropy(critic(fake_w1), labels_)
    errC_fake.backward()

    errC = errC_real + errC_fake
    return errC

# FisherGAN stuff

def fisher_critic_loss(critic, w1, fake_w1, one, mone, args, Lambda):
    vphi_fake = critic(fake_w1)
    vphi_real = critic(w1)
    # NOTE here f = <v,phi>   , but with modified f the below two lines are the
    # only ones that need change. E_P and E_Q refer to Expectation over real and fake.
    E_P_f,  E_Q_f  = vphi_real.mean(), vphi_fake.mean()
    E_P_f2, E_Q_f2 = (vphi_real**2).mean(), (vphi_fake**2).mean()
    constraint = (1 - (0.5*E_P_f2 + 0.5*E_Q_f2))
    # See Equation (9)
    obj_D = E_P_f - E_Q_f + Lambda * constraint - args.rho/2 * constraint**2
    # max_w min_Lambda obj_D. Compute negative gradients, apply updates with negative sign.
    obj_D.backward(mone)

    # artisanal sgd. We minimze Lambda so Lambda <- Lambda + lr * (-grad)
    Lambda.data += args.rho * Lambda.grad.data
    Lambda.grad.data.zero_()		

    return obj_D
