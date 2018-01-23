import torch
from torch.autograd import Variable
import torch.autograd as autograd

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

def wgan_gen_loss(critic_out):
    err_G = torch.mean(critic_out)
    return err_G

def wgan_critic_loss(critic, w1, fake_w1, one, mone, gp):
    if not gp:
        for p in critic.parameters():
            p.data.clamp_(-0.01, 0.01)

    # train with real (w1)
    errC_real = torch.mean(critic(w1))
    errC_real.backward(mone)

    # train with fake
    errC_fake = torch.mean(critic(fake_w1))
    errC_fake.backward(one)

    return errC_real, errC_fake
