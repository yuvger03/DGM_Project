import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn.parameter import Parameter


def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def DCN(x):
    return x.data.cpu().numpy()


def batchedInv(batchedTensor):
    if np.prod(batchedTensor.shape[:-2]) >= 256 * 256 - 1:
        chunk_num = int(np.prod(batchedTensor.shape[1:-2]))
        if chunk_num >= (256 * 256 - 1):
            print("TOO BIG TENSOR")
        max_split = (256 * 256 - 1) // chunk_num
        temp = []
        for t in torch.split(batchedTensor, max_split):
            temp.append(torch.inverse(t))
        return torch.cat(temp)
    else:
        return torch.inverse(batchedTensor)


class BiGMM():
    def __init__(self):
        pass

    def __call__(self, y, mu, sig, corr, coef, loss_out=True, cv_out=False):
        y = y.unsqueeze(-2)
        corr = corr.squeeze(-1)
        # print(y.shape, mu.shape, sig.shape, corr.shape)
        cv = torch.stack((sig[..., 0] ** 2, sig[..., 0] * sig[..., 1] * corr,
                          sig[..., 0] * sig[..., 1] * corr, sig[..., 1] ** 2),
                         dim=-1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 2, 2)
        inv_cv = batchedInv(cv)
        if cv_out and not loss_out:
            return cv
        else:
            if torch.sum(torch.isnan(inv_cv)) > 0:
                print(sig[0, 0, :], corr[0, 0, :])

            xmu = (y - mu).unsqueeze(-1)
            nll = 0.5 * (torch.logdet(cv) + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(
                -1).squeeze(-1)).squeeze(-1)

        if loss_out and cv_out:
            return nll, cv
        else:
            return nll


class NLL():
    def __init__(self, mode='default'):
        self.mode = mode

    def __call__(self, y, mu, sig, loss_out=True, add_const=True):
        # print(mu.shape, y.shape, sig.shape)
        neg_log_p = ((mu - y) ** 2 / (2 * sig ** 2))
        if add_const:
            const = 0.5 * torch.log(2 * sig ** 2)
            neg_log_p = neg_log_p + const
        if self.mode == 'default':
            return (neg_log_p)
        elif self.mode == 'sum':
            return torch.sum(neg_log_p) / (y.size(0) * y.size(1))

class TriGMM():
    def __init__(self):
        pass
    def __call__(self, y, mu, sig, corr, coef, loss_out = True, cv_out = False):

        y = y.unsqueeze(-2)
        zeros = torch.zeros_like(sig[:,:,:,0])
        cv = torch.stack((sig[:, :, :, 0] ** 2, sig[:, :, :, 0] * sig[:, :, :, 1] * corr[:, :, :, 0], sig[:, :, :, 0] * sig[:, :, :, 2] * corr[:, :, :, 1],
                sig[:, :, :, 0] * sig[:, :, :, 1] * corr[:, :, :, 0], sig[:, :, :, 1] ** 2, sig[:, :, :, 1] * sig[:, :, :, 2] * corr[:, :, :, 2],
                sig[:, :, :, 0] * sig[:, :, :, 2] * corr[:, :, :, 1], sig[:, :, :, 1] * sig[:, :, :, 2] * corr[:, :, :, 2], sig[:, :, :, 2] ** 2), dim = -1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 3, 3)

        inv_cv = batchedInv(cv)
        #inv_cv = torch.tensor(np.linalg.inv(DCN(cv))).to(y.device)
        #inv_cv = torch.inverse(cv)
        if torch.sum(torch.isnan(inv_cv)) > 0:
            print('inv_cv')
            print(sig[0, 0, :], corr[0, 0, :])
        if cv_out and not loss_out:
            return cv
        else:
            xmu = (y - mu).unsqueeze(-1)
            nll = 0.5 * (torch.logdet(cv) + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)).squeeze(-1)
        if loss_out and cv_out:
            return nll, cv
        else:
            return nll


class QuadGMM():
    def __init__(self):
        pass

    def __call__(self, y, mu, sig, corr, coef, loss=True, cv=False):

        y = y.unsqueeze(-2)
        zeros = torch.zeros_like(sig[:, :, :, 0])
        L = torch.stack(
            (sig[:, :, :, 0], corr[:, :, :, 0], corr[:, :, :, 1], corr[:, :, :, 2],
             zeros, sig[:, :, :, 1], corr[:, :, :, 3], corr[:, :, :, 4],
             zeros, zeros, sig[:, :, :, 2], corr[:, :, :, 5],
             zeros, zeros, zeros, sig[:, :, :, 3]),
            dim=-1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 4, 4)
        inv_cv = torch.matmul(L.transpose(-1, -2), L)
        if cv and not loss:
            return batchedInv(inv_cv)
        else:
            log_det = -2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
            if torch.sum(torch.isnan(log_det)) > 0:
                print(sig[0, 0, :], corr[0, 0, :])
            # print(y.shape, mu.shape)
            xmu = (y - mu).unsqueeze(-1)

            terms = -0.5 * (log_det
                            + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)
                            + torch.log(torch.tensor(2 * np.pi)))

            nll = -torch.logsumexp(torch.log(coef) + terms, dim=-1)

        if loss and cv:
            return nll, batchedInv(inv_cv)
        else:
            return nll


class HexaGMM():
    def __init__(self):
        pass

    def __call__(self, y, mu, sig, corr, coef, loss=True, cv=False):
        y = y.unsqueeze(-2)
        zeros = torch.zeros_like(sig[:, :, :, 0])
        L = torch.stack(
            (sig[:, :, :, 0], corr[:, :, :, 0], corr[:, :, :, 1], corr[:, :, :, 2], corr[:, :, :, 3], corr[:, :, :, 4],
             zeros, sig[:, :, :, 1], corr[:, :, :, 5], corr[:, :, :, 6], corr[:, :, :, 7], corr[:, :, :, 8],
             zeros, zeros, sig[:, :, :, 2], corr[:, :, :, 9], corr[:, :, :, 10], corr[:, :, :, 11],
             zeros, zeros, zeros, sig[:, :, :, 3], corr[:, :, :, 12], corr[:, :, :, 13],
             zeros, zeros, zeros, zeros, sig[:, :, :, 4], corr[:, :, :, 14],
             zeros, zeros, zeros, zeros, zeros, sig[:, :, :, 5]),
            dim=-1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 6, 6)
        inv_cv = torch.matmul(L.transpose(-1, -2), L)
        if cv and not loss:
            return batchedInv(inv_cv)
        else:

            log_det = -2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
            if torch.sum(torch.isnan(log_det)) > 0:
                print(sig[0, 0, :], corr[0, 0, :])
            # print(y.shape, mu.shape)
            xmu = (y - mu).unsqueeze(-1)

            terms = -0.5 * (log_det
                            + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)
                            + torch.log(torch.tensor(2 * np.pi)))

            nll = -torch.logsumexp(torch.log(coef) + terms, dim=-1)

        if loss and cv:
            return nll, batchedInv(inv_cv)
        else:
            return nll
def gmm_criterion(D_s, mode='default'):
    criterion = None
    if D_s == 1:
        criterion = NLL(mode)
    elif D_s == 2:
        criterion = BiGMM()
    elif D_s == 3:
        criterion = TriGMM()
    elif D_s == 4:
        criterion = QuadGMM()
    elif D_s == 6:
        criterion = HexaGMM()
    else:
        print('NOT IMPLEMENTED : GMM')
    return criterion

class gmm_sample():
    def __init__(self, D_s, r=False):
        self.D_s = D_s
        self.r = r
    def __call__(self, mu, L):
        if self.D_s == 1:
            distrib = torch.distributions.Normal(mu.cpu(), L.cpu())
            if self.r:
                sampled_mu = distrib.rsample()
            else:
                sampled_mu = distrib.sample()
            return sampled_mu
        else:
            original_shape = mu.shape
            mu = mu.view(-1, self.D_s)
            L = L.view(-1, self.D_s, self.D_s)
            try:
                distrib = torch.distributions.MultivariateNormal(loc=mu.cpu(), covariance_matrix=L.cpu())
                if self.r:
                    sampled_mu = distrib.rsample()
                else:
                    sampled_mu = distrib.sample()
                return sampled_mu.view(original_shape)
            except Exception as e:
                print(e)
                return None


# Pytorch Activations
class SReLU_limited(nn.Module):
    """
    SReLU (S-shaped Rectified Linear Activation Unit): a combination of three linear functions, which perform mapping R → R with the following formulation:
    .. math::
        h(x_i) = \\left\\{\\begin{matrix} t_i^r + a_i^r(x_i - t_i^r), x_i \\geq t_i^r \\\\  x_i, t_i^r > x_i > t_i^l\\\\  t_i^l + a_i^l(x_i - t_i^l), x_i \\leq  t_i^l \\\\ \\end{matrix}\\right.
    with 4 trainable parameters.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        .. math:: \\{t_i^r, a_i^r, t_i^l, a_i^l\\}
    4 trainable parameters, which model an individual SReLU activation unit. The subscript i indicates that we allow SReLU to vary in different channels. Parameters can be initialized manually or randomly.
    References:
        - See SReLU paper:
        https://arxiv.org/pdf/1512.07030.pdf
    Examples:
        >>> srelu_activation = srelu((2,2))
        >>> t = torch.randn((2,2), dtype=torch.float, requires_grad = True)
        >>> output = srelu_activation(t)
    """

    def __init__(self, parameters=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - parameters: (tr, tl, ar, al) parameters for manual initialization, default value is None. If None is passed, parameters are initialized randomly.
        """
        super(SReLU_limited, self).__init__()

        if parameters is None:
            self.tr = Parameter(
                torch.tensor(2.0, dtype=torch.float, requires_grad=True)
            )
            self.tl = Parameter(
                torch.tensor(-2.0, dtype=torch.float, requires_grad=True)
            )

            self.ar = Parameter(
                torch.tensor(2.0, dtype=torch.float, requires_grad=True)
            )

        else:
            self.tr, self.tl, self.yr, self.yl = parameters

    def forward(self, x):
        """
        Forward pass of the function
        """
        return (
                (x >= self.tr).float() * self.ar
                + (x < self.tr).float() * (x > self.tl).float() * self.ar * (x - self.tl) / (self.tr - self.tl)
                + (x <= self.tl).float() * 0
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return F.sigmoid(y / tau)


class mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


def MLP_layers(cfg, nl_type, batch_norm=False, dropout=False, final_bias=True):
    layers = []
    nl_dict = {'RL': nn.ReLU(), 'TH': nn.Tanh(), 'LR': nn.LeakyReLU(0.2), 'EL': nn.ELU(), 'GL': nn.GELU(),
               'SL': nn.SELU(), 'MS': mish()}
    nl = nl_dict[nl_type]

    for i in range(1, len(cfg)):
        if i != len(cfg) - 1:
            layers += [('FC' + str(i) + '0', nn.Linear(cfg[i - 1], cfg[i], bias=True))]
            if batch_norm:
                layers += [('BN' + str(i) + '0', nn.BatchNorm1d(batch_norm))]
            if dropout:
                layers += [('DO' + str(i) + '0', nn.Dropout(dropout))]
            layers += [(nl_type + str(i) + '0', nl)]
        else:
            layers += [('FC' + str(i) + '0', nn.Linear(cfg[i - 1], cfg[i], bias=final_bias))]

    return nn.Sequential(OrderedDict(layers))


def Res_layers(cfg, nl_type, batch_norm=False, dropout=False):
    meta_layers = []
    nl_dict = {'RL': nn.ReLU(), 'TH': nn.Tanh(), 'LR': nn.LeakyReLU(0.2), 'EL': nn.ELU(), 'GL': nn.GELU(),
               'SL': nn.SELU(), 'MS': mish()}
    nl = nl_dict[nl_type]
    bias = True if len(cfg) > 2 else False

    for i in range(1, len(cfg)):
        layers = []
        for j in range(2):
            layers += [('FC' + str(i + 1) + str(j), nn.Linear(cfg[i], cfg[i], bias=bias))]
            if batch_norm:
                layers += [('BN' + str(i + 1) + str(j), nn.BatchNorm1d(batch_norm))]
            if dropout:
                layers += [('DO' + str(i + 1) + str(j), nn.Dropout(dropout))]
            if j == 0:
                layers += [(nl_type + str(i + 1) + '0', nl)]

        meta_layers.append(nn.Sequential(OrderedDict(layers)))

    return nn.Sequential(*meta_layers)


class Res_Block(nn.Module):
    def __init__(self, cfg, nl_type, batch_norm=False, dropout=False):
        super(Res_Block, self).__init__()

        nl_dict = {'RL': nn.ReLU(), 'TH': nn.Tanh(), 'LR': nn.LeakyReLU(0.2), 'GL': nn.GELU(), 'SL': nn.SELU(),
                   'MS': mish()}
        self.nl = nl_dict[nl_type]
        self.cfg = cfg

        self.FC1 = MLP_layers(cfg[:2], nl_type)
        self.RS = Res_layers(cfg[1:-1], nl_type, batch_norm, dropout)
        self.FC2 = MLP_layers(cfg[-2:], nl_type)

    def forward(self, x):
        x = self.FC1(x)
        for m in self.RS.children():
            x = self.nl(m(x) + x)
        x = self.FC2(x)
        return x


def cfg_Block(block_type, cfg, nl_type, batch_norm=False, dropout=False, final_bias=True):
    if block_type == 'mlp':
        block = MLP_Block(cfg, nl_type, batch_norm, dropout, final_bias=final_bias)
    elif block_type == 'res':
        block = Res_Block(cfg, nl_type, batch_norm, dropout)
    else:
        print("NOT IMPLEMENTED : cfg_Block")
    return block


class MLP_Block(nn.Module):
    def __init__(self, cfg, nl_type, batch_norm=False, dropout=False, final_bias=True):
        super(MLP_Block, self).__init__()
        self.FC = MLP_layers(cfg, nl_type, batch_norm, dropout, final_bias=final_bias)

    def forward(self, x):
        return self.FC(x)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data(batch_size=1, suffix=''):
    loc_train = np.load('data/loc_train' + suffix + '.npy')
    vel_train = np.load('data/vel_train' + suffix + '.npy')
    edges_train = np.load('data/edges_train' + suffix + '.npy')

    loc_valid = np.load('data/loc_valid' + suffix + '.npy')
    vel_valid = np.load('data/vel_valid' + suffix + '.npy')
    edges_valid = np.load('data/edges_valid' + suffix + '.npy')

    loc_test = np.load('data/loc_test' + suffix + '.npy')
    vel_test = np.load('data/vel_test' + suffix + '.npy')
    edges_test = np.load('data/edges_test' + suffix + '.npy')

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min


def load_kuramoto_data(batch_size=1, suffix=''):
    feat_train = np.load('data/feat_train' + suffix + '.npy')
    edges_train = np.load('data/edges_train' + suffix + '.npy')
    feat_valid = np.load('data/feat_valid' + suffix + '.npy')
    edges_valid = np.load('data/edges_valid' + suffix + '.npy')
    feat_test = np.load('data/feat_test' + suffix + '.npy')
    edges_test = np.load('data/edges_test' + suffix + '.npy')

    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]

    # Normalize each feature dim. individually
    feat_max = feat_train.max(0).max(0).max(0)
    feat_min = feat_train.min(0).min(0).min(0)

    feat_max = np.expand_dims(np.expand_dims(np.expand_dims(feat_max, 0), 0), 0)
    feat_min = np.expand_dims(np.expand_dims(np.expand_dims(feat_min, 0), 0), 0)

    # Normalize to [-1, 1]
    feat_train = (feat_train - feat_min) * 2 / (feat_max - feat_min) - 1
    feat_valid = (feat_valid - feat_min) * 2 / (feat_max - feat_min) - 1
    feat_test = (feat_test - feat_min) * 2 / (feat_max - feat_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_kuramoto_data_old(batch_size=1, suffix=''):
    feat_train = np.load('data/old_kuramoto/feat_train' + suffix + '.npy')
    edges_train = np.load('data/old_kuramoto/edges_train' + suffix + '.npy')
    feat_valid = np.load('data/old_kuramoto/feat_valid' + suffix + '.npy')
    edges_valid = np.load('data/old_kuramoto/edges_valid' + suffix + '.npy')
    feat_test = np.load('data/old_kuramoto/feat_test' + suffix + '.npy')
    edges_test = np.load('data/old_kuramoto/edges_test' + suffix + '.npy')

    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_motion_data(batch_size=1, suffix=''):
    feat_train = np.load('data/motion_train' + suffix + '.npy')
    feat_valid = np.load('data/motion_valid' + suffix + '.npy')
    feat_test = np.load('data/motion_test' + suffix + '.npy')
    adj = np.load('data/motion_adj' + suffix + '.npy')

    # NOTE: Already normalized

    # [num_samples, num_nodes, num_timesteps, num_dims]
    num_nodes = feat_train.shape[1]

    edges_train = np.repeat(np.expand_dims(adj.flatten(), 0),
                            feat_train.shape[0], axis=0)
    edges_valid = np.repeat(np.expand_dims(adj.flatten(), 0),
                            feat_valid.shape[0], axis=0)
    edges_test = np.repeat(np.expand_dims(adj.flatten(), 0),
                           feat_test.shape[0], axis=0)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(np.array(edges_train, dtype=np.int64))
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(np.array(edges_valid, dtype=np.int64))
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(np.array(edges_test, dtype=np.int64))

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
        [num_nodes, num_nodes])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def get_minimum_distance(data):
    data = data[:, :, :, :2].transpose(1, 2)
    data_norm = (data ** 2).sum(-1, keepdim=True)
    dist = data_norm + \
           data_norm.transpose(2, 3) - \
           2 * torch.matmul(data, data.transpose(2, 3))
    min_dist, _ = dist.min(1)
    return min_dist.view(min_dist.size(0), -1)


def get_buckets(dist, num_buckets):
    dist = dist.cpu().data.numpy()

    min_dist = np.min(dist)
    max_dist = np.max(dist)
    bucket_size = (max_dist - min_dist) / num_buckets
    thresholds = bucket_size * np.arange(num_buckets)

    bucket_idx = []
    for i in range(num_buckets):
        if i < num_buckets - 1:
            idx = np.where(np.all(np.vstack((dist > thresholds[i],
                                             dist <= thresholds[i + 1])), 0))[0]
        else:
            idx = np.where(dist > thresholds[i])[0]
        bucket_idx.append(idx)

    return bucket_idx, thresholds


def get_correct_per_bucket(bucket_idx, pred, target):
    pred = pred.cpu().numpy()[:, 0]
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def get_correct_per_bucket_(bucket_idx, pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))
