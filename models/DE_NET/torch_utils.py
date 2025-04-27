import torch.nn as nn
from copy import deepcopy
import torch
def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel,
                           nn.parallel.DistributedDataParallel)

def model_info(model, verbose=False):
    import thop

    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters()
              if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' %
              ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu',
               'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(
                      p.shape), p.mean(), p.std()))

    # try:  # FLOPS
    device = next(model.parameters()).device  # get model device
    flops = thop.profile(deepcopy(model.eval()),
                         inputs=(torch.zeros(1, 3, 640, 352).to(device), ),
                         verbose=False)[0] / 1E9 * 2
    fs = ', %.1f GFLOPS' % (flops)  # 640x352 FLOPS
    # except:
    #     fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' %
          (len(list(model.parameters())), n_p, n_g, fs))

