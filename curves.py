import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
from torch.utils.data import Dataset
from scipy.special import binom
from tqdm import tqdm

def train_curve(
    models: list,
    trainloader: Dataset,
    curve_class: nn.Module,
    input_size: int,
    hidden_layers: list,
    curve: nn.Module,
    fix_start: bool,
    fix_end: bool,
    optim: str,
    lr: float,
    epochs: int,
    disable_tqdm: bool = True,
):
    # Create model (with initial parameters)
    model = CurveNet(
        curve=curve,  # e.g. polychain
        architecture=curve_class,  # e.g. TabularModelCurve
        num_bends=len(models),
        input_size=input_size,
        hidden_layers=hidden_layers,
        fix_start=fix_start,
        fix_end=fix_end,
    )
    # Load initial parameters
    for i, base_model in enumerate(models):
        if base_model is not None:
            model.import_base_parameters(base_model, i)

    # r, t = 0, 0
    # for param in list(model.parameters()):
    #     if param.requires_grad:
    #         r += 1
    #     t += 1
    # print('Number of trainable parameters: %d/%d' % (r, t))

    criterion = nn.CrossEntropyLoss()
    if optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=lr,
        )
    elif optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=lr,
        )
    else:
        raise ValueError('Invalid optim: %s' % optim)

    model.train()
    for epoch in tqdm(range(epochs), disable=disable_tqdm):
        for iter, (input, target) in enumerate(trainloader):
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return model


class Bezier(Module):
    def __init__(self, num_bends):
        super(Bezier, self).__init__()
        self.register_buffer(
            'binom',
            torch.Tensor(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32))
        )
        self.register_buffer('range', torch.arange(0, float(num_bends)))
        self.register_buffer('rev_range', torch.arange(float(num_bends - 1), -1, -1))

    def forward(self, t):
        return self.binom * \
               torch.pow(t, self.range) * \
               torch.pow((1.0 - t), self.rev_range)


class PolyChain(Module):
    def __init__(self, num_bends):
        super(PolyChain, self).__init__()
        self.num_bends = num_bends
        self.register_buffer('range', torch.arange(0, float(num_bends)))

    def forward(self, t):
        t_n = t * (self.num_bends - 1)
        return torch.max(self.range.new([0.0]), 1.0 - torch.abs(t_n - self.range))


class CurveModule(Module):

    def __init__(self, fix_points, parameter_names=()):
        super(CurveModule, self).__init__()
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_names = parameter_names
        self.l2 = 0.0

    def compute_weights_t(self, coeffs_t):
        w_t = [None] * len(self.parameter_names)
        self.l2 = 0.0
        for i, parameter_name in enumerate(self.parameter_names):
            for j, coeff in enumerate(coeffs_t):
                parameter = getattr(self, '%s_%d' % (parameter_name, j))
                if parameter is not None:
                    if w_t[i] is None:
                        w_t[i] = parameter * coeff
                    else:
                        w_t[i] += parameter * coeff
            if w_t[i] is not None:
                self.l2 += torch.sum(w_t[i] ** 2)
        return w_t


class Linear(CurveModule):

    def __init__(self, in_features, out_features, fix_points, bias=True):
        super(Linear, self).__init__(fix_points, ('weight', 'bias'))
        self.in_features = in_features
        self.out_features = out_features

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter(torch.Tensor(out_features, in_features), requires_grad=not fixed)
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(out_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.linear(input, weight_t, bias_t)


class Conv2d(CurveModule):

    def __init__(self, in_channels, out_channels, kernel_size, fix_points, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(fix_points, ('weight', 'bias'))
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter(
                    torch.Tensor(out_channels, in_channels // groups, *kernel_size),
                    requires_grad=not fixed
                )
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(out_channels), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.conv2d(input, weight_t, bias_t, self.stride,
                        self.padding, self.dilation, self.groups)


class _BatchNorm(CurveModule):
    _version = 2

    def __init__(self, num_features, fix_points, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(fix_points, ('weight', 'bias'))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'weight_%d' % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('weight_%d' % i, None)
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for i in range(self.num_bends):
                getattr(self, 'weight_%d' % i).data.uniform_()
                getattr(self, 'bias_%d' % i).data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, coeffs_t):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.batch_norm(
            input, self.running_mean, self.running_var, weight_t, bias_t,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class BatchNorm2d(_BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class CurveNet(Module):
    def __init__(self, curve, architecture, num_bends,
                 input_size, hidden_layers, fix_start=True, fix_end=True):
        """
        Tabular CurveNet: Not yet implemented for FMNIST
        """
        super(CurveNet, self).__init__()
        self.num_bends = num_bends
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]
        
        self.curve = curve  # e.g. PolyChain
        self.architecture = architecture  # TabularModelCurve

        self.l2 = 0.0
        self.coeff_layer = self.curve(self.num_bends)
        self.net = self.architecture(input_size=input_size,
                                     hidden_layers=hidden_layers,
                                     fix_points=self.fix_points)
        self.curve_modules = []
        for module in self.net.modules():
            if issubclass(module.__class__, CurveModule):
                self.curve_modules.append(module)

    def import_base_parameters(self, base_model, index):
        parameters = list(self.net.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            parameter.data.copy_(base_parameter.data)

    def import_base_buffers(self, base_model):
        for buffer, base_buffer in zip(self.net._all_buffers(), base_model._all_buffers()):
            buffer.data.copy_(base_buffer.data)

    def export_base_parameters(self, base_model, index):
        parameters = list(self.net.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            base_parameter.data.copy_(parameter.data)

    def init_linear(self):
        """
        Initialize each layer on each bend to be a linear combination of
        that layer on the first bend and that layer on the last bend.
        In other words, each bend's parameters are initialized to the
        line segment between the first and last bend's parameters.
        """
        parameters = list(self.net.parameters())
        for i in range(0, len(parameters), self.num_bends):
            weights = parameters[i:i+self.num_bends]
            for j in range(1, self.num_bends - 1):
                alpha = j * 1.0 / (self.num_bends - 1)
                weights[j].data.copy_(alpha * weights[-1].data + (1.0 - alpha) * weights[0].data)

    def init_radial(self):
        """
        Assumes the end points have roughly the same l2 norm in weight space.
        Initialize each layer on each bend to be a linear combination of
        that layer on the first bend and that layer on the last bend.
        
        Then scale each layer on each bend such that the l2 norm of the
        parameters on that bend is the same as that of the first bend
        
        Again, assuming the end points have roughly the same l2 norm in weight space).
        """
        self.init_linear()
        parameters = list(self.net.parameters())
        bend_norms = [np.linalg.norm(self.weights(t))\
                              for t in np.linspace(0, 1, self.num_bends)]
        norm_diff = bend_norms[-1] - bend_norms[0]
        for i in range(0, len(parameters), self.num_bends):
            weights = parameters[i:i+self.num_bends]
            for j in range(1, self.num_bends - 1):
                scaler = bend_norms[0] + norm_diff * j / (self.num_bends - 1)
                weights[j].data.mul_(scaler / bend_norms[j])

    def weights(self, t, concatenate=True):
        coeffs_t = self.coeff_layer(t)
        weights = []
        for module in self.curve_modules:
            weights.extend([w for w in module.compute_weights_t(coeffs_t) if w is not None])
        if concatenate:
            return np.concatenate([w.detach().cpu().numpy().ravel() for w in weights])
        return weights

    def _compute_l2(self):
        self.l2 = sum(module.l2 for module in self.curve_modules)

    def forward(self, input, t=None):
        if t is None:
            t = input.data.new(1).uniform_()
        coeffs_t = self.coeff_layer(t)
        output = self.net(input, coeffs_t)
        self._compute_l2()
        return output
    
    def get_model_from_curve(self, model_class, t, perturb_class=None, perturb_args=None):
        """
        Return a model with the same architecture as self.net, but with
        parameters corresponding to the curve at time t

        # This works, but is slow. Average gradients could be computed directly somehow
        # using the curve parameters, but this is easier to implement
        """
        # Get the weights at t
        weights = self.weights(t, concatenate=False)

        # Instantiate a new model
        model = model_class(self.net.input_size,
                            self.net.hidden_layers) # assumes tabular format

        # Copy over the weights into this TabularModel class instance
        state_dict = model.state_dict()
        for i, key in enumerate(state_dict.keys()):
            state_dict[key] = weights[i]
        model.load_state_dict(state_dict)
        if perturb_class is not None:
            model = perturb_class(model, *perturb_args)
        return model
    
    def compute_gradients(self, x, model_class, ts=np.linspace(0,1,50), perturb_class=None, perturb_args=None):
        """
        Compute the gradient of the loss with respect to the curve parameters
        """
        models = nn.ModuleList(modules=[self.get_model_from_curve(model_class, t, perturb_class, perturb_args) for t in ts])
        
        if perturb_class is not None:
            p_curve_grads = [model.compute_gradients(x, mean=True) for model in models]
        else:
            p_curve_grads = [model.compute_gradients(x, return_numpy=True) for model in models]
        return np.array(p_curve_grads)
    
    def compute_logits(self, x, model_class, ts=np.linspace(0,1,50)):
        """
        Compute the gradient of the loss with respect to the curve parameters
        """
        x = torch.FloatTensor(x)
        models = nn.ModuleList(modules=[self.get_model_from_curve(model_class, t) for t in ts])
        logits = [model(x).detach().numpy() for model in models]
        return np.array(logits)
    
    def compute_losses(self, x, y, loss_fn, model_class, ts=np.linspace(0,1,50)):
        """
        Compute the gradient of the loss with respect to the curve parameters
        """
        x = torch.FloatTensor(x)
        y = torch.tensor(y)
        models = nn.ModuleList(modules=[self.get_model_from_curve(model_class, t) for t in ts])
        losses = [loss_fn(model(x), y).detach().numpy() for model in models]
        return np.array(losses)
    
    def compute_perturbed_stats(self, x, model_class, perturb_class,
                                perturb_args, x_full=None,
                                ts=np.linspace(0,1,50)):
        if x_full is None:
            x_full = x

        pert_models = nn.ModuleList()
        for t in ts:
            pert_model = self.get_model_from_curve(model_class, t,
                                                   perturb_class,
                                                   perturb_args)
            pert_models.append(pert_model)

        grads = np.array([pert_mod.compute_gradients(x)\
                          for pert_mod in pert_models]).mean(axis=0)
        logits = np.array([pert_mod.compute_logits(x_full)\
                            for pert_mod in pert_models]).mean(axis=0)
        preds = logits.argmax(axis=-1)
        return grads, preds
    
class CurveNetPerturb(Module):
    def __init__(self, base_curve, model_class, perturb_class,
                 perturb_args, ts=np.linspace(0,1,50)):
        super().__init__()
        mods = [base_curve.get_model_from_curve(model_class, t,
                                                perturb_class,
                                                perturb_args) for t in ts]
        self.pert_models = nn.ModuleList(modules=mods)

    def forward(self, x):
        outputs = [pert_mod(x) for pert_mod in self.pert_models]
        return torch.stack(outputs, dim=0).mean(axis=0)

    def compute_gradients(self, x):
        return np.array([pert_mod.compute_gradients(x)\
                         for pert_mod in self.pert_models]).mean(axis=0)
    
    def compute_logits(self, x):
        return np.array([pert_mod.compute_logits(x)\
                         for pert_mod in self.pert_models]).mean(axis=0)


def l2_regularizer(weight_decay):
    return lambda model: 0.5 * weight_decay * model.l2
