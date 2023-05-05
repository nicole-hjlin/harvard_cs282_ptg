import torch
import numpy as np
import ot

def load_aligned_model(avg_aligned_layers, model_class,
                       model_args, bias=True):
    model = model_class(*model_args)
    state_dict = model.state_dict()
    j = 0
    for layer in state_dict.keys():
        if len(state_dict[layer].shape) == 1:
            if bias:
                state_dict[layer] = avg_aligned_layers[j]
                j += 1
        else:
            state_dict[layer] = avg_aligned_layers[j]
            j+=1
    model.load_state_dict(state_dict)
    return model

def isnan(x):
    return x != x

def get_histogram(args, cardinality):
    # returns a uniform measure
    if not args.unbalanced:
        print("returns a uniform measure of cardinality: ", cardinality)
        return np.ones(cardinality)/cardinality
    else:
        return np.ones(cardinality)

class GroundMetric:
    """
        Ground Metric object for Wasserstein computations:

    """

    def __init__(self, params, not_squared = False):
        self.params = params
        self.ground_metric_type = params.ground_metric
        self.ground_metric_normalize = params.ground_metric_normalize
        self.reg = params.reg
        if hasattr(params, 'not_squared'):
            self.squared = not params.not_squared
        else:
            # so by default squared will be on!
            self.squared = not not_squared
        self.mem_eff = params.ground_metric_eff

    def _clip(self, ground_metric_matrix):
        return ground_metric_matrix
        # if self.params.debug:
        #     print("before clipping", ground_metric_matrix.data)

        # percent_clipped = (float((ground_metric_matrix >= self.reg * self.params.clip_max).long().sum().data) \
        #                    / ground_metric_matrix.numel()) * 100
        # print("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        # setattr(self.params, 'percent_clipped', percent_clipped)
        # # will keep the M' = M/reg in range clip_min and clip_max
        # ground_metric_matrix.clamp_(min=self.reg * self.params.clip_min,
        #                                      max=self.reg * self.params.clip_max)
        # if self.params.debug:
        #     print("after clipping", ground_metric_matrix.data)
        # return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):

        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            print("Normalizing by max of ground metric and which is ", ground_metric_matrix.max())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.ground_metric_normalize == "median":
            print("Normalizing by median of ground metric and which is ", ground_metric_matrix.median())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.ground_metric_normalize == "mean":
            print("Normalizing by mean of ground metric and which is ", ground_metric_matrix.mean())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (isnan(ground_metric_matrix).any())

    def _cost_matrix_xy(self, x, y, p=2, squared = True):
        # TODO: Use this to guarantee reproducibility of previous results and then move onto better way
        "Returns the matrix of $|x_i-y_j|^p$."
        bias = True if len(x.size()) == 1 else False
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        diff = torch.abs(x_col - y_lin) ** p
        print("diff is ", diff.size())
        c = diff if bias else torch.sum(diff, 2)
        print("cost matrix is ", c.size())
        if not squared:
            print("dont leave off the squaring of the ground metric")
            c = c ** (1/2)
        # print(c.size())
        if self.params.dist_normalize:
            assert NotImplementedError
        return c


    def _pairwise_distances(self, x, y=None, squared=True):
        '''
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        dist = torch.clamp(dist, min=0.0)

        if self.params.activation_histograms and self.params.dist_normalize:
            dist = dist/self.params.act_num_samples
            print("Divide squared distances by the num samples")

        if not squared:
            print("dont leave off the squaring of the ground metric")
            dist = dist ** (1/2)

        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        # TODO: Replace by torch.pdist (which is said to be much more memory efficient)

        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1]) \
                - coordinates, p=2, dim=2
            )
        else:
            if self.mem_eff:
                matrix = self._pairwise_distances(coordinates, other_coordinates, squared=self.squared)
            else:
                matrix = self._cost_matrix_xy(coordinates, other_coordinates, squared = self.squared)

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        print("stats of vecs are: mean {}, min {}, max {}, std {}".format(
            norms.mean(), norms.min(), norms.max(), norms.std()
        ))
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1) @ torch.norm(other_coordinates, dim=1).view(1, -1)
            )
        return matrix.clamp_(min=0)

    def _get_angular(self, coordinates, other_coordinates=None):
        pass

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            'euclidean': self._get_euclidean,
            'cosine': self._get_cosine,
            'angular': self._get_angular,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        print('Processing the coordinates to form ground_metric')
        if self.params.geom_ensemble_type == 'wts' and self.params.normalize_wts:
            print("In weight mode: normalizing weights to unit norm")
            coordinates = self._normed_vecs(coordinates)
            if other_coordinates is not None:
                other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)

        if self.params.debug:
            print("coordinates is ", coordinates)
            if other_coordinates is not None:
                print("other_coordinates is ", other_coordinates)
            print("ground_metric_matrix is ", ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        ground_metric_matrix = self._normalize(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params.clip_gm:
            ground_metric_matrix = self._clip(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params.debug:
            print("ground_metric_matrix at the end is ", ground_metric_matrix)

        return ground_metric_matrix

def align_networks(args, networks):
    '''
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    # simple_model_0, simple_model_1 = networks[0], networks[1]
    # simple_model_0 = get_trained_model(0, model='simplenet')
    # simple_model_1 = get_trained_model(1, model='simplenet')

    avg_aligned_layers = []
    # cumulative_T_var = None
    T_var = None
    # print(list(networks[0].parameters()))
    previous_layer_shape = None
    ground_metric_object = GroundMetric(args)

    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):

        fc_layer0_weight_data = fc_layer0_weight.data
        fc_layer1_weight_data = fc_layer1_weight.data

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        if (len(fc_layer0_weight_data.shape) > 1) or (args.bias == True):
            assert fc_layer0_weight.shape == fc_layer1_weight.shape
            print("Previous layer shape is ", previous_layer_shape)
            previous_layer_shape = fc_layer1_weight.shape
            if idx == 0:
                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)
                aligned_wt = fc_layer0_weight_data
            else:
                print("shape of layer: model 0", fc_layer0_weight_data.shape)
                print("shape of layer: model 1", fc_layer1_weight_data.shape)
                print("shape of previous transport map", T_var.shape)

                # aligned_wt = None, this caches the tensor and causes OOM
                # if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                #     # Handles the switch from convolutional layers to fc layers
                #     fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                #     aligned_wt = torch.bmm(
                #         fc_layer0_unflattened,
                #         T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                #     ).permute(1, 2, 0)
                #     aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                # else:
                aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                # M = cost_matrix(aligned_wt, fc_layer1_weight)
                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
                print("ground metric shape is ", M.shape)

                if idx == (num_layers - 1):
                    print("Simple averaging of last layer weights. NO transport map needs to be computed")
                    if args.ensemble_step != 0.5:
                        avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
                                            args.ensemble_step * fc_layer1_weight)
                    else:
                        avg_aligned_layers.append((aligned_wt + fc_layer1_weight)/2)
                    return avg_aligned_layers
            
            mu = get_histogram(args, mu_cardinality)
            nu = get_histogram(args, nu_cardinality)

            cpuM = M.data.cpu().numpy()
            if args.exact:
                T = ot.emd(mu, nu, cpuM)
            else:
                T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
            T_var = torch.from_numpy(T).float()
            print("Shape of transport map is ", T_var.shape)

            if args.past_correction:
                print("this is past correction for weight mode")
                print("Shape of aligned wt is ", aligned_wt.shape)
                print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)
                t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
            else:
                t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

            if t_fc0_model.shape[1] == 1:
                t_fc0_model = t_fc0_model.squeeze(1)
                print("bias shape", t_fc0_model.shape)

            # Average the weights of aligned first layers
            if args.ensemble_step != 0.5:
                if len(fc_layer1_weight_data.shape) == 1:
                    geometric_fc = ((1-args.ensemble_step) * t_fc0_model +
                                args.ensemble_step * fc_layer1_weight_data)
                else:
                    geometric_fc = ((1-args.ensemble_step) * t_fc0_model +
                                    args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                print("fc_layer1_weight_data shape is ", fc_layer1_weight_data.shape)
                if len(fc_layer1_weight_data.shape) == 1:
                    geometric_fc = (t_fc0_model + fc_layer1_weight_data)/2
                else:
                    geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))/2
            print("geomtric fc shape is", geometric_fc.shape)
            avg_aligned_layers.append(geometric_fc)

    return avg_aligned_layers
