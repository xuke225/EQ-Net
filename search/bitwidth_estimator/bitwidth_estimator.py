import copy

__all__ = ['BW_Estimator']


def print_model_params(params, bitwidth):
    bw = copy.deepcopy(bitwidth)
    # bw[0]=bw[-1]=8
    bw[0] = bw[-1] = 0  # exclude the first and the last stage

    quan_params = [a * b for a, b in zip(bw, params)]
    quan_total_params = sum(quan_params)
    return quan_total_params


def print_model_featuremap(featuremaps, bitwidth):
    bw = copy.deepcopy(bitwidth)
    # bw[0]=bw[-1]=8
    bw[0] = bw[-1] = 0  # exclude the first and the last stage
    quan_featuremaps = [a * b for a, b in zip(bw, featuremaps)]
    quan_total_featuremaps = sum(quan_featuremaps)
    return quan_total_featuremaps


def get_avg_bw(params, featuremaps, total_params, total_featuremap, bw_w, bw_a):
    quan_params = print_model_params(params, bw_w)
    quan_featuremaps = print_model_featuremap(featuremaps, bw_a)

    return float(quan_params / total_params), float(quan_featuremaps / total_featuremap)


class BW_Estimator:
    def __init__(self, params, total_params, featuremaps, total_featuremap):
        self.params = params
        self.featuremaps = featuremaps
        self.total_params = total_params
        self.total_featuremap = total_featuremap

    def get_efficiency(self, sample):
        return get_avg_bw(self.params, self.featuremaps, self.total_params, self.total_featuremap, sample['w_bit_list'],
                          sample['a_bit_list'])
