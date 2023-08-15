import time
import sys
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Agg')

sys.path.append('../')

from search.search_algorithm import EvolutionFinder
import time
import sys
from matplotlib import pyplot as plt
import core.config as config
import logger.logging as logging
# import models.cifar10 as models
import torchvision.models as models
from logger.meter import *
from search.accuracy_predictor.acc_predictor import AccuracyPredictor
from search.accuracy_predictor.arch_encoder import OQAEncoder
from search.bitwidth_estimator import BW_Estimator
from search.bitwidth_estimator import BitwidthDataset

config.load_configs()
logger = logging.get_logger(__name__)


def main():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    featuremaps = calc_model_featuremap(models.__dict__[cfg.ARCH]().to(device), 224)
    flops, weights, total_params, total_flops, conv_num, fc_num = get_params_flops(model=models.__dict__[cfg.ARCH](),
                                                                                   input_size=224)

    encoder = OQAEncoder(module_nums=conv_num + fc_num)
    predictor = AccuracyPredictor(encoder, hidden_size=200, n_layers=7, device=device)

    ckpt = torch.load(cfg.acc_model, map_location=device)
    predictor.load_state_dict(ckpt['state_dict'])

    bw_estimator = BW_Estimator(weights, sum(weights[1:-1]), featuremaps, sum(featuremaps[1:-1]))
    bitwidthdataset = BitwidthDataset(Bitwidth_estimator=bw_estimator, module_nums=conv_num + fc_num,
                                      path=cfg.bw_dataset_path)
    prob_map = bitwidthdataset.build_trasition_prob_matrix(0.2)

    """ Hyper-parameters for the evolutionary search process
        You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
    """
    P = 100  # The size of population in each generation
    N = 500  # How many generations of population to be searched
    r = 0.25  # The ratio of networks that are used as parents for next generation
    params = {
        'mutate_prob': 0.3,  # The probability of mutation in evolutionary search
        # 'mutate_prob': 0.3, # The probability of mutation in evolutionary search
        'mutation_ratio': 0.5,  # The ratio of networks that are generated through mutation in generation n >= 2.
        'efficiency_predictor': bw_estimator,  # To use a predefined efficiency predictor.
        'prob_map': prob_map,
        'accuracy_predictor': predictor,  # To use a predefined accuracy_predictor predictor.
        'population_size': P,
        'max_time_budget': N,
        'parent_ratio': r,
    }

    # build the evolution finder
    finder = EvolutionFinder(**params)

    # start searching
    result_lis = []

    w_bit_list = []
    a_bit_list = []
    print('channel-wise:{},w-sym:{},a-sym:{}'.format(cfg.EVAL.channel_wise, cfg.EVAL.w_sym, cfg.EVAL.a_sym))
    for bw_w, bw_a in zip([2.9, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0],
                          [2.9, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]):
        st = time.time()
        best_valids, best_info = finder.run_evolution_search(constraint=(bw_w, bw_a), verbose=True,
                                                             channel_wise=cfg.EVAL.channel_wise, w_sym=cfg.EVAL.w_sym,
                                                             a_sym=cfg.EVAL.a_sym)
        ed = time.time()
        # print('Found best architecture at flops <= %.2f M in %.2f seconds! It achieves %.2f%s predicted accuracy with %.2f MFLOPs.' % (flops, ed-st, best_info[0] * 100, '%', best_info[-1]))
        result_lis.append(best_info)
        print('\nbw_w, bw_a: \n', best_info)
        w_bit_list.append(best_info[1]['w_bit_list'])
        a_bit_list.append(best_info[1]['a_bit_list'])

        print('\n')
    print(w_bit_list)
    print(a_bit_list)
    plt.figure(figsize=(4, 4))
    plt.plot([x[-1][0] for x in result_lis], [x[0] * 100 for x in result_lis], 'x-', marker='*', color='darkred',
             linewidth=2, markersize=8, label='OFA')
    plt.xlabel('avg bit width of weights (M)', size=12)
    plt.ylabel('Predicted Holdout Top-1 Accuracy (%)', size=12)
    plt.legend(['OFA'], loc='lower right')
    plt.grid(True)
    # plt.show()
    plt.savefig(cfg.OUT_DIR + '/Pareto-Result.png', format='png')

    logger.info("Search Reslut")

    for result in result_lis:
        logger.info(result)


if __name__ == '__main__':
    main()
