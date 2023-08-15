import random
import torch.nn as nn
from .qat_layer import QATQuantModule, StraightThrough


class QATQuantSuperModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params, act_quant_params, quantizer, search_space):
        super().__init__()
        self.model = model
        self.layer_number = 1
        self.w_bit_list = [2, 3, 4, 5, 6, 7, 8] if search_space['w_bit_list'] is None else search_space['w_bit_list']
        self.a_bit_list = [2, 3, 4, 5, 6, 7, 8] if search_space['a_bit_list'] is None else search_space['a_bit_list']
        self.w_max = max(self.w_bit_list)
        self.a_max = max(self.a_bit_list)
        self.w_min = min(self.w_bit_list)
        self.a_min = min(self.a_bit_list)
        self.w_sym_list = [False, True] if search_space['w_sym_list'] is None else search_space['w_sym_list']
        self.a_sym_list = [False, True] if search_space['a_sym_list'] is None else search_space['a_sym_list']
        self.channel_wise_list = [False, True] if search_space['channel_wise_list'] is None else search_space[
            'channel_wise_list']

        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params, quantizer)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params, act_quant_params, quantizer):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param scale_method:
        :param quantizer:
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        # prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                weight_quant_params['layer_number'] = self.layer_number
                act_quant_params['layer_number'] = self.layer_number
                setattr(module, name,
                        QATQuantModule(child_module, weight_quant_params, act_quant_params, quantizer=quantizer))
                # prev_quantmodule = getattr(module, name)
                self.layer_number += 1
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, quantizer)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            # if isinstance(m, (QuantModule, BaseQuantBlock)):
            if isinstance(m, QATQuantModule):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input):
        return self.model(input)

    def set_random_subnet(self, ):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QATQuantModule):
                # temp_weight_params, temp_act_params = self.sample_quantization_params_for_one_layer()
                temp_weight_params = {'bit': random.choice(self.w_bit_list),
                                      'channel_wise': random.choice(self.channel_wise_list),
                                      'symmetric': random.choice(self.w_sym_list)}
                temp_act_params = {'bit': random.choice(self.a_bit_list), 'symmetric': random.choice(self.a_sym_list)}
                m.set_quantization_params(temp_weight_params, temp_act_params)
                module_list.append(m)

        module_list[0].set_quantization_bit(8, 8)
        module_list[-1].set_quantization_bit(8, 8)

    def set_biggest_subnet(self, ):
        for m in self.model.modules():
            if isinstance(m, QATQuantModule):
                weight_params = {'bit': self.w_max, 'channel_wise': random.choice(self.channel_wise_list),
                                 'symmetric': random.choice(self.w_sym_list)}
                act_params = {'bit': self.a_max, 'symmetric': random.choice(self.a_sym_list)}
                m.set_quantization_params(weight_params, act_params)

    def set_smallest_subnet(self, ):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QATQuantModule):
                weight_params = {'bit': self.w_min, 'channel_wise': random.choice(self.channel_wise_list),
                                 'symmetric': random.choice(self.w_sym_list)}
                act_params = {'bit': self.a_min, 'symmetric': random.choice(self.a_sym_list)}
                m.set_quantization_params(weight_params, act_params)
                module_list.append(m)
        module_list[0].set_quantization_bit(8, 8)
        module_list[-1].set_quantization_bit(8, 8)

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QATQuantModule):
                module_list += [m]
        module_list[0].set_quantization_bit(8, 8)
        module_list[-1].set_quantization_bit(8, 8)
        # ignore reconstruction of the first layer
        module_list[0].ignore_reconstruction = True

    def set_quantization_params(self, channel_wise, w_sym, a_sym, w_bit_list, a_bit_list):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QATQuantModule):
                if isinstance(w_bit_list, list) and isinstance(a_bit_list, list):
                    w_bit = int(w_bit_list[len(module_list)])
                    a_bit = int(a_bit_list[len(module_list)])
                else:
                    w_bit = w_bit_list
                    a_bit = a_bit_list
                wq_params = {'bit': w_bit, 'channel_wise': channel_wise, 'symmetric': w_sym}
                aq_params = {'bit': a_bit, 'symmetric': a_sym}
                m.set_quantization_params(wq_params, aq_params)
                module_list += [m]

        module_list[0].set_quantization_bit(8, 8)
        module_list[-1].set_quantization_bit(8, 8)
        # ignore reconstruction of the first layer
        module_list[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QATQuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True
