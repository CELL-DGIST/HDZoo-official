"""
TrainableHD - Yeseong Kim & Jiseung Kim (CELL) @ DGIST, 2023
"""
import torch
from torch.ao.quantization import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver


# Quantizer templates
quint8_fake_quantizer = FakeQuantize.with_args(
		observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
		dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True)

qint8_fake_quantizer = FakeQuantize.with_args(
		observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
		dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)

""" Fake Quantization for INT8 """
class GradientAwareFakeQuantizer:
    def __init__(self, dtype, update_thre):
        self.dtype = dtype
        if dtype == torch.qint8:
            self.quant_func = qint8_fake_quantizer()
        elif dtype:
            self.quant_func = quint8_fake_quantizer()
        else:
            assert False, "Not supported quantization type"

        if torch.cuda.is_available():
            self.quant_func = self.quant_func.cuda()

        self.update_thre = update_thre
        self.sum = None
        self.accumulate = 0

        self.requested_fq = 0
        self.performed_fq = 0

    def fake_quantize(self, tensor, last_grad=None):
        self.last_quantized = True
        if last_grad is not None:
            self.accumulate += abs(last_grad.sum())
            changed_rate = self.accumulate / self.sum
            self.last_quantized = changed_rate > self.update_thre

        if self.last_quantized:
            tensor = self.quant_func(tensor)
            self.sum = abs(tensor.sum())
            self.accumulate = 0
            self.performed_fq += 1

        self.requested_fq += 1
        return tensor

    def real_quantize(self, tensor, return_parameterized=True, param_transpose=True):
        scale, zero_point = self.quant_func.calculate_qparams()
        tensor = torch.quantize_per_tensor(tensor, scale, zero_point, self.dtype)
        if return_parameterized:
            if param_transpose:
                tensor = tensor.T
            tensor = torch.ops.quantized.linear_prepack(
                    tensor.cpu(), None)
        else:
            tensor = tensor.cpu()
        return tensor

    def qparams(self):
        return self.quant_func.calculate_qparams()

    def print_stat(self, log=None):
        s = "FakeQuantization:\t{} runs\t/\t{} reqs".format(
                self.performed_fq,
                self.requested_fq)

        if log is None:
            print(s)
        else:
            log.d(s)