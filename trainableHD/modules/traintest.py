"""
TrainableHD - Yeseong Kim & Jiseung Kim (CELL) @ DGIST, 2023
"""
import sys
import torch

from tqdm import tqdm

from .optimizers import choose_optim, SGD
from .quant_util import GradientAwareFakeQuantizer

from hdzoo.core.encoder import hardsign
from hdzoo.core.train import default_callback
from hdzoo.core.sim_metrics import sim_metric
from hdzoo.utils.logger import log


""" Create the base in the way used in non-linear encoding """
def create_base(x, D):
    # Gaussian sampler configuration
    mu = 0.0
    sigma = 1.0

    # Get feature size from the dataset
    F = x.size(1)

    # Create base hypervector
    bases = torch.empty(D, F, dtype=x.dtype, device=x.device)
    bases = bases.normal_(mu, sigma).T
    return bases


"""
Training
the baseline is from MASS [CascadeHD, DAC'21]; but perform the encoder training
"""
def train(
        model, bases,
        x, y, x_test, y_test, K, lr, epochs, batch_size, args,
        callback=default_callback):
    N = x.size(0)
    D = bases.size(1)
    
    # Compute code (one hot encoding)
    gcode = torch.zeros(N, K, dtype=x.dtype, device=x.device)
    for c in range(K):
        gcode[y == c, c] = 1 

    pbar = tqdm(range(epochs))
    train_accuracy = None
    test_accuracy =  None
    loss = None

    # Optimizer
    H_optim = choose_optim(args.optimizer)
    B_optim = choose_optim(args.optimizer)

    # Quantization-aware Training
    if args.qat:
        # Fake Quantizers
        input_fake_quant = GradientAwareFakeQuantizer(
                torch.quint8, args.qupdate_thre)
        base_hv_fake_quant = GradientAwareFakeQuantizer(
                torch.qint8, args.qupdate_thre)
        model_hv_fake_quant = GradientAwareFakeQuantizer(
                torch.qint8, args.qupdate_thre)

        # Fake-Quantize required values in advance
        x = input_fake_quant.fake_quantize(x)
        bases = base_hv_fake_quant.fake_quantize(bases)
        model = model_hv_fake_quant.fake_quantize(model)

    for epoch in pbar:
        if train_accuracy is not None:
            if test_accuracy is None:
                pbar.set_description(
                        "ACC=%.4f" % (train_accuracy))
            else:
                pbar.set_description(
                        "ACC=%.4f TACC=%.4f LOSS=%.4f" % (
                            train_accuracy, test_accuracy, loss))

        loss = 0
        n_correct = 0
        if x_test is not None:
            H_test = torch.empty(
                    x_test.size(0), D, dtype=x.dtype, device=x.device)
        H_temp = torch.empty(x.size(0), D, dtype=x.dtype, device=x.device)  # EIT store

        for i in range(0, N, batch_size):
            bs = min(batch_size, N-i)
            x_batch = x[i:i+bs]
            y_batch = y[i:i+bs]

            # Encoder Interval Training (EIT)
            if epoch % args.enc_int==0:
                X = torch.matmul(x_batch, bases)
                H_temp[i:i+bs] = X
            else:
                X = H_temp[i:i+bs]
            H = hardsign(X)

            # Simliarity search
            code_batch = gcode[i:i+bs]
            sims = sim_metric(H, model)
            y_pred = sims.argmax(1)
            wrong = y_batch != y_pred
            n_correct += x_batch.size(0) - wrong.sum().cpu().numpy()
            sims.relu_() 
            sims = torch.nn.functional.softmax(sims, dim=1)
            
            # Model adjustment
            updates = (code_batch - sims)
            updates[sims < 0] = 0 
            if H_optim == SGD:  # to reduce the computation on base adjustment
                updates = updates * lr
            H_updates = torch.matmul(updates.T, H)
            if H_optim != SGD:
                H_updates = H_optim.optimize(H_updates, lr)
            model.add_(H_updates)

            # Base Adjustment - performed for valid EIT epoch
            if epoch % args.enc_int == 0:
                # Propagate training to the base
                E = torch.matmul(updates, model)
                H_act = 1 - torch.pow(torch.tanh(X), 2)
                B_updates = E * H_act

                B_updates = torch.matmul(x_batch.T, B_updates)
                if H_optim != SGD:
                    B_updates = B_optim.optimize(B_updates, lr)
                bases.add_(B_updates)

            # Quantization-aware Training
            if args.qat:
                bases = base_hv_fake_quant.fake_quantize(bases, B_updates)
                model = model_hv_fake_quant.fake_quantize(model, H_updates)

        if callback is not None:
            # Calculate loss for debugging
            simsize = sims.size(0) * sims.size(1)
            loss += updates.abs().sum().cpu().numpy() / simsize

            # Accuracy
            train_accuracy = n_correct / N
            if x_test is not None:
                # Encoding
                torch.matmul(x_test, bases, out=H_test)
                H_test = hardsign(H_test)

                # Similarity search
                sims = sim_metric(H_test, model)
                y_test_pred = sims.argmax(1)
                n_correct_test = (y_test == y_test_pred.T).sum().cpu().numpy()
                test_accuracy = n_correct_test / y_test.size(0)

            callback(epoch+1, train_accuracy, test_accuracy, loss)

    if args.qat:
        bases = base_hv_fake_quant.fake_quantize(bases)
        base_hv_fake_quant.print_stat(log)
        bases = base_hv_fake_quant.real_quantize(bases)
        model = model_hv_fake_quant.fake_quantize(model)
        model_hv_fake_quant.print_stat(log)
        model = model_hv_fake_quant.real_quantize(model, True, False)

    return model, bases


""" Testing supporting quantization """
def test(model, bases, x, y, batch_size, args):
    N = x.size(0)
    use_cuda = torch.cuda.is_available()

    # Quantize the input if needed
    if args.qat:
        input_fake_quant = GradientAwareFakeQuantizer(
                torch.quint8, args.qupdate_thre)
        x = input_fake_quant.fake_quantize(x)
        x = input_fake_quant.real_quantize(x, False)
        x_qscale, x_qzeropoint = input_fake_quant.qparams()

    warn_printed = False

    n_correct = 0
    for i in range(0, N, batch_size):
        bs = min(batch_size, N-i)
        x_batch = x[i:i+bs]
        y_batch = y[i:i+bs]

        # Encode
        if args.qat:
            # Note: for fair comparison,
            # the result should be compared with the testing on CPU
            H = torch.ops.quantized.linear(
                    x[i:i+bs], bases,
                    x_qscale, x_qzeropoint)
            if not args.qat:
                H = H.dequantize()
                if use_cuda:
                    H = H.cuda()
        else:
            H = torch.matmul(x_batch, bases)

        if not args.qat:
            H = hardsign(H)
        else:
            try:
                H = torch.ops.quantized.sign(H)
            except:
                # NOTE: Inference with the quantized model needs the sign operation running for qint type
                # which does not exist in the original Pytorch.
                # Here, we workaround it by converting it back to float32 but lose performance
                if not warn_printed:
                    print("Warning: Quantized inference is not support in your installed pytorch.")
                    print("You must patch the pytorch to get the actual performance benefit.")
                    print("See the instruction in the repo.", file=sys.sterr)
                    warn_printed = True

                H = H.dequantize()
                H = hardsign(H)
                H = torch.quantize_per_tensor(H, 0.01, 128, torch.quint8)

        # Simliarity search
        if args.qat:
            y_pred = torch.ops.quantized.linear(
                H, model, H.q_scale(), H.q_zero_point())
        else:
            y_pred = sim_metric(H, model)

        if args.qat:
            # This should not be accounted in measuring inference time
            y_pred = y_pred.dequantize()
            if use_cuda:
                y_pred = y_pred.cuda()

        y_pred = y_pred.argmax(1)
        n_correct += sum(y_batch == y_pred.T)

    return n_correct.sum().cpu().numpy()
