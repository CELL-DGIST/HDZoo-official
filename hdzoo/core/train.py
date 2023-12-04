"""
HD Zoo - Yeseong Kim (CELL) @ DGIST, 2023
"""
import torch

from tqdm import tqdm

from ..core.sim_metrics import sim_metric
from ..utils.logger import log


"""
Singlepass training
Note: It does not perform any similarity checks during the training,
so the model tends to be overfit, and the accuracy is typically very low.
So, in our framework, a better way would be to initialize the model with zero elements
and run a single-epoch learning.
"""
def singlepass_training(x, y, K):
    D = x.size(1) # extract from the tensor for convenience
    model = torch.zeros(K, D, dtype=x.dtype, device=x.device)
    for c in range(K):
        model[c].add_(x[y == c].sum(0))
    return model

"""
Initialize the model with zero elements (use it for the retraining)
Provide a tensor x, to infer the dimension size, data type, and device
"""
def init_model(x, K):
    model = torch.zeros(
            K, x.size(1), dtype=x.dtype, device=x.device)

    return model

"""
Default callback to print the training progress
"""
def default_callback(epoch, acc, test_acc, loss):
    log.i("[Retraining] Epoch {}: {} {} {}".format(
            epoch, acc, test_acc, loss))

"""
Retraining procedure -
The general idea of bundling the hypervectors only for the misclassified samples is proposed here (probably for the first time):
- Imani, Mohsen, Deqian Kong, Abbas Rahimi, and Tajana Rosing. "Voicehd: Hyperdimensional computing for efficient speech recognition." In 2017 IEEE international conference on rebooting computing (ICRC), pp. 1-8. IEEE, 2017.
If you want to look at the pseudocode, you may refer to this paper:
- Kim, Yeseong, Mohsen Imani, and Tajana S. Rosing. "Efficient human activity recognition using hyperdimensional computing." In Proceedings of the 8th International Conference on the Internet of Things, pp. 1-6. 2018.
I'm not sure when we started using the learning rate, but at least the following paper would be one of our earlier works that introduced the learning rate.
- Imani, Mohsen, Yeseong Kim, Sadegh Riazi, John Messerly, Patric Liu, Farinaz Koushanfar, and Tajana Rosing. "A framework for collaborative learning in secure high-dimensional space." In 2019 IEEE 12th International Conference on Cloud Computing (CLOUD), pp. 435-446. IEEE, 2019.
"""
def retrain(
        model, x, y, K,
        x_test=None, y_test=None,
        epochs=30, batch_size=32, lr=0.035,
        callback=default_callback):
    pbar = tqdm(range(epochs))
    train_accuracy = None
    test_accuracy =  None
    fully_trained = False
    N = x.size(0)

    for epoch in pbar:
        if train_accuracy is not None:
            if test_accuracy is None:
                pbar.set_description(
                        "ACC=%.4f" % (train_accuracy))
            else:
                pbar.set_description(
                        "ACC=%.4f TACC=%.4f" % (train_accuracy, test_accuracy))

        n_correct = 0
        for i in range(0, N, batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            sims = sim_metric(x_batch, model)
            y_pred = sims.argmax(1)
            wrong = y_batch != y_pred
            n_correct += x_batch.size(0) - wrong.sum().cpu().numpy()

            # Model adjustment:
            # Note: It may looks complicated if you're not familiar with torch.eye().
            # But, it implements the common idea of the retraining procedure using bundling:
            # "for misclassified sample, add the hypervectors into the desired class
            # and subtracts them from the incorrect class."
            # We implement it with torch.eye to enhance the performance.
            wrong_mask = wrong.repeat(K, 1).T   # Take the samples of misclassified samples
            eye = torch.eye(K, dtype=torch.bool, device=y_batch.device)
            correct_update = wrong_mask & eye[y_batch.long()]  # For the misclassified samples, set the true mask for the desired class
            wrong_update = wrong_mask & eye[y_pred]  # For the misclassified samples, set the true mask for the incorrect class
            correct_update = correct_update.float()
            wrong_update = wrong_update.float()
            updates = torch.matmul((correct_update - wrong_update).T, x_batch)  # add and subtracts misclassified hypervectors
            model.add_(updates, alpha=lr)  # bundle them with a learning rate

        if callback is not None:
            train_accuracy = n_correct / N
            if x_test is not None:
                sims = sim_metric(x_test, model)
                y_test_pred = sims.argmax(1)
                n_correct_test = (y_test == y_test_pred.T).sum().cpu().numpy()
                test_accuracy = n_correct_test / y_test.size(0)

            callback(epoch+1, train_accuracy, test_accuracy, 0.0)

            if train_accuracy == 1.0:
                if fully_trained: # fully trained two times
                    log.d("Fully trained. Stop.")
                    break

                fully_trained = True

    return model

"""
MASS Retraining procedure
- Kim, Yeseong, Jiseung Kim, and Mohsen Imani. "Cascadehd: Efficient many-class learning framework using hyperdimensional computing." In 2021 58th ACM/IEEE Design Automation Conference (DAC), pp. 775-780. IEEE, 2021.
"""
def retrain_mass(
        model, x, y, K,
        x_test=None, y_test=None,
        epochs=50, batch_size=32, 
        lr=0.035, callback=default_callback):
    N = x.size(0)
    # Precompute one-hot encoding for the desired class
    gcode = torch.zeros(N, K, dtype=x.dtype, device=x.device)
    for c in range(K):
        gcode[y == c, c] = 1 

    pbar = tqdm(range(epochs))
    train_accuracy = None
    test_accuracy =  None
    loss = None
    for epoch in pbar:
        if train_accuracy is not None:
            if test_accuracy is None:
                pbar.set_description(
                        "ACC=%.4f" % (train_accuracy))
            else:
                pbar.set_description(
                        "ACC=%.4f TACC=%.4f LOSS=%.4f" % (
                            train_accuracy, test_accuracy, loss))

        # Compute code (one hot encoding)
        n_correct = 0
        for i in range(0, N, batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            code_batch = gcode[i:i+batch_size]
            sims = sim_metric(x_batch, model)
            y_pred = sims.argmax(1)
            wrong = y_batch != y_pred
            n_correct += x_batch.size(0) - wrong.sum().cpu().numpy()

            # Similarity thresholding and regularzation
            sims.relu_()
            sims = torch.nn.functional.softmax(sims, dim=1)

            # Model adjustment
            updates = (code_batch - sims) 
            updates = torch.matmul(updates.T, x_batch)
            model.add_(updates, alpha=lr)

        if callback is not None:
            # Accuracy
            train_accuracy = n_correct / N
            if x_test is not None:
                sims = sim_metric(x_test, model)
                y_test_pred = sims.argmax(1)
                n_correct_test = (y_test == y_test_pred.T).sum().cpu().numpy()
                test_accuracy = n_correct_test / y_test.size(0)

            # Loss
            sims = sim_metric(x, model)
            vcode = torch.zeros_like(sims)
            vcode[y == c] = 1.0
            simsize = sims.size(0) * sims.size(1)
            loss = (sims - vcode).abs().sum().cpu().numpy() / simsize

            callback(epoch+1, train_accuracy, test_accuracy, loss)

    return model
