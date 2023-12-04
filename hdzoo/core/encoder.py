"""
HD Zoo - Yeseong Kim (CELL) @ DGIST, 2023
"""
import torch

from tqdm import tqdm

from ..utils.logger import log


""" Encoder selection """
def choose_encoder(encoder, nonbinarize, q_in_idlevel):
    global encode
    if encoder == 'idlevel':
        # ID Level []
        encode = encode_idlevel
        encode.q = q_in_idlevel
        log.d(">>>>> IDLEVEL encoding")
    elif encoder == 'randomproj':
        # random projection []
        encode = encode_random_projection
        log.d(">>>>> Random Projection encoding")
    elif encoder == 'nonlinear':
        # nonlinear encoding []
        encode = encode_nonlinear
        log.d(">>>>> Non-linear encoding")
    else:
        raise NotImplementedError

    encode.nonbinarize = nonbinarize
    return encode


""" Sign function that only returns -1 or 1 """
def hardsign(x):
    # torch.sign() returns teneray data, i.e., -1, 0, 1, so the valid implementation will be follows
    x_ = torch.ones_like(x)
    x_[x < 0] = -1.0
    return x_


"""
ID-LEVEL encoding
- Imani, Mohsen, Deqian Kong, Abbas Rahimi, and Tajana Rosing. "Voicehd: Hyperdimensional computing for efficient speech recognition." In 2017 IEEE international conference on rebooting computing (ICRC), pp. 1-8. IEEE, 2017.
"""
def encode_idlevel(x, x_test, D):
    batch_size = 64
    F = x.size(1)
    Q = encode.q

    # Build level hypervectors
    base = np.ones(D)
    base[:D//2] = -1.0
    l0 = np.random.permutation(base)
    levels = list() 
    for i in range(Q+1):
        flip = int(int(i/float(Q) * D) / 2)
        li = np.copy(l0)
        li[:flip] = l0[:flip] * -1
        levels.append(li)
    levels = torch.tensor(np.array(levels), dtype=x.dtype, device=x.device)

    # Create base (ID) hypervector 
    bases = []
    for _ in range(F):
        base = np.ones(D)
        base[:D//2] = -1.0
        base = np.random.permutation(base)
        bases.append(base)
    bases = torch.tensor(np.array(bases), dtype=x.dtype, device=x.device)

    def _encode(samples, levels, bases):
        N = samples.size(0)
        H = torch.empty(N, D, dtype=samples.dtype, device=samples.device)
        for i in tqdm(range(0, N, batch_size)):
            ids = []
            sample = samples[i:i+batch_size]
            level_indices = (sample * Q).long()

            levels_batch = levels[level_indices]
            hv = levels_batch.mul_(bases).sum(dim=1)
            if not encode.nonbinarize:
                H[i:i+batch_size] = hardsign(hv)
            del sample
            del level_indices
            del levels_batch
        return H

    x_h = _encode(x, levels, bases)
    x_test_h = None
    if x_test is not None:
        x_test_h = _encode(x_test, levels, bases)

    return x_h, x_test_h


"""
Random projection encoding
- Imani, Mohsen, Yeseong Kim, Sadegh Riazi, John Messerly, Patric Liu, Farinaz Koushanfar, and Tajana Rosing. "A framework for collaborative learning in secure high-dimensional space." In 2019 IEEE 12th International Conference on Cloud Computing (CLOUD), pp. 435-446. IEEE, 2019.
"""
def encode_random_projection(x, x_test, D):
    # Configurations: no impacts on training quality
    batch_size = 512
    F = x.size(1)

    # Create base hypervector
    bases = []
    for _ in range(F):
        base = np.ones(D)
        base[:D//2] = -1.0
        base = np.random.permutation(base)
        bases.append(base)

    bases = torch.tensor(np.array(bases), dtype=x.dtype, device=x.device)
    def _encode(samples, bases):
        N = samples.size(0)
        H = torch.empty(N, D, dtype=samples.dtype, device=samples.device)
        for i in tqdm(range(0, N, batch_size)):
            torch.matmul(samples[i:i+batch_size], bases, out=H[i:i+batch_size])

        if not encode.nonbinarize:
            H = hardsign(H)
        return H

    x_h = _encode(x, bases)
    x_test_h = None
    if x_test is not None:
        x_test_h = _encode(x_test, bases)

    return x_h, x_test_h


"""
Non-linear encoding
- Imani, Mohsen, Saikishan Pampana, Saransh Gupta, Minxuan Zhou, Yeseong Kim, and Tajana Rosing. "Dual: Acceleration of clustering algorithms using digital-based processing in-memory." In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 356-371. IEEE, 2020.
"""
def encode_nonlinear(x, x_test, D, y=None):
    # Gaussian sampler configuration
    mu = 0.0
    sigma = 1.0

    # Configurations: no impacts on training quality
    batch_size = 512
    F = x.size(1)

    # Create base hypervector
    bases = torch.empty(D, F, dtype=x.dtype, device=x.device)
    bases = bases.normal_(mu, sigma).T

    def _encode(samples, bases):
        N = samples.size(0)
        H = torch.empty(N, D, dtype=samples.dtype, device=samples.device)
        for i in tqdm(range(0, N, batch_size)):
            torch.matmul(samples[i:i+batch_size], bases, out=H[i:i+batch_size])
            H[i:i+batch_size].cos_()
            if not encode.nonbinarize:
                H[i:i+batch_size] = hardsign(H[i:i+batch_size])
        return H

    x_h = _encode(x, bases)
    x_test_h = None
    if x_test is not None:
        x_test_h = _encode(x_test, bases)

    return x_h, x_test_h
