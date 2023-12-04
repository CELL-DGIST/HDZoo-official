"""
HD Zoo - Yeseong Kim (CELL) @ DGIST, 2023
"""
import torch

from ..utils.logger import log


# dot product similarity
dot_metric = lambda x, model: torch.matmul(x, model.T)

# cos simlarity
def cos_metric(x, model):
    numerator = torch.matmul(x, model.T)
    x_norms = x.norm(dim=1).max(cos_metric.eps)
    model_norms = model.norm(dim=1).max(cos_metric.eps)
    denominator = torch.mul(x_norms.unsqueeze(1), model_norms.unsqueeze(0))

    return numerator.div_(denominator)

cos_metric.eps = torch.tensor(1e-8)
if torch.cuda.is_available():
    cos_metric.eps = cos_metric.eps.cuda()

# Set Global Similarity Metric
def setup_global_sim_metric(sim_metric_str):
    global sim_metric
    if sim_metric_str == 'dot':
        Sim_metric_holder.default = dot_metric
        log.d("Dot product similarity metric")
    elif sim_metric_str == 'cos':
        Sim_metric_holder.default = cos_metric
        log.d("Cosine similarity metric")
    else:
        raise NotImplementedError

class Sim_metric_holder:
    default = dot_metric
    def __call__(self, x, model):
        return Sim_metric_holder.default(x, model)
Sim_metric_holder.default = dot_metric

sim_metric = Sim_metric_holder()
