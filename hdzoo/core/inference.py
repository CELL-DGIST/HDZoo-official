"""
HD Zoo - Yeseong Kim (CELL) @ DGIST, 2023
"""
from ..core.sim_metrics import sim_metric


""" Perform inference """
def test(model, x, y):
    y_pred = sim_metric(x, model).argmax(1)
    n_correct = y == y_pred.T
    return n_correct.sum().cpu().numpy()