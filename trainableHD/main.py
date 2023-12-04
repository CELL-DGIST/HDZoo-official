"""
TrainableHD - Yeseong Kim & Jiseung Kim (CELL) @ DGIST, 2023
"""
from modules.traintest import create_base, train, test
from modules.cmdparse import parse_args

from hdzoo.core.train import init_model
from hdzoo.core.sim_metrics import setup_global_sim_metric

from hdzoo.utils.common import setup_seed, to_tensor
from hdzoo.utils.logger import setup_global_logger, log
from hdzoo.utils.dataset import load_dataset, shuffle_data, normalize


""" Main function """
def main():
    # Setup
    args = parse_args()
    setup_global_logger(args.logfile)
    setup_seed(args.random_seed)
    setup_global_sim_metric(args.sim_metric)

    # Data Loading
    log.d("Loading dataset" + args.filename)
    x, y, x_test, y_test, K = load_dataset(args.filename)
    n_test = len(x_test)

    log.d("Normalize with " + args.normalizer)
    x, x_test = normalize(x, x_test, args.normalizer)
    x, y = shuffle_data(x, y)  # need to shuffle training set only
    x, y = to_tensor(x, y)
    x_test, y_test = to_tensor(x_test, y_test)

    # Initialize Model and Base
    log.d("Creating Bases: D = {}".format(args.dimensions))
    bases = create_base(x, args.dimensions)

    log.d("Initialize empty models")
    model = init_model(bases, K)

    # Retraining
    log.d("Training: B = {}".format(args.batch_size))
    log.d("Learning rate: lr = {}".format(args.learning_rate))
    model, bases = train(
            model, bases,
            x, y, x_test, y_test, K,
            args.learning_rate, args.iterations, args.batch_size, args)

    # Testing
    if x_test is not None:
        n_correct = test(model, bases, x_test, y_test, args.batch_size, args)
        log.d("Final Testing Accuracy\t{} / {} = {}".format(
            n_correct, n_test, n_correct / n_test))
        

if __name__ == "__main__":
    main()
