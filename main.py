"""
HD Zoo - Yeseong Kim (CELL) @ DGIST, 2023
"""
from hdzoo.core.train import singlepass_training, init_model, retrain, retrain_mass
from hdzoo.core.inference import test

from hdzoo.core.encoder import choose_encoder
from hdzoo.core.sim_metrics import setup_global_sim_metric

from hdzoo.utils.cmdparse import parse_args
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
    encode = choose_encoder(args.encoder, args.nonbinarize, args.q)

    # Data Loading
    log.d("Loading dataset " + args.filename)
    x, y, x_test, y_test, K = load_dataset(args.filename)
    n_test = len(x_test)

    log.d("Normalize with " + args.normalizer)
    x, x_test = normalize(x, x_test, args.normalizer)
    x, y = shuffle_data(x, y)  # need to shuffle training set only
    x, y = to_tensor(x, y)
    x_test, y_test = to_tensor(x_test, y_test)

    # Encoding
    log.d("Encoding: D = {}".format(args.dimensions))
    x_h, x_test_h = encode(x, x_test, args.dimensions) 

    # Training
    if args.use_singlepass:
        log.d("Single Pass Training")
        model = singlepass_training(x_h, y, K)
    else:
        log.d("Initialize empty models")
        model = init_model(x_h, K)

    if args.iterations > 0:
        log.d("Retraining: B = {}".format(args.batch_size))
        if not args.use_mass:
            model = retrain(model, x_h, y, K,
                    x_test_h, y_test,
                    args.iterations, args.batch_size, args.learning_rate)
        else:
            model = retrain_mass(model, x_h, y, K,
                    x_test_h, y_test,
                    args.iterations, args.batch_size, args.learning_rate)

    # Testing
    n_correct = test(model, x_test_h, y_test)
    log.d("Final Testing Accuracy\t{} / {} = {}".format(
        n_correct, n_test, n_correct / n_test))


if __name__ == "__main__":
    main()
