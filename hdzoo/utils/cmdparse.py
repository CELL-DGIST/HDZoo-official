"""
HD Zoo - Yeseong Kim (CELL) @ DGIST, 2023
"""
import argparse


""" Argument parser """
def parse_args():
    global args
    parser = argparse.ArgumentParser()
 
    parser.add_argument('filename', type=str,
            help='choir training dataset')
 
    parser.add_argument('-d', '--dimensions', default=10000, type=int,
            required=False, help='set dimensions value', dest='dimensions')
 
    parser.add_argument('-i', '--iterations', default=50, type=int,
            required=False, help='set iteration number', dest='iterations')
 
    parser.add_argument('-b', '--batchsize', default=32, type=int,
            required=False, help='set batch size', dest='batch_size')
    
    parser.add_argument('-lr', '--learning_rate', default=0.035, type=float, 
            required=False, help='set learning rate value', dest='learning_rate')
 
    parser.add_argument('--normalizer', default='l2', type=str,
            required=False, help='set normalizer protocol', dest='normalizer',
            choices=['l2', 'minmax'])
 
    parser.add_argument('-sp', '--singlepass', action='store_true',
            required=False, help='use single pass(oneshot) training ' +
            '(some datasets may overfit with this', dest='use_singlepass') 
 
    parser.add_argument('--encoder', default='nonlinear', type=str,
            required=False, help='sets encoding protocol', dest='encoder',
            choices=['idlevel', 'randomproj', 'nonlinear'])
                
    parser.add_argument('-nb', '--nonbin', action='store_true',
            required=False, help='do not apply binarize hypervectors in the selected encoding method',
            dest='nonbinarize')
    
    parser.add_argument('-q', '--quantization', default=100, type=int,
            required=False, help='sets quantization level for IDLEVEL encoder',
            dest='q')
 
    parser.add_argument('-m', '--mass', action='store_true',
            required=False, help='use mass retraining',
            dest='use_mass') 
 
    parser.add_argument('-s', '--sim_metric', default='dot', type=str,
            required=False, help='set similarity metric', dest='sim_metric',
            choices=['dot', 'cos'])
  
    parser.add_argument('-l', '--logfile', default='HDzoo.log', type=str,
            required=False, help='set log file', dest='logfile')
 
    parser.add_argument('-r', '--randomseed', default=0, type=int,
            required=False, help='set random seed', dest='random_seed')
 
    args = parser.parse_args()
 
    if args.encoder == "idlevel":
            print("For the best performance, we use the minmax normalizer for the ID-LEVEL encoder")
            args.normalizer = "minmax"
 
    return args
