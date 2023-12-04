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
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, 
            required=False, help='set learning rate value', dest='learning_rate')
 
    parser.add_argument('--optimizer', default='Adam', type=str,
            required=False, help='set choose optimizer', dest='optimizer',
            choices=['AdaDelta', 'RMS', 'Adagrad', 'NAG', 'Momentum', 'SGD', 'Adam'])
 
    parser.add_argument('--normalizer', default='l2', type=str,
            required=False, help='set normalizer protocol', dest='normalizer',
            choices=['l2', 'minmax'])
 
    parser.add_argument('-e', '--encoding_interval', default=1, type=float, 
            required=False, help='set encoding interval (EIT)', dest='enc_int')
 
    parser.add_argument('-qat', '--enable_qat', action='store_true',
            required=False, help='enable quantization-aware training (QAT)', dest='qat')
 
    parser.add_argument('-qth', '--qupdate_thre', default=0.01, type=float,
            required=False, help='set threshold to update quantized results', dest='qupdate_thre')
 
    parser.add_argument('-s', '--sim_metric', default='dot', type=str,
            required=False, help='set similarity metric', dest='sim_metric',
            choices=['dot', 'cos'])
 
    parser.add_argument('-l', '--logfile', default='trainableHD.log', type=str,
            required=False, help='set log file', dest='logfile')
 
    parser.add_argument('-r', '--randomseed', default=0, type=int,
            required=False, help='set random seed', dest='random_seed')
       
    args = parser.parse_args()
 
    return args
