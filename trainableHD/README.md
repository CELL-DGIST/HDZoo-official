# TrainableHD

TrainableHD is an advanced hyperdimensional computing (HDC) framework that introduces dynamic encoder training and the integration of modern optimizers into HDC-based learning systems.
Another key feature of TrainableHD is the quantization-aware training, which facilitates the development of low-precision models without significantly compromising accuracy.
The TrainableHD repository provides the implementation details as presented in the associated journal paper, as well as its original conference paper:

```
@inproceedings{kim2023efficient,
  title={Efficient Hyperdimensional Learning with Trainable, Quantizable, and Holistic Data Representation},
  author={Kim, Jiseung and Lee, Hyunsei and Imani, Mohsen and Kim, Yeseong},
  booktitle={2023 Design, Automation \& Test in Europe Conference \& Exhibition (DATE)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```

## Prerequisite
The implementation is built on the HD Zoo, which is the root directory in this repository. After cloning, you will find a soft link to the `hdzoo` directory.
If this link isn't functioning properly, you can run the `bootstrap.sh` script provided to resolve the issue.

### Requirement and Dataset
For information on the Python library requirements and datasets, please refer to the README page in the HD Zoo. 

To enable inference with quantized data types in PyTorch (i.e., when using quantization-aware training,)
you will need to apply a patch to [the PyTorch code](https://github.com/pytorch/pytorch) and build the library from the source code.
Information about this process can be found in the relevant [section](#torch-quantization-support-for-inference) provided below.

## Usage
Here is a simple example of how to use the TrainableHD framework.
```
python main.py ../dataset/mnist_train.choir_dat
```

You can run the implementation using different optimizers, such as Adam, to customize the training process according to your specific requirements or to experiment with various optimization strategies.
```
python main.py --optimizer Adam ../dataset/mnist_train.choir_dat
```

For optimal performance, it's advisable to conduct hyperparameter searches, such as determining the most suitable learning rate and optimizer for your learning parameters.
To get a comprehensive understanding of how to configure these settings, please refer to the following full usage instructions:

```
$ python main.py --help
usage: main.py [-h] [-d DIMENSIONS] [-i ITERATIONS] [-b BATCH_SIZE] [-lr LEARNING_RATE] [--optimizer {AdaDelta,RMS,Adagrad,NAG,Momentum,SGD,Adam}] [--normalizer {l2,minmax}] [-e ENC_INT] [-qat] [-qth QUPDATE_THRE] [-s {dot,cos}] [-l LOGFILE] [-r RANDOM_SEED] filename

positional arguments:
  filename              choir training dataset

optional arguments:
  -h, --help            show this help message and exit
  -d DIMENSIONS, --dimensions DIMENSIONS
                        set dimensions value
  -i ITERATIONS, --iterations ITERATIONS
                        set iteration number
  -b BATCH_SIZE, --batchsize BATCH_SIZE
                        set batch size
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        set learning rate value
  --optimizer {AdaDelta,RMS,Adagrad,NAG,Momentum,SGD,Adam}
                        set choose optimizer
  --normalizer {l2,minmax}
                        set normalizer protocol
  -e ENC_INT, --encoding_interval ENC_INT
                        set encoding interval (EIT)
  -qat, --enable_qat    enable quantization-aware training (QAT)
  -qth QUPDATE_THRE, --qupdate_thre QUPDATE_THRE
                        set threshold to update quantized results
  -s {dot,cos}, --sim_metric {dot,cos}
                        set similarity metric
  -l LOGFILE, --logfile LOGFILE
                        set log file
  -r RANDOM_SEED, --randomseed RANDOM_SEED
                        set random seed
``` 


## Torch Quantization Support for Inference
While quantization-aware training in this framework might not require the patch, as it emulates the INT8 data type using float32 precision during the training procedure, performing inference with quantized data types does necessitate patching PyTorch.
This is specifically to support the sign() operation, which is not natively accommodated in standard PyTorch builds.
Please follow the detailed instructions provided in the [pytorch_patch_quant_sign](pytorch_patch_quant_sign) directory.
These instructions will guide you through the process of modifying the PyTorch code to include support for the sign() operation in quantized data types.
