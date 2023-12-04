# HD Zoo
<p align="center"><img src="hdzoo.png" width="300" alt="HD zoo logo"/></p>

Hyperdimensional Computing (HD Computing or HDC) is an alternative computing approach inspired by the human brain. Researchers are exploring HDC for its potential as an efficient learning method.

The repository, HD Zoo, serves as a collection of classification learning algorithms, covering multiple academic papers.
While HD Zoo may not encompass every essence of HDC literature, we hope it can at least offer a good enough overview of our previous work and provide a starting/reference point to devise new ideas in the field of HDC.
For a list of the specific papers implemented in HD Zoo, please see the [Supported Algorithms section](#supported-algorithms).

The implementations in HD Zoo are derived from various codes developed over a series of research studies in our research team, [CELL @ DGIST](https://cell.dgist.ac.kr/) (probably, some code would come from my old times at UCSD).
We have assembled these in conjunction with the release of our latest publication, TrainableHD (DAC'23), the code for which is also included in this repository.

Hope it helps!

Yeseong

### Torch-based Implementation
The HD Zoo framework is primarily built using PyTorch and is thus optimized for GPU acceleration using float32 precision.
It should be fully functional to obtain experimental results for the accuracies. (If you notice significant discrepancies in accuracy compared to your expectations, it would be due to the bugs that happened during our code collection. In that case, please let us know.)
While the framework performs well for most tasks, certain features like ID-level encoding may be more efficiently executed on dedicated hardware accelerators, such as FPGA or ASIC, due to their binary-centric nature.
Additionally, even within the GPU domain, there may be more optimized implementations, possibly involving native CUDA coding, that leverage these binary-centric characteristics more effectively.
So, please approach performance extrapolation cautiously when considering different hardware and platform configurations.
The HD Zoo should be viewed as a potential, but not exclusive, implementation for GPU-based code, adaptable to various hardware setups depending on specific requirements and optimizations.


## Requirements
The repository requires several Python libraries.
As our code has been in use for a couple of years, it is generally compatible with the most recent versions of these libraries, and there is no strict version enforcement.
However, for your convenience, we provide the details of our verified testbed setup:

```
Python 3.9.16
torch 1.12.1
numpy 1.24.3
sklearn 1.2.2
tqdm 4.65.0
argparse 1.1
```

## Dataset
The datasets provided in the HD Zoo include most benchmarks used by my group and colleagues at institutions like UCSD, UCI, and UL Lafayette for HDC evaluation
You can find detailed information about these datasets in the [dataset](dataset) directory. However, due to the large size of these files, they are not included directly in the repository. Instead, you can download them from the following Google Drive link:
- [HD Zoo Datasets](https://drive.google.com/file/d/1cMtHl5S1TGt21W_IJCWaxJO8ngIHbXt2/view?usp=sharing)

The format supported by HD Zoo is the `choir_dat` format, which I initially developed for my ICCAD'21 paper ("Massive parallel, ...").
This format is designed to accommodate various datasets.
If you're interested in testing other datasets or extending the existing ones for your research, feel free to refer to the code in the [dataset utility script](hdzoo/utils/dataset.py) and modify it as needed for your purposes.

## Usage
Here is a simple example of how to use the HD Zoo framework.
To train a model using a dataset, you simply run the `main.py` script with the path to your training dataset.
There's no need to specify the testing dataset separately, as the code will locate the corresponding testing dataset automatically.
```
python main.py dataset/mnist_train.choir_dat
```

You have the option to change the encoding method. For example, if you want to use the `randomproj` encoding method, the command would be modified as follows:

```
python main.py --encoder randomproj dataset/mnist_train.choir_dat
```

For a comprehensive overview of all the options and configurations supported by the HD Zoo framework, please refer to the following full usage section.


### Full Usage
```
$ python main.py --help
usage: main.py [-h] [-d DIMENSIONS] [-i ITERATIONS] [-b BATCH_SIZE] [-lr LEARNING_RATE] [--normalizer {l2,minmax}] [-sp] [--encoder {idlevel,linear,nonlinear}] [-nb] [-q Q] [-m] [-s {dot,cos}] [-l LOGFILE] [-r RANDOM_SEED] filename

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
  --normalizer {l2,minmax}
                        set normalizer protocol
  -sp, --singlepass     use single pass(oneshot) training (some datasets may overfit with this
  --encoder {idlevel,randomproj,nonlinear}
                        sets encoding protocol
  -nb, --nonbin         do not apply binarze hypervectors in the selected encoding method
  -q Q, --quantization Q
                        sets quantization level for IDLEVEL encoder
  -m, --mass            use mass retraining
  -s {dot,cos}, --sim_metric {dot,cos}
                        set similarity metric
  -l LOGFILE, --logfile LOGFILE
                        set log file
  -r RANDOM_SEED, --randomseed RANDOM_SEED
                        set random see
```

## Supported Algorithms
The HD Zoo framework supports a variety of encoding algorithms and training methods, as outlined below. It's important to note that the names of these methods may vary across different papers.

- Conventional single-pass training
  - The single-pass training option of HD Zoo enables the really old-fashioned single-pass training -- it does not perform any similarity checks. So, it might result in very low accuracies.
  - A more effective (and state-of-the-art) single-pass learning would be to use the retraining method and run it for a single epoch, which can be done using the -i 1 option.
- Typical iterative training (a.k.a. retraining)
  - This method involves bundling hypervectors only for misclassified samples. The general idea was (probably first) proposed in:
    - Imani, Mohsen, Deqian Kong, Abbas Rahimi, and Tajana Rosing. "Voicehd: Hyperdimensional computing for efficient speech recognition." In 2017 IEEE international conference on rebooting computing (ICRC), pp. 1-8. IEEE, 2017.
  - If you want to look at the pseudocode for a detailed understanding, you may refer to this paper:
    - Kim, Yeseong, Mohsen Imani, and Tajana S. Rosing. "Efficient human activity recognition using hyperdimensional computing." In Proceedings of the 8th International Conference on the Internet of Things, pp. 1-6. 2018.
  - I'm not sure when we started using the learning rate, but at least the following paper would be one of our earlier works that introduced the learning rate.
    - Imani, Mohsen, Yeseong Kim, Sadegh Riazi, John Messerly, Patric Liu, Farinaz Koushanfar, and Tajana Rosing. "A framework for collaborative learning in secure high-dimensional space." In 2019 IEEE 12th International Conference on Cloud Computing (CLOUD), pp. 435-446. IEEE, 2019.
- ID-level encoding
  - Imani, Mohsen, Deqian Kong, Abbas Rahimi, and Tajana Rosing. "Voicehd: Hyperdimensional computing for efficient speech recognition." In 2017 IEEE international conference on rebooting computing (ICRC), pp. 1-8. IEEE, 2017.
- Random projection encoding
  - Imani, Mohsen, Yeseong Kim, Sadegh Riazi, John Messerly, Patric Liu, Farinaz Koushanfar, and Tajana Rosing. "A framework for collaborative learning in secure high-dimensional space." In 2019 IEEE 12th International Conference on Cloud Computing (CLOUD), pp. 435-446. IEEE, 2019.
- Nonlinear encoding
  - Imani, Mohsen, Saikishan Pampana, Saransh Gupta, Minxuan Zhou, Yeseong Kim, and Tajana Rosing. "Dual: Acceleration of clustering algorithms using digital-based processing in-memory." In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 356-371. IEEE, 2020.
- MASS retraining
  - Kim, Yeseong, Jiseung Kim, and Mohsen Imani. "Cascadehd: Efficient many-class learning framework using hyperdimensional computing." In 2021 58th ACM/IEEE Design Automation Conference (DAC), pp. 775-780. IEEE, 2021.

## Citation
This repository is compiled to support the experiments presented in our recent papers:

```
@inproceedings{kim2023efficient,
  title={Efficient Hyperdimensional Learning with Trainable, Quantizable, and Holistic Data Representation},
  author={Kim, Jiseung and Lee, Hyunsei and Imani, Mohsen and Kim, Yeseong},
  booktitle={2023 Design, Automation \& Test in Europe Conference \& Exhibition (DATE)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
@inproceedings{lee2023comprehensive,
  title={Comprehensive Integration of Hyperdimensional Computing with Deep Learning towards Neuro-Symbolic AI},
  author={Lee, Hyunsei and Kim, Jiseung and Chen, Hanning and Zeira, Ariela and Srinivasa, Narayan and Imani, Mohsen and Kim, Yeseong},
  booktitle={2023 60th ACM/IEEE Design Automation Conference (DAC)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```
