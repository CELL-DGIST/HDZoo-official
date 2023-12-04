# HDC evaluation datasets

## Download
Please download and uncompress the achived file and locate them in the directory you want:
* https://drive.google.com/file/d/1cMtHl5S1TGt21W_IJCWaxJO8ngIHbXt2/view?usp=drive_link

```
tar zxvf hdzoo_dataset.tar.gz
mv hdzoo_dataset/*.choir_dat .
rmdir hdzoo_dataset
```


## Data format
The `choir_dat` format is designed to accommodate various datasets for classification tasks, which I initially developed for my ICCAD'21 paper at UCSD:
```
@inproceedings{kim2021massively,
  title={Massively Parallel Big Data Classification on a Programmable Processing In-Memory Architecture},
  author={Kim, Yeseong and Imani, Mohsen and Gupta, Saransh and Zhou, Minxuan and Rosing, Tajana S},
  booktitle={2021 IEEE/ACM International Conference On Computer Aided Design (ICCAD)},
  pages={1--9},
  year={2021},
  organization={IEEE}
}
```
- The binary data format described begins with metadata consisting of two 4-byte integers. The first integer represents the number of features (F), and the second represents the number of classes (K).
- Following the metadata, the format details each sample using 4 * (F+1) bytes. This allocation includes F features, each represented by a 4-byte floating point value, and is followed by a 4-byte integer label for each sample.
- If the data reaches the End Of File (EOF), it means the end of all samples in the dataset.

### How to read/write the dataset
- Read: Please take a look at the code included in this repo: [../hdzoo/utils/dataset.py](../hdzoo/utils/dataset.py).
- Write: If you want to make a custom dataset, please refer to the following simple example.
```
import struct
def writeDataSetForChoirSIM(X, y, filename):
  X, y = ds

  f = open(filename, "wb")
  nFeatures = len(X[0])
  nClasses = len(set(y))
  
  f.write(struct.pack('i', nFeatures))
  f.write(struct.pack('i', nClasses))
  for V, l in zip(X, y):
    for v in V:
        f.write(struct.pack('f', v))
    f.write(struct.pack('i', l))
```

## List of Datasets
For the details of the datasets and their references, please refer to the original source provided below.

- Emotion (emotion recognition): emotion_*.choir_dat
  - https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions
  ```
  @inproceedings{bird2019mental,
    title={Mental emotional sentiment classification with an eeg-based brain-machine interface},
    author={Bird, Jordan J and Ekart, Aniko and Buckingham, Christopher D and Faria, Diego R},
    booktitle={Proceedings of theInternational Conference on Digital Image and Signal Processing (DISP’19)},
    year={2019}
  }
  ```
- ExtraMany (smartphone sensor-based activity recognition): extramany_*.choir_dat
  - http://extrasensory.ucsd.edu/
  ```
    @article{vaizman2017recognizing,
    title={Recognizing detailed human context in the wild from smartphones and smartwatches},
    author={Vaizman, Yonatan and Ellis, Katherine and Lanckriet, Gert},
    journal={IEEE pervasive computing},
    volume={16},
    number={4},
    pages={62--74},
    year={2017},
    publisher={IEEE}
  }
  ```
- Human activity recognition: har_*.choir_dat
  - https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
  ```
  @inproceedings{anguita2013public,
    title={A public domain dataset for human activity recognition using smartphones.},
    author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge Luis and others},
    booktitle={Esann},
    volume={3},
    pages={3},
    year={2013}
  }
  ```
- Smartphone-based activity recognition, 12 classes: sa12_*.choir_dat
- Smartphone-based activity recognition, 6 classes: sa6_*.choir_dat
  - https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions
  ```
  @inproceedings{anguita2012human,
    title={Human activity recognition on smartphones using a multiclass hardware-friendly support vector machine},
    author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge L},
    booktitle={Ambient Assisted Living and Home Care: 4th International Workshop, IWAAL 2012, Vitoria-Gasteiz, Spain, December 3-5, 2012. Proceedings 4},
    pages={216--223},
    year={2012},
    organization={Springer}
  }
  ```
- PAMAP2 (IMU-based human activity recognition)
  - https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
  ```
  @inproceedings{reiss2012introducing,
    title={Introducing a new benchmarked dataset for activity monitoring},
    author={Reiss, Attila and Stricker, Didier},
    booktitle={2012 16th international symposium on wearable computers},
    pages={108--109},
    year={2012},
    organization={IEEE}
  }
  ```
  - (The dataset includes features obtained through a series of preprocessing suggested in the original author's paper.)
- ISOLET (voice recognition): isolet_*.choir_dat
  - https://archive.ics.uci.edu/dataset/54/isolet
  ```
    @misc{misc_isolet_54,
    author       = {Cole,Ron and Fanty,Mark},
    title        = {{ISOLET}},
    year         = {1994},
    howpublished = {UCI Machine Learning Repository},
    note         = {{DOI}: https://doi.org/10.24432/C51G69}
  }
  ```
- MNIST (human-written digit text recognition): mnist_*.choir_dat
  - http://yann.lecun.com/exdb/mnist/
  - The link above is not accessible at this moment; you can find the original source from everywhere :)
  ```
  @article{lecun1998gradient,
    title={Gradient-based learning applied to document recognition},
    author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
    journal={Proceedings of the IEEE},
    year={1998},
    publisher={IEEE}
  }
  ```
- Fashion MNIST (fashion image recognition): fmnist_*.choir_dat
  - https://github.com/zalandoresearch/fashion-mnist
  ```
    @online{xiao2017/online,
    author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
    title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
    date         = {2017-08-28},
    year         = {2017},
    eprintclass  = {cs.LG},
    eprinttype   = {arXiv},
    eprint       = {cs.LG/1708.07747},
  }
  ```
- MNIST Mix (MNIST-style images but 100 classes): mnistmix_*.choir_dat
  - https://github.com/jwwthu/MNIST-MIX
  ```
  @article{jiang2020mnist,
    title={MNIST-MIX: a multi-language handwritten digit recognition dataset},
    author={Jiang, Weiwei},
    journal={IOP SciNotes},
    volume={1},
    number={2},
    pages={025002},
    year={2020},
    publisher={IOP Publishing}
  }
  ```
- Caltech 10k Web Faces (face recognition): face_*.choir_dat
  - https://www.vision.caltech.edu/datasets/caltech_10k_webfaces/
  ```
  @inproceedings{angelova2005pruning,
    title={Pruning training sets for learning of object categories},
    author={Angelova, Anelia and Abu-Mostafam, Yaser and Perona, Pietro},
    booktitle={2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)},
    volume={1},
    pages={494--501},
    year={2005},
    organization={IEEE}
  }
  ```
  - The dataset includes features obtained by HOG preprocessing as initially shown in my paper:
  ```
  @inproceedings{kim2017orchard,
    title={Orchard: Visual object recognition accelerator based on approximate in-memory processing},
    author={Kim, Yeseong and Imani, Mohsen and Rosing, Tajana},
    booktitle={2017 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
    pages={25--32},
    year={2017},
    organization={IEEE}
  }
  ```
  - The negative samples are obtained from CIFAR100 and PASCAL VOC.
    - https://www.cs.toronto.edu/~kriz/cifar.html
    - https://host.robots.ox.ac.uk/pascal/VOC/
- Google AudioSet (Audio genre recognition): audioset_*.choir_dat
  - https://research.google.com/audioset/dataset/index.html
  ```
  @inproceedings{gemmeke2017audio,
    title={Audio set: An ontology and human-labeled dataset for audio events},
    author={Gemmeke, Jort F and Ellis, Daniel PW and Freedman, Dylan and Jansen, Aren and Lawrence, Wade and Moore, R Channing and Plakal, Manoj and Ritter, Marvin},
    booktitle={2017 IEEE international conference on acoustics, speech and signal processing (ICASSP)},
    pages={776--780},
    year={2017},
    organization={IEEE}
  }
  ```
  - The dataset includes features obtained through a series of preprocessing.
- Heart: heart_*.choir_dat
  - https://www.kaggle.com/datasets/shayanfazeli/heartbeat
   ```
   @inproceedings{kachuee2018ecg,
    title={Ecg heartbeat classification: A deep transferable representation},
    author={Kachuee, Mohammad and Fazeli, Shayan and Sarrafzadeh, Majid},
    booktitle={2018 IEEE International Conference on Healthcare Informatics (ICHI)},
    pages={443--444},
    year={2018},
    organization={IEEE}
  }
   ```
- MAR, SHA (Plant speciy classification), TEX: mar_*.choir_dat, sha_*.choir_dat, tex_*.choir_dat
  - https://archive.ics.uci.edu/dataset/241/one+hundred+plant+species+leaves+data+set
  ```
  @misc{misc_one-hundred_plant_species_leaves_data_set_241,
    author       = {Cope,James, Beghin,Thibaut, Remagnino,Paolo, and Barman,Sarah},
    title        = {{One-hundred plant species leaves data set}},
    year         = {2012},
    howpublished = {UCI Machine Learning Repository},
    note         = {{DOI}: https://doi.org/10.24432/C5RG76}
  }
  ```
