# Training Region-based Object Detectors with Online Hard Example Mining
By Abhinav Shrivastava, Abhinav Gupta, Ross Girshick

### Introduction
Online Hard Example Mining (OHEM) is an online bootstrapping algorithm for training region-based ConvNet object detectors like [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn). OHEM 
- works nicely in the Stochastic Gradient Descent (SGD) paradigm,
- simplifies training by removing some heuristics and hyperparameters,
- leads to better convergence (lower training set loss),
- consistently gives significanlty higher mAP on PASCAL VOC and MS COCO.

OHEM was initially presented at CVPR 2016 as an Oral Presentation. For more details, see the [arXiv tech report](http://arxiv.org/abs/1604.03540).

### License

This code is released under the MIT License (refer to the LICENSE file for details).

### Citing

If you find this useful in your research, please consider citing:

    @inproceedings{shrivastavaCVPR16ohem,
        Author = {Abhinav Shrivastava and Abhinav Gupta and Ross Girshick},
        Title = {Training Region-based Object Detectors with Online Hard Example Mining},
        Booktitle = {Conference on Computer Vision and Pattern Recognition ({CVPR})},
        Year = {2016}
    }

### Disclaimer

This implementation is built on a *fork* of Faster R-CNN Python code ([here](https://github.com/rbgirshick/py-faster-rcnn)), which in turn builds on Fast R-CNN ([here](https://github.com/rbgirshick/fast-rcnn)). Please cite the appropriate papers depending on which part of the code and/or model you are using.

### Results

|                        | training data                       | test data    | mAP (paper)   | mAP (this repo) |
|:--- | :--- | :--- | :--- | :--- |
|Fast R-CNN  (FRCN)      | VOC 07 trainval                     | VOC 07 test  | 66.9   | 67.6  |
|FRCN with OHEM          | VOC 07 trainval                     | VOC 07 test  | 69.9   | 71.5 |
|FRCN, +M, +B            | VOC 07 trainval                     | VOC 07 test  | 72.4   |  |
|FRCN with OHEM, +M, +B  | VOC 07 trainval                     | VOC 07 test  | 75.1   |  |
|FRCN                    | VOC 07 trainval + 12 trainval       | VOC 07 test  | 70.0   |  |
|FRCN with OHEM          | VOC 07 trainval + 12 trainval       | VOC 07 test  | 74.6   | 75.5 |
|FRCN with OHEM, +M, +B  | VOC 07 trainval + 12 trainval       | VOC 07 test  | 78.9   |  |
|FRCN                    | VOC 12 trainval                     | VOC 12 test  | 65.7   |  |
|FRCN with OHEM          | VOC 12 trainval                     | VOC 12 test  | 69.8   |  |
|FRCN with OHEM, +M, +B  | VOC 12 trainval                     | VOC 12 test  | 72.9   |  |
|FRCN                    | VOC 07 trainval&test + 12 trainval  | VOC 12 test  | 68.4   |  |
|FRCN with OHEM          | VOC 07 trainval&test + 12 trainval  | VOC 12 test  | 71.9   |  |
|FRCN with OHEM, +M, +B  | VOC 07 trainval&test + 12 trainval  | VOC 12 test  | 76.3   |  |
|FRCN with OHEM, +M, +B  |  *above* + COCO 14 trainval         | VOC 12 test  | 80.1   |  |

**Note**: All methods above use the VGG16 network. `mAP (paper)` is the mAP reported in the paper. `mAP (this repo)` is the mAP reproduced by this codebase.

**Legend**: `+M`: using multi-scale for training and testing, `+B`: multi-stage bounding box regression. See the paper for details.

### Released
- [x] Initial OHEM release

### Sometime in the future
- [ ] Support for Multi-scale training and testing
- [ ] Support for Multi-stage bounding box regression
- [ ] Scripts/models for results in [this Table](#results)
- [ ] Support for Faster R-CNN (see [below](#faq-regarding-faster-r-cnn-support))

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)
7. [FAQ regarding Faster R-CNN support](#faq-regarding-faster-r-cnn-support)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download Ross's [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`, `yaaml'
3. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

1. For training smaller networks (VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 4G of memory suffices
2. For training VGG16, you'll need a K40 or Titan X (or better).

### Installation (similar to Fast(er) R-CNN)

1. Clone the OHEM repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/abhi2610/ohem.git
  ```

2. We'll call the directory that you cloned OHEM into `OHEM_ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*

   **Note 1:** If you didn't clone OHEM with the `--recursive` flag, then you'll need to manually clone the `caffe-fast-rcnn` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `caffe-fast-rcnn` submodule needs to be on the `faster-rcnn` branch (or equivalent detached state). This will happen automatically *if you followed step 1 instructions*.

3. Build the Cython modules
    ```Shell
    cd $OHEM_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $OHEM_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

5. Download pre-computed Fast R-CNN detector trained with OHEM using VGG16 and VGG_CNN_M_1024 networks.
    ```Shell
    cd $OHEM_ROOT
    ./data/scripts/fetch_fast_rcnn_ohem_models.sh
    ```
    This will populate the `$OHEM_ROOT/data` folder with a `fast_rcnn_ohem_models` folder which contains VGG16 and VGG_CNN_M_1024 models (Fast R-CNN detectors trained with OHEM). 
    The format will be `fast_rcnn_ohem_models/TRAINING_SET/MODEL_FILE`.

*These models were re-trained using this codebase and achieve slightly better performance (see [this Table](#results)). In particular, on the standard split, VGG_CNN_M_1024 model gets 62.8 mAP (compared to 62.0 mAP reported in paper) and VGG16 model gets 71.5 mAP (compared to 69.9 mAP). All models from the paper will be released soon.*

### Installation for training and testing models
1. Download the training, validation, test data and VOCdevkit

  ```Shell
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
  ```

2. Extract all of these tars into one directory named `VOCdevkit`

  ```Shell
  tar xvf VOCtrainval_06-Nov-2007.tar
  tar xvf VOCtest_06-Nov-2007.tar
  tar xvf VOCdevkit_08-Jun-2007.tar
  ```

3. It should have this basic structure

  ```Shell
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2007                    # image sets, annotations, etc.
    # ... and several other directories ...
    ```

4. Create symlinks for the PASCAL VOC dataset

  ```Shell
    cd $OHEM_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
6. Follow the next sections to download pre-trained ImageNet models

*COCO instructions and models will be released soon.*

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the two networks described in the paper: VGG_CNN_M_1024 and VGG16.

```Shell
cd $OHEM_ROOT
./data/scripts/fetch_imagenet_models.sh
```
Models come from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but are provided here for your convenience..

### Usage

To train a Fast R-CNN detector using the **OHEM** algorithm on voc_2007_trainval, use `experiments/scripts/fast_rcnn_ohem.sh`. See `experiments/scripts/` directory for other scripts. Output is written underneath `$OHEM_ROOT/output`.

```Shell
cd $OHEM_ROOT
./experiments/scripts/fast_rcnn_ohem.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {VGG16, VGG_CNN_M_1024} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks with OHEM are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```

*The VGG_CNN_M_1024 model should get ~62.8 mAP and VGG16 model should get ~71.5 mAP. For reference, you can download my logs from [here](http://graphics.cs.cmu.edu/projects/ohem/data/logs.tgz).*

### FAQ regarding Faster R-CNN support

I have received a lot of queries regarding using OHEM with Faster R-CNN. I have not spent too much time combining OHEM with Faster R-CNN yet. Some researchers have informed me that OHEM works well in the 'alternating optimization' setup, but not so much with the 'end to end learning' setup. I hope to try and release the support for Faster R-CNN in the coming months. If you would like an update when I release it, send me an email. 

Also, the authors of [R-FCN](https://github.com/daijifeng001/R-FCN) succesfully used OHEM with R-FCN and Faster R-CNN; you might find their codebase helpful.
