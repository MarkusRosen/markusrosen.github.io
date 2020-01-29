---
layout: post
title: Detectron2 - How to use Instance Image Segmentation for Window and Building Recognition
date: 2020-01-29 18:53:30 +0300
description: This tutorial teaches you how to implement instance image segmentation with a concrete example. 
img:  /post1/teaser.jpg
tags: [PyTorch, Deep Learning, NumPy, Visualization] 
---


# Tutorial on how to use instance image segmentation for window and building recognition

This small tutorial is targeted at researchers that have basic machine learning and Python programming skills that want to implement instance image segmentation for further use in their models. detectron2 is still under heavy development and as of January 2020 usable with Windows without some code changes [that are explained here ](https://github.com/InformationSystemsFreiburg/image_segmentation_japan). Instead of using detectron2 on a local machine, you can also use Google Colab and a free GPU from Google for your models. The GPU is either a Nvidia K80, T4, P4 or P100, all of which are powerfull enough to train detectron2 models. **Important note: Computation time on Google Colab is limited to 12 hours**.

The first part of this tutorials is based on the beginners tutorial of detectron2, the second part and third part come from the research stay of [Markus Rosenfelder](https://www.is.uni-freiburg.de/mitarbeiter-en/team/markus-rosenfelder) at [GCP NIES](https://www.cger.nies.go.jp/gcp/) in Tsukuba.

## Part 1 Installation and setup

### Installation within Google Colab 


First, check what kind of GPU Google is providing you in the current session. You will find the GPU model name in the third row of the first column. 




```python
!nvidia-smi
```

    Wed Jan 29 07:35:39 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 440.44       Driver Version: 418.67       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   38C    P8     9W /  70W |      0MiB / 15079MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    

Now onto the installation of all needed packages. Please keep in mind that this will take a few minutes.


```python
# install dependencies:
# (use +cu100 because colab is on CUDA 10.0)
!pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html 
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
import torch, torchvision
torch.__version__
!gcc --version
# opencv is pre-installed on colab
```

    Looking in links: https://download.pytorch.org/whl/torch_stable.html
    Requirement already up-to-date: torch==1.4+cu100 in /usr/local/lib/python3.6/dist-packages (1.4.0+cu100)
    Requirement already up-to-date: torchvision==0.5+cu100 in /usr/local/lib/python3.6/dist-packages (0.5.0+cu100)
    Requirement already satisfied, skipping upgrade: six in /usr/local/lib/python3.6/dist-packages (from torchvision==0.5+cu100) (1.12.0)
    Requirement already satisfied, skipping upgrade: numpy in /usr/local/lib/python3.6/dist-packages (from torchvision==0.5+cu100) (1.17.5)
    Requirement already satisfied, skipping upgrade: pillow>=4.1.1 in /usr/local/lib/python3.6/dist-packages (from torchvision==0.5+cu100) (6.2.2)
    Requirement already satisfied: cython in /usr/local/lib/python3.6/dist-packages (0.29.14)
    Requirement already satisfied: pyyaml==5.1 in /usr/local/lib/python3.6/dist-packages (5.1)
    Collecting git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
      Cloning https://github.com/cocodataset/cocoapi.git to /tmp/pip-req-build-h76zwqfl
      Running command git clone -q https://github.com/cocodataset/cocoapi.git /tmp/pip-req-build-h76zwqfl
    Requirement already satisfied, skipping upgrade: setuptools>=18.0 in /usr/local/lib/python3.6/dist-packages (from pycocotools==2.0) (42.0.2)
    Requirement already satisfied, skipping upgrade: cython>=0.27.3 in /usr/local/lib/python3.6/dist-packages (from pycocotools==2.0) (0.29.14)
    Requirement already satisfied, skipping upgrade: matplotlib>=2.1.0 in /usr/local/lib/python3.6/dist-packages (from pycocotools==2.0) (3.1.2)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.1.0->pycocotools==2.0) (2.6.1)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.1.0->pycocotools==2.0) (0.10.0)
    Requirement already satisfied, skipping upgrade: numpy>=1.11 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.1.0->pycocotools==2.0) (1.17.5)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.1.0->pycocotools==2.0) (2.4.6)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.1.0->pycocotools==2.0) (1.1.0)
    Requirement already satisfied, skipping upgrade: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil>=2.1->matplotlib>=2.1.0->pycocotools==2.0) (1.12.0)
    Building wheels for collected packages: pycocotools
      Building wheel for pycocotools (setup.py) ... [?25l[?25hdone
      Created wheel for pycocotools: filename=pycocotools-2.0-cp36-cp36m-linux_x86_64.whl size=275266 sha256=045c93eaa57ba3a00c13ea4b3431474ca59b726b0cfaeb1ef76fb18535ef7b2f
      Stored in directory: /tmp/pip-ephem-wheel-cache-p_ebvnas/wheels/90/51/41/646daf401c3bc408ff10de34ec76587a9b3ebfac8d21ca5c3a
    Successfully built pycocotools
    Installing collected packages: pycocotools
      Found existing installation: pycocotools 2.0
        Uninstalling pycocotools-2.0:
          Successfully uninstalled pycocotools-2.0
    Successfully installed pycocotools-2.0
    



    gcc (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0
    Copyright (C) 2017 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    
    


```python
# install detectron2:
!git clone https://github.com/facebookresearch/detectron2 detectron2_repo
!pip install -e detectron2_repo
```

    fatal: destination path 'detectron2_repo' already exists and is not an empty directory.
    Obtaining file:///content/detectron2_repo
    Requirement already satisfied: termcolor>=1.1 in /usr/local/lib/python3.6/dist-packages (from detectron2==0.1) (1.1.0)
    Requirement already satisfied: Pillow==6.2.2 in /usr/local/lib/python3.6/dist-packages (from detectron2==0.1) (6.2.2)
    Requirement already satisfied: yacs>=0.1.6 in /usr/local/lib/python3.6/dist-packages (from detectron2==0.1) (0.1.6)
    Requirement already satisfied: tabulate in /usr/local/lib/python3.6/dist-packages (from detectron2==0.1) (0.8.6)
    Requirement already satisfied: cloudpickle in /usr/local/lib/python3.6/dist-packages (from detectron2==0.1) (1.2.2)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from detectron2==0.1) (3.1.2)
    Requirement already satisfied: tqdm>4.29.0 in /usr/local/lib/python3.6/dist-packages (from detectron2==0.1) (4.42.0)
    Requirement already satisfied: tensorboard in /usr/local/lib/python3.6/dist-packages (from detectron2==0.1) (1.15.0)
    Requirement already satisfied: fvcore in /usr/local/lib/python3.6/dist-packages (from detectron2==0.1) (0.1.dev200114)
    Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from detectron2==0.1) (0.16.0)
    Requirement already satisfied: pydot in /usr/local/lib/python3.6/dist-packages (from detectron2==0.1) (1.3.0)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.6/dist-packages (from yacs>=0.1.6->detectron2==0.1) (5.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->detectron2==0.1) (2.4.6)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->detectron2==0.1) (1.1.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->detectron2==0.1) (2.6.1)
    Requirement already satisfied: numpy>=1.11 in /usr/local/lib/python3.6/dist-packages (from matplotlib->detectron2==0.1) (1.17.5)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->detectron2==0.1) (0.10.0)
    Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.6/dist-packages (from tensorboard->detectron2==0.1) (0.9.0)
    Requirement already satisfied: wheel>=0.26; python_version >= "3" in /usr/local/lib/python3.6/dist-packages (from tensorboard->detectron2==0.1) (0.33.6)
    Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard->detectron2==0.1) (42.0.2)
    Requirement already satisfied: grpcio>=1.6.3 in /usr/local/lib/python3.6/dist-packages (from tensorboard->detectron2==0.1) (1.15.0)
    Requirement already satisfied: protobuf>=3.6.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard->detectron2==0.1) (3.10.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.6/dist-packages (from tensorboard->detectron2==0.1) (0.16.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.6/dist-packages (from tensorboard->detectron2==0.1) (3.1.1)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.6/dist-packages (from tensorboard->detectron2==0.1) (1.12.0)
    Requirement already satisfied: portalocker in /usr/local/lib/python3.6/dist-packages (from fvcore->detectron2==0.1) (1.5.2)
    Installing collected packages: detectron2
      Found existing installation: detectron2 0.1
        Can't uninstall 'detectron2'. No files were found to uninstall.
      Running setup.py develop for detectron2
    Successfully installed detectron2
    


```python
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
```

### Running a pretrained model

In this chapter we will run inference on a pretrained and prelabeled model to see if our setup has been succesful so far. For this we will download an image and run inference on it.


```python
!wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
im = cv2.imread("./input.jpg")
cv2_imshow(im)
```

    --2020-01-29 06:12:27--  http://images.cocodataset.org/val2017/000000439715.jpg
    Resolving images.cocodataset.org (images.cocodataset.org)... 52.216.185.99
    Connecting to images.cocodataset.org (images.cocodataset.org)|52.216.185.99|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 209222 (204K) [image/jpeg]
    Saving to: ‘input.jpg’
    
    input.jpg           100%[===================>] 204.32K  --.-KB/s    in 0.09s   
    
    2020-01-29 06:12:28 (2.32 MB/s) - ‘input.jpg’ saved [209222/209222]
    
    


![png](output_10_1.png)



```python
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
```

    model_final_f10217.pkl: 178MB [00:10, 16.6MB/s]                           
    


```python
# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
outputs["instances"].pred_classes
outputs["instances"].pred_boxes
```




    Boxes(tensor([[126.6035, 244.8977, 459.8291, 480.0000],
            [251.1083, 157.8127, 338.9731, 413.6379],
            [114.8496, 268.6864, 148.2352, 398.8111],
            [  0.8217, 281.0327,  78.6072, 478.4210],
            [ 49.3954, 274.1229,  80.1545, 342.9808],
            [561.2248, 271.5816, 596.2755, 385.2552],
            [385.9072, 270.3125, 413.7130, 304.0397],
            [515.9295, 278.3744, 562.2792, 389.3802],
            [335.2409, 251.9167, 414.7491, 275.9375],
            [350.9300, 269.2060, 386.0984, 297.9081],
            [331.6292, 230.9996, 393.2759, 257.2009],
            [510.7349, 263.2656, 570.9865, 295.9194],
            [409.0841, 271.8646, 460.5582, 356.8722],
            [506.8767, 283.3257, 529.9403, 324.0392],
            [594.5663, 283.4820, 609.0577, 311.4124]], device='cuda:0'))




```python
# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(v.get_image()[:, :, ::-1])
```


![png](output_13_0.png)


## Part 2 - Training and Inferencing (detecting windows and buildings)

### Creating a custom dataset with VGG Image Annotator
Before continuing with programming, one important step has to be learned: how to annotate images efficiently. For this, we will use the [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/via.html). Please keep in mind that you can also [download the annotator](http://www.robots.ox.ac.uk/~vgg/software/via/) for free and use it from your local machine.

On your local machine, do the following:

- Create a folder named `buildings`
- within this folder, create two folders: `val` and `train`
- Open the `VGG Annotator` either locally or via the URL mentioned above.
- Short introductions on how to use the tool:
  - Go to settings and specify the default path to where your train folder is located, example: `../data/buildings/train/` 
  - create a new attributes called `class`
  - set this attribute to `checkbox`
  - add `building` and `window` as options to `class`
- save the project
- copy images to the `train` and `val` folders
- import the images to the `VGG Annotator`
- zoom into the image with `CTRL` + `Mousewheel`
- select the `polygon region shape` tool and start with marking the `windows`
- after a polygon is finished, press `Enter` to save it
- after all `window` polygons are created, create the `building` polygons
- press `Spacebar` to open the annotations
- specify the correct `class` to each polygon
- after an image is done, save the project
- after all images are done, export the annotations to `train` as .json files and rename them to `via_region_data.json`
- do all of the above steps also for the validation data

For all the pictures on which you want to run inference on and want to keep a specific building ID provided by external data:

- open the `validation project file`
- add import all images as mentioned above
- add a new category `tagged_id`
- the new category should be of type `text`
- in the approximate center of each building:
  - create a `point` with the point-tool
  - fill the `tagged_id` with the respective ID from the other data set
- do this for all pictures and buildings

The resulting annotated image should look similar to below. An image from Shinagawa takes on average 20-60 minutes to annotate, depending on the amount of buildings and windows.

![VGG example](https://github.com/InformationSystemsFreiburg/image_segmentation_japan/raw/master/vgg_annotator.png)

### Download the data from GitHub

For this tutorial we will use the 114 images already annotated from the [GitHub repository](https://github.com/InformationSystemsFreiburg/image_segmentation_japan). We use `wget` and `unzip` to download the data and unzip it. The data is now located at `/content/train/` and `/content/val/`.


```python
!wget https://github.com/InformationSystemsFreiburg/image_segmentation_japan/raw/master/buildings.zip
!unzip buildings.zip
```

    --2020-01-29 06:12:57--  https://github.com/InformationSystemsFreiburg/image_segmentation_japan/raw/master/buildings.zip
    Resolving github.com (github.com)... 192.30.253.113
    Connecting to github.com (github.com)|192.30.253.113|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://raw.githubusercontent.com/InformationSystemsFreiburg/image_segmentation_japan/master/buildings.zip [following]
    --2020-01-29 06:12:58--  https://raw.githubusercontent.com/InformationSystemsFreiburg/image_segmentation_japan/master/buildings.zip
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 23618597 (23M) [application/zip]
    Saving to: ‘buildings.zip’
    
    buildings.zip       100%[===================>]  22.52M  38.2MB/s    in 0.6s    
    
    2020-01-29 06:12:59 (38.2 MB/s) - ‘buildings.zip’ saved [23618597/23618597]
    
    Archive:  buildings.zip
       creating: train/
      inflating: train/0000002_0003951_0000003_0000759.jpg  
      inflating: train/0004-0143409_p.jpg  
      inflating: train/0004-0143410_i.jpg  
      inflating: train/0004-0143486_p.jpg  
      inflating: train/0004-0143599_i.jpg  
      inflating: train/0004-0143600_p.jpg  
      inflating: train/0004-0143721_i.jpg  
      inflating: train/0004-0144297_p.jpg  
      inflating: train/0004-0144323_p.jpg  
      inflating: train/0004-0144571_p.jpg  
      inflating: train/0004-0144953_p.jpg  
      inflating: train/0004-0144965_p.jpg  
      inflating: train/0004-0145282_p.jpg  
      inflating: train/0004-0145309_p.jpg  
      inflating: train/0004-0145992_i.jpg  
      inflating: train/0004-0146648_p.jpg  
      inflating: train/0004-0147248_p.jpg  
      inflating: train/0004-0147764_i.jpg  
      inflating: train/0004-0148364_i.jpg  
      inflating: train/0004-0150203_p.jpg  
      inflating: train/0004-0151205_i.jpg  
      inflating: train/0004-0151294_p.jpg  
      inflating: train/0004-0151743_i.jpg  
      inflating: train/0004-0152440_p.jpg  
      inflating: train/0004-0153237_p.jpg  
      inflating: train/0004-0153241_p.jpg  
      inflating: train/0004-0154048_i.jpg  
      inflating: train/0005-0189008_p.jpg  
      inflating: train/0005-0189263_p.jpg  
      inflating: train/0005-0189629_p.jpg  
      inflating: train/0005-0189911_i.jpg  
      inflating: train/0005-0190651_p.jpg  
      inflating: train/0005-0190658_i.jpg  
      inflating: train/0005-0191718_p.jpg  
      inflating: train/0005-0191779_i.jpg  
      inflating: train/0005-0192885_p.jpg  
      inflating: train/0005-0194004_p.jpg  
      inflating: train/0005-0194440_i.jpg  
      inflating: train/0009-0322901_p.jpg  
      inflating: train/0009-0323309_i.jpg  
      inflating: train/0009-0323811_p.jpg  
      inflating: train/0009-0326089_p.jpg  
      inflating: train/0009-0326090_i.jpg  
      inflating: train/0009-0326343_p.jpg  
      inflating: train/0009-0326875_p.jpg  
      inflating: train/0010-0283861_p.jpg  
      inflating: train/0010-0284448_i.jpg  
      inflating: train/0010-0285340_i.jpg  
      inflating: train/0010-0285981_p.jpg  
      inflating: train/0010-0286248_i.jpg  
      inflating: train/0010-0286650_p.jpg  
      inflating: train/0010-0287145_p.jpg  
      inflating: train/0010-0287146_p.jpg  
      inflating: train/0010-0287868_p.jpg  
      inflating: train/0010-0289016_p.jpg  
      inflating: train/0011-0267177_p.jpg  
      inflating: train/0011-0267245_i.jpg  
      inflating: train/0011-0271548_p.jpg  
      inflating: train/0011-0271633_p.jpg  
      inflating: train/0012-0131277_p.jpg  
      inflating: train/0012-0131914_p.jpg  
      inflating: train/0012-0132476_i.jpg  
      inflating: train/0012-0132657_i.jpg  
      inflating: train/0012-0132667_i.jpg  
      inflating: train/0012-0132701_p.jpg  
      inflating: train/0012-0134139_i.jpg  
      inflating: train/0012-0134179_i.jpg  
      inflating: train/0012-0134906_p.jpg  
      inflating: train/0012-0134914_i.jpg  
      inflating: train/0012-0135392_p.jpg  
      inflating: train/0012-0135467_p.jpg  
      inflating: train/0015-0287179_p.jpg  
      inflating: train/0015-0287260_p.jpg  
      inflating: train/0015-0287264_i.jpg  
      inflating: train/0015-0287892_i.jpg  
      inflating: train/0015-0287986_p.jpg  
      inflating: train/0015-0290731_p.jpg  
      inflating: train/0015-0290736_i.jpg  
      inflating: train/0016-0195247_p.jpg  
      inflating: train/0016-0303591_p.jpg  
      inflating: train/0016-0303610_p.jpg  
      inflating: train/0016-0303927_p.jpg  
      inflating: train/0016-0304520_p.jpg  
      inflating: train/0016-0304721_i.jpg  
      inflating: train/0016-0304815_p.jpg  
      inflating: train/0016-0305415_i.jpg  
      inflating: train/0016-0305494_p.jpg  
      inflating: train/0018-0164411_i.jpg  
      inflating: train/0018-0165849_p.jpg  
      inflating: train/0020-0076136_p.jpg  
      inflating: train/0020-0092512_p.jpg  
      inflating: train/0020-0092660_p.jpg  
      inflating: train/0023-0045587_p.jpg  
      inflating: train/0028-0120716_p.jpg  
      inflating: train/0028-0121152_i.jpg  
      inflating: train/0028-0121252_i.jpg  
      inflating: train/0028-0121932_p.jpg  
      inflating: train/0028-0122282_i.jpg  
      inflating: train/0028-0122871_p.jpg  
      inflating: train/0028-0140606_i.jpg  
      inflating: train/0028-0141771_p.jpg  
      inflating: train/0028-0143120_p.jpg  
      inflating: train/0028-0175614_p.jpg  
      inflating: train/DvR4T-XKUa2ehD-3uTTgtx.jpg  
      inflating: train/dvRYMdfB2oz8roEaNB-2tg.jpg  
      inflating: train/DWCf628rMiMc6e5Xo18Li0.jpg  
      inflating: train/dwiM-W406o017ahrjL7c5Q.jpg  
      inflating: train/DwKaDnrqAKGehLP0rQ5CjQ.jpg  
      inflating: train/dzMdPtKTM4vUm7ORPBpHKA.jpg  
      inflating: train/E0U59p9aeDAqGF_J8rB-0w.jpg  
      inflating: train/E1h7MaFb_m6XHDmhferbCA.jpg  
      inflating: train/e81tnLN1q3ianW2zN63J8A.jpg  
      inflating: train/EapSyJQ3Lrs_Bqd8T8R4RA.jpg  
      inflating: train/ehBkKdzLmKXfj_kQ_F4hRA.jpg  
      inflating: train/EKdZvzuLxMBbjO65GofCGw.jpg  
      inflating: train/El8WLGC4oJNsHvt-8MTtag.jpg  
      inflating: train/ELaubdmbGTIH2c37lPgrJQ.jpg  
      inflating: train/emJPb45MbiDp_TN_1HSkDg.jpg  
      inflating: train/kvOglx8V4dJ3Ml4cxvJA-Q.jpg  
      inflating: train/l16eufKFroW5L_xhYSoqog.jpg  
      inflating: train/l1fb1pkKRpx4NORDhc0tgA.jpg  
      inflating: train/l1m5JxqY1MeLiloKKNaVUQ.jpg  
      inflating: train/l4hRIDsp5KAWl63QLETbaw.jpg  
      inflating: train/l9WJ_ScpoDGaSHjnXWR1DA.jpg  
      inflating: train/LDSLFXxdvOKgF_eDRD0qiw.jpg  
      inflating: train/LflfUcthpatXpBowaxYufg.jpg  
      inflating: train/mWaPrwkyWaJj0lnhtqwT3A.jpg  
      inflating: train/Us2r5l4yLMg-kEO0OAgAeQ.jpg  
      inflating: train/uTflbFq291GpKfst1dex1A.jpg  
      inflating: train/via_region_data.json  
       creating: val/
      inflating: val/p0BLyIbSaWEhftL1Cuasaw.jpg  
      inflating: val/P2VBr3KZuYBOF1QA-DIRrQ.jpg  
      inflating: val/p2YsGfPOCT8FXYvjta4row.jpg  
      inflating: val/P3dr3y0NZpnQsTcjIW-x9U.jpg  
      inflating: val/p4w0rP6eCcUI3w3htiMm3g.jpg  
      inflating: val/P4we3TZeMBwNVD2Q8KyegQ.jpg  
      inflating: val/p4X8gwA8Q15OI9ZUHrTJZg.jpg  
      inflating: val/p4yU2_tWslUKT24cW54oyg.jpg  
      inflating: val/p4zCd5huZE35tEWakIhZv6.jpg  
      inflating: val/p4zxq9hYv_OWvCwoy83HJA.jpg  
      inflating: val/P5BoM4S8cmOBMS0kxjNgBQ.jpg  
     extracting: val/P6PAuiLcfV7cWvpnMZ0ijg.jpg  
      inflating: val/p7Mu8DjZsfEomtAVPDy0hw.jpg  
      inflating: val/P7W1YnIB99tOroMW4BVQNQ.jpg  
      inflating: val/P7zqBvtQSjv16d7s2MZjiA.jpg  
      inflating: val/P8eVIvM6OscXlR5SR6Z05w.jpg  
      inflating: val/p8KcAK9bYM1MhB6m1LZBzA.jpg  
      inflating: val/p9LjGq-iOOegdkOiA4JRPQ.jpg  
      inflating: val/p9MdBwk8aMwfvNYVdsrzwQ.jpg  
      inflating: val/p9pRPA0o-i7xU2EXelvqvA.jpg  
      inflating: val/p9r3VS5NDbvjErpX9tVxbQ.jpg  
      inflating: val/P9x_OoS8_cN6gAFBfWqZ4Q.jpg  
      inflating: val/p9ZXCIT-rMlLwGC-BxRHNA.jpg  
      inflating: val/PAJ-MaBAsXToem1f8tQXQA.jpg  
      inflating: val/Pb73PiqChAM_VjnM6-mByA.jpg  
      inflating: val/PBq3z7XcxIDwKfaRSSieqA.jpg  
      inflating: val/PbQB6gupFxRsFwR9LljW_Q.jpg  
      inflating: val/Pbu3KVTnBvhRxFuRU7mQDg.jpg  
      inflating: val/PBwMzmcTJXTlPcpdemsZNA.jpg  
      inflating: val/PcEaOBL5k5V2-GV6TRcoIA.jpg  
      inflating: val/p_pGU7tYYIfdfimFqyD_Is.jpg  
      inflating: val/via_region_data.json  
      inflating: val/via_region_data__old.json  
      inflating: val/_0CfAgwU4GRi9S6yzBbnxQ.jpg  
      inflating: val/_1COpxhN8GOKkjj0fV8yrg.jpg  
      inflating: val/_40EqDxSm7VfFa3loCybQA.jpg  
      inflating: val/_6qHHdUHjbyLGdwqfTZgRA.jpg  
      inflating: val/_CrNw7UUGzt7WknYd63I4w.jpg  
      inflating: val/_FczOQH2Gps-DKWQb9aLlw.jpg  
      inflating: val/_hJfLUBz3M0MQXMz_5afWw.jpg  
      inflating: val/_IiT38b3e9Sy5Pq1B49p4s.jpg  
      inflating: val/_k84RoRK92PvZFj4AWcvWw.jpg  
      inflating: val/_lQKTfgy5oLfrnDqnb7_ng.jpg  
      inflating: val/_ObTdbCdQQTHL1kYnT7dyg.jpg  
      inflating: val/_Odc9YG8R9RXBkBE9vRNPw.jpg  
      inflating: val/_QCaqWVWeYXZWIUK_0Wbag.jpg  
      inflating: val/_qOoozLWFdlXfwI4lbFKCw.jpg  
      inflating: val/_Ujvl8m3THd78AyUC6we8w.jpg  
      inflating: val/_VOe8cqDxWY8vhZ6j_rMeA.jpg  
      inflating: val/_w6T3cr3IYC2PmcGvmyvMg.jpg  
      inflating: val/_YBDP9H7rtRL8ooFlNKGew.jpg  
    

### Import more packages

For the following code, we need to import additional packages


```python
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import random
from datetime import datetime
import pickle
from pathlib import Path
from tqdm import tqdm

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
```

### Read the output `JSON`-file from the VGG Image Annotator

This function is needed to read the annotations for all the images correctly and convert into a format that is usable by `detectron2`. If you have additional information, like for example a `building id` or other target classes, you need to change the function accordingly.


```python
def get_building_dicts(img_dir):
    """This function loads the JSON file created with the annotator and converts it to 
    the detectron2 metadata specifications.
    """
    # load the JSON file
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    # loop through the entries in the JSON file
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        # add file_name, image_id, height and width information to the records
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]

        objs = []
        # one image can have multiple annotations, therefore this loop is needed
        for annotation in annos:
            # reformat the polygon information to fit the specifications
            anno = annotation["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            region_attributes = annotation["region_attributes"]["class"]

            # specify the category_id to match with the class.

            if "building" in region_attributes:
                category_id = 1
            elif "window" in region_attributes:
                category_id = 0

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
                "iscrowd": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
```

### Prepare the data

The data needs to be loaded and prepared, therefore this code is executed for the training and validation data.


```python
# the data has to be registered within detectron2, once for the train and once for
# the val data
for d in ["train", "val"]:
    DatasetCatalog.register(
        "buildings_" + d,
        lambda d=d: get_building_dicts("/content/" + d),
    )

building_metadata = MetadataCatalog.get("buildings_train")

dataset_dicts = get_building_dicts("/content/train")
```

### View the input data

To check if everything is working as intended, you can view some of the input data before continuing. You should see two images, with bounding boxes around windows and buildings, where the buildings have a `1` as category and windows a `0`. Try it a few times, if you have images in your `JSON`-annotation file that you have not yet annotated, they will show without any annotations. These images will be skipped in the training.


```python
for i, d in enumerate(random.sample(dataset_dicts, 2)):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=building_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)

    cv2_imshow(vis.get_image()[:, :, ::-1])
    # if you want to save the files, uncomment the line below, but keep in mind that 
    # the folder inputs has to be created first
    # plt.savefig(f"./inputs/input_{i}.jpg")
```


![png](output_27_0.png)



![png](output_27_1.png)


### Configure the `detectron2` model

Now we need to configure our `detectron2` model before we can start training. There are more possible parameters to configure, for more information you can visit the [detectron2 documentation](https://detectron2.readthedocs.io/modules/config.html).


```python
cfg = get_cfg()
# you can choose alternative models as backbone here
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
))

cfg.DATASETS.TRAIN = ("buildings_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
# if you changed the model above, you need to adapt the following line as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR, 0.00025 seems a good start
cfg.SOLVER.MAX_ITER = (
    1000  # 1000 iterations is a good start, for better accuracy increase this value
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    512  # (default: 512), select smaller if faster training is needed
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # for the two classes window and building

```

### Start training

The next four lines of code create an output directory, a `trainer` and start training. If you only want to inference with an existing model, skip these four lines.


```python
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

    [32m[01/29 06:13:44 d2.engine.defaults]: [0mModel:
    GeneralizedRCNN(
      (backbone): FPN(
        (fpn_lateral2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (fpn_output2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (fpn_lateral3): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (fpn_output3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (fpn_lateral4): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        (fpn_output4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (fpn_lateral5): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        (fpn_output5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (top_block): LastLevelMaxPool()
        (bottom_up): ResNet(
          (stem): BasicStem(
            (conv1): Conv2d(
              3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
              (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
            )
          )
          (res2): Sequential(
            (0): BottleneckBlock(
              (shortcut): Conv2d(
                64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv1): Conv2d(
                64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv2): Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv3): Conv2d(
                64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
            )
            (1): BottleneckBlock(
              (conv1): Conv2d(
                256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv2): Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv3): Conv2d(
                64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
            )
            (2): BottleneckBlock(
              (conv1): Conv2d(
                256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv2): Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
              )
              (conv3): Conv2d(
                64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
            )
          )
          (res3): Sequential(
            (0): BottleneckBlock(
              (shortcut): Conv2d(
                256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv1): Conv2d(
                256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv2): Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv3): Conv2d(
                128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
            )
            (1): BottleneckBlock(
              (conv1): Conv2d(
                512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv2): Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv3): Conv2d(
                128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
            )
            (2): BottleneckBlock(
              (conv1): Conv2d(
                512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv2): Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv3): Conv2d(
                128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
            )
            (3): BottleneckBlock(
              (conv1): Conv2d(
                512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv2): Conv2d(
                128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=128, eps=1e-05)
              )
              (conv3): Conv2d(
                128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
            )
          )
          (res4): Sequential(
            (0): BottleneckBlock(
              (shortcut): Conv2d(
                512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
              (conv1): Conv2d(
                512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (1): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (2): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (3): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (4): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (5): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (6): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (7): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (8): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (9): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (10): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (11): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (12): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (13): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (14): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (15): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (16): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (17): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (18): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (19): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (20): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (21): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
            (22): BottleneckBlock(
              (conv1): Conv2d(
                1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv2): Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
              )
              (conv3): Conv2d(
                256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=1024, eps=1e-05)
              )
            )
          )
          (res5): Sequential(
            (0): BottleneckBlock(
              (shortcut): Conv2d(
                1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
              )
              (conv1): Conv2d(
                1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv2): Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv3): Conv2d(
                512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
              )
            )
            (1): BottleneckBlock(
              (conv1): Conv2d(
                2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv2): Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv3): Conv2d(
                512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
              )
            )
            (2): BottleneckBlock(
              (conv1): Conv2d(
                2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv2): Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=512, eps=1e-05)
              )
              (conv3): Conv2d(
                512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False
                (norm): FrozenBatchNorm2d(num_features=2048, eps=1e-05)
              )
            )
          )
        )
      )
      (proposal_generator): RPN(
        (anchor_generator): DefaultAnchorGenerator(
          (cell_anchors): BufferList()
        )
        (rpn_head): StandardRPNHead(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (objectness_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
          (anchor_deltas): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (roi_heads): StandardROIHeads(
        (box_pooler): ROIPooler(
          (level_poolers): ModuleList(
            (0): ROIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=0, aligned=True)
            (1): ROIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=0, aligned=True)
            (2): ROIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=0, aligned=True)
            (3): ROIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=0, aligned=True)
          )
        )
        (box_head): FastRCNNConvFCHead(
          (fc1): Linear(in_features=12544, out_features=1024, bias=True)
          (fc2): Linear(in_features=1024, out_features=1024, bias=True)
        )
        (box_predictor): FastRCNNOutputLayers(
          (cls_score): Linear(in_features=1024, out_features=3, bias=True)
          (bbox_pred): Linear(in_features=1024, out_features=8, bias=True)
        )
        (mask_pooler): ROIPooler(
          (level_poolers): ModuleList(
            (0): ROIAlign(output_size=(14, 14), spatial_scale=0.25, sampling_ratio=0, aligned=True)
            (1): ROIAlign(output_size=(14, 14), spatial_scale=0.125, sampling_ratio=0, aligned=True)
            (2): ROIAlign(output_size=(14, 14), spatial_scale=0.0625, sampling_ratio=0, aligned=True)
            (3): ROIAlign(output_size=(14, 14), spatial_scale=0.03125, sampling_ratio=0, aligned=True)
          )
        )
        (mask_head): MaskRCNNConvUpsampleHead(
          (mask_fcn1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (mask_fcn2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (mask_fcn3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (mask_fcn4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (deconv): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
          (predictor): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    [32m[01/29 06:13:45 d2.data.build]: [0mRemoved 15 images with no usable annotations. 114 images left.
    [32m[01/29 06:13:45 d2.data.detection_utils]: [0mTransformGens used in training: [ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'), RandomFlip()]
    [32m[01/29 06:13:45 d2.data.build]: [0mUsing training sampler TrainingSampler
    


<p style="color: red;">
The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.<br>
We recommend you <a href="https://www.tensorflow.org/guide/migrate" target="_blank">upgrade</a> now 
or ensure your notebook will continue to use TensorFlow 1.x via the <code>%tensorflow_version 1.x</code> magic:
<a href="https://colab.research.google.com/notebooks/tensorflow_version.ipynb" target="_blank">more info</a>.</p>



    model_final_a3ec72.pkl: 254MB [00:15, 16.3MB/s]                           
    'roi_heads.box_predictor.cls_score.weight' has shape (81, 1024) in the checkpoint but (3, 1024) in the model! Skipped.
    'roi_heads.box_predictor.cls_score.bias' has shape (81,) in the checkpoint but (3,) in the model! Skipped.
    'roi_heads.box_predictor.bbox_pred.weight' has shape (320, 1024) in the checkpoint but (8, 1024) in the model! Skipped.
    'roi_heads.box_predictor.bbox_pred.bias' has shape (320,) in the checkpoint but (8,) in the model! Skipped.
    'roi_heads.mask_head.predictor.weight' has shape (80, 256, 1, 1) in the checkpoint but (2, 256, 1, 1) in the model! Skipped.
    'roi_heads.mask_head.predictor.bias' has shape (80,) in the checkpoint but (2,) in the model! Skipped.
    

    [32m[01/29 06:14:06 d2.engine.train_loop]: [0mStarting training from iteration 0
    [32m[01/29 06:14:18 d2.utils.events]: [0meta: 0:09:23  iter: 19  total_loss: 4.911  loss_cls: 0.972  loss_box_reg: 0.278  loss_mask: 0.693  loss_rpn_cls: 2.737  loss_rpn_loc: 0.240  time: 0.6119  data_time: 0.1153  lr: 0.000005  max_mem: 4338M
    [32m[01/29 06:14:31 d2.utils.events]: [0meta: 0:09:26  iter: 39  total_loss: 3.581  loss_cls: 0.911  loss_box_reg: 0.398  loss_mask: 0.689  loss_rpn_cls: 1.602  loss_rpn_loc: 0.185  time: 0.6221  data_time: 0.1212  lr: 0.000010  max_mem: 4338M
    [32m[01/29 06:14:43 d2.utils.events]: [0meta: 0:09:04  iter: 59  total_loss: 3.125  loss_cls: 0.803  loss_box_reg: 0.339  loss_mask: 0.682  loss_rpn_cls: 1.012  loss_rpn_loc: 0.251  time: 0.6270  data_time: 0.1204  lr: 0.000015  max_mem: 4436M
    [32m[01/29 06:14:55 d2.utils.events]: [0meta: 0:08:54  iter: 79  total_loss: 2.445  loss_cls: 0.688  loss_box_reg: 0.474  loss_mask: 0.669  loss_rpn_cls: 0.401  loss_rpn_loc: 0.182  time: 0.6157  data_time: 0.0988  lr: 0.000020  max_mem: 4436M
    [32m[01/29 06:15:08 d2.utils.events]: [0meta: 0:08:56  iter: 99  total_loss: 2.304  loss_cls: 0.606  loss_box_reg: 0.417  loss_mask: 0.657  loss_rpn_cls: 0.370  loss_rpn_loc: 0.202  time: 0.6227  data_time: 0.1108  lr: 0.000025  max_mem: 4436M
    [32m[01/29 06:15:21 d2.utils.events]: [0meta: 0:08:47  iter: 119  total_loss: 2.251  loss_cls: 0.567  loss_box_reg: 0.474  loss_mask: 0.635  loss_rpn_cls: 0.258  loss_rpn_loc: 0.313  time: 0.6295  data_time: 0.1091  lr: 0.000030  max_mem: 4978M
    [32m[01/29 06:15:35 d2.utils.events]: [0meta: 0:08:37  iter: 139  total_loss: 2.033  loss_cls: 0.502  loss_box_reg: 0.418  loss_mask: 0.629  loss_rpn_cls: 0.156  loss_rpn_loc: 0.184  time: 0.6365  data_time: 0.1366  lr: 0.000035  max_mem: 4978M
    [32m[01/29 06:15:48 d2.utils.events]: [0meta: 0:08:27  iter: 159  total_loss: 1.825  loss_cls: 0.454  loss_box_reg: 0.415  loss_mask: 0.589  loss_rpn_cls: 0.134  loss_rpn_loc: 0.145  time: 0.6389  data_time: 0.0956  lr: 0.000040  max_mem: 4988M
    [32m[01/29 06:16:01 d2.utils.events]: [0meta: 0:08:16  iter: 179  total_loss: 1.797  loss_cls: 0.433  loss_box_reg: 0.492  loss_mask: 0.586  loss_rpn_cls: 0.118  loss_rpn_loc: 0.140  time: 0.6387  data_time: 0.1161  lr: 0.000045  max_mem: 4988M
    [32m[01/29 06:16:15 d2.utils.events]: [0meta: 0:08:07  iter: 199  total_loss: 1.826  loss_cls: 0.459  loss_box_reg: 0.439  loss_mask: 0.557  loss_rpn_cls: 0.108  loss_rpn_loc: 0.162  time: 0.6458  data_time: 0.1403  lr: 0.000050  max_mem: 4988M
    [32m[01/29 06:16:29 d2.utils.events]: [0meta: 0:08:01  iter: 219  total_loss: 1.874  loss_cls: 0.431  loss_box_reg: 0.497  loss_mask: 0.531  loss_rpn_cls: 0.139  loss_rpn_loc: 0.164  time: 0.6496  data_time: 0.1200  lr: 0.000055  max_mem: 4988M
    [32m[01/29 06:16:42 d2.utils.events]: [0meta: 0:07:49  iter: 239  total_loss: 1.823  loss_cls: 0.435  loss_box_reg: 0.501  loss_mask: 0.524  loss_rpn_cls: 0.128  loss_rpn_loc: 0.233  time: 0.6522  data_time: 0.1287  lr: 0.000060  max_mem: 4988M
    [32m[01/29 06:16:56 d2.utils.events]: [0meta: 0:07:38  iter: 259  total_loss: 1.741  loss_cls: 0.437  loss_box_reg: 0.478  loss_mask: 0.526  loss_rpn_cls: 0.098  loss_rpn_loc: 0.155  time: 0.6556  data_time: 0.1141  lr: 0.000065  max_mem: 4988M
    [32m[01/29 06:17:10 d2.utils.events]: [0meta: 0:07:27  iter: 279  total_loss: 1.801  loss_cls: 0.412  loss_box_reg: 0.530  loss_mask: 0.514  loss_rpn_cls: 0.120  loss_rpn_loc: 0.182  time: 0.6578  data_time: 0.1336  lr: 0.000070  max_mem: 4988M
    [32m[01/29 06:17:23 d2.utils.events]: [0meta: 0:07:15  iter: 299  total_loss: 1.622  loss_cls: 0.399  loss_box_reg: 0.482  loss_mask: 0.483  loss_rpn_cls: 0.089  loss_rpn_loc: 0.140  time: 0.6580  data_time: 0.1076  lr: 0.000075  max_mem: 4988M
    [32m[01/29 06:17:38 d2.utils.events]: [0meta: 0:07:07  iter: 319  total_loss: 1.703  loss_cls: 0.405  loss_box_reg: 0.530  loss_mask: 0.464  loss_rpn_cls: 0.138  loss_rpn_loc: 0.180  time: 0.6615  data_time: 0.1389  lr: 0.000080  max_mem: 4988M
    [32m[01/29 06:17:52 d2.utils.events]: [0meta: 0:06:56  iter: 339  total_loss: 1.848  loss_cls: 0.432  loss_box_reg: 0.593  loss_mask: 0.467  loss_rpn_cls: 0.107  loss_rpn_loc: 0.179  time: 0.6638  data_time: 0.1256  lr: 0.000085  max_mem: 4988M
    [32m[01/29 06:18:05 d2.utils.events]: [0meta: 0:06:43  iter: 359  total_loss: 1.620  loss_cls: 0.365  loss_box_reg: 0.464  loss_mask: 0.450  loss_rpn_cls: 0.081  loss_rpn_loc: 0.193  time: 0.6629  data_time: 0.0979  lr: 0.000090  max_mem: 4988M
    [32m[01/29 06:18:19 d2.utils.events]: [0meta: 0:06:31  iter: 379  total_loss: 1.751  loss_cls: 0.387  loss_box_reg: 0.572  loss_mask: 0.416  loss_rpn_cls: 0.116  loss_rpn_loc: 0.248  time: 0.6656  data_time: 0.1338  lr: 0.000095  max_mem: 4988M
    [32m[01/29 06:18:33 d2.utils.events]: [0meta: 0:06:19  iter: 399  total_loss: 1.471  loss_cls: 0.340  loss_box_reg: 0.532  loss_mask: 0.432  loss_rpn_cls: 0.057  loss_rpn_loc: 0.144  time: 0.6666  data_time: 0.1383  lr: 0.000100  max_mem: 4988M
    [32m[01/29 06:18:46 d2.utils.events]: [0meta: 0:06:07  iter: 419  total_loss: 1.702  loss_cls: 0.360  loss_box_reg: 0.605  loss_mask: 0.417  loss_rpn_cls: 0.076  loss_rpn_loc: 0.159  time: 0.6676  data_time: 0.1331  lr: 0.000105  max_mem: 4988M
    [32m[01/29 06:19:00 d2.utils.events]: [0meta: 0:05:55  iter: 439  total_loss: 1.494  loss_cls: 0.321  loss_box_reg: 0.507  loss_mask: 0.396  loss_rpn_cls: 0.085  loss_rpn_loc: 0.189  time: 0.6691  data_time: 0.1190  lr: 0.000110  max_mem: 4988M
    [32m[01/29 06:19:14 d2.utils.events]: [0meta: 0:05:42  iter: 459  total_loss: 1.628  loss_cls: 0.352  loss_box_reg: 0.512  loss_mask: 0.396  loss_rpn_cls: 0.081  loss_rpn_loc: 0.197  time: 0.6687  data_time: 0.1171  lr: 0.000115  max_mem: 4988M
    [32m[01/29 06:19:27 d2.utils.events]: [0meta: 0:05:30  iter: 479  total_loss: 1.524  loss_cls: 0.341  loss_box_reg: 0.514  loss_mask: 0.412  loss_rpn_cls: 0.060  loss_rpn_loc: 0.137  time: 0.6689  data_time: 0.1097  lr: 0.000120  max_mem: 4988M
    [32m[01/29 06:19:41 d2.utils.events]: [0meta: 0:05:17  iter: 499  total_loss: 1.595  loss_cls: 0.331  loss_box_reg: 0.514  loss_mask: 0.373  loss_rpn_cls: 0.068  loss_rpn_loc: 0.210  time: 0.6692  data_time: 0.1156  lr: 0.000125  max_mem: 4988M
    [32m[01/29 06:19:54 d2.utils.events]: [0meta: 0:05:04  iter: 519  total_loss: 1.463  loss_cls: 0.307  loss_box_reg: 0.517  loss_mask: 0.357  loss_rpn_cls: 0.098  loss_rpn_loc: 0.170  time: 0.6700  data_time: 0.1308  lr: 0.000130  max_mem: 4988M
    [32m[01/29 06:20:08 d2.utils.events]: [0meta: 0:04:52  iter: 539  total_loss: 1.401  loss_cls: 0.307  loss_box_reg: 0.454  loss_mask: 0.376  loss_rpn_cls: 0.078  loss_rpn_loc: 0.131  time: 0.6701  data_time: 0.1209  lr: 0.000135  max_mem: 4988M
    [32m[01/29 06:20:22 d2.utils.events]: [0meta: 0:04:39  iter: 559  total_loss: 1.349  loss_cls: 0.315  loss_box_reg: 0.487  loss_mask: 0.347  loss_rpn_cls: 0.055  loss_rpn_loc: 0.144  time: 0.6704  data_time: 0.1200  lr: 0.000140  max_mem: 4988M
    [32m[01/29 06:20:36 d2.utils.events]: [0meta: 0:04:27  iter: 579  total_loss: 1.326  loss_cls: 0.306  loss_box_reg: 0.477  loss_mask: 0.334  loss_rpn_cls: 0.073  loss_rpn_loc: 0.114  time: 0.6720  data_time: 0.1310  lr: 0.000145  max_mem: 4988M
    [32m[01/29 06:20:50 d2.utils.events]: [0meta: 0:04:15  iter: 599  total_loss: 1.506  loss_cls: 0.296  loss_box_reg: 0.464  loss_mask: 0.370  loss_rpn_cls: 0.070  loss_rpn_loc: 0.152  time: 0.6726  data_time: 0.1421  lr: 0.000150  max_mem: 4988M
    [32m[01/29 06:21:04 d2.utils.events]: [0meta: 0:04:02  iter: 619  total_loss: 1.219  loss_cls: 0.267  loss_box_reg: 0.432  loss_mask: 0.351  loss_rpn_cls: 0.053  loss_rpn_loc: 0.127  time: 0.6735  data_time: 0.1207  lr: 0.000155  max_mem: 4988M
    [32m[01/29 06:21:17 d2.utils.events]: [0meta: 0:03:49  iter: 639  total_loss: 1.359  loss_cls: 0.293  loss_box_reg: 0.472  loss_mask: 0.338  loss_rpn_cls: 0.072  loss_rpn_loc: 0.160  time: 0.6738  data_time: 0.1062  lr: 0.000160  max_mem: 4988M
    [32m[01/29 06:21:31 d2.utils.events]: [0meta: 0:03:36  iter: 659  total_loss: 1.169  loss_cls: 0.260  loss_box_reg: 0.396  loss_mask: 0.317  loss_rpn_cls: 0.068  loss_rpn_loc: 0.145  time: 0.6736  data_time: 0.1209  lr: 0.000165  max_mem: 4988M
    [32m[01/29 06:21:45 d2.utils.events]: [0meta: 0:03:24  iter: 679  total_loss: 1.390  loss_cls: 0.310  loss_box_reg: 0.455  loss_mask: 0.356  loss_rpn_cls: 0.071  loss_rpn_loc: 0.158  time: 0.6750  data_time: 0.1346  lr: 0.000170  max_mem: 4988M
    [32m[01/29 06:21:59 d2.utils.events]: [0meta: 0:03:12  iter: 699  total_loss: 1.169  loss_cls: 0.249  loss_box_reg: 0.385  loss_mask: 0.305  loss_rpn_cls: 0.049  loss_rpn_loc: 0.121  time: 0.6755  data_time: 0.1130  lr: 0.000175  max_mem: 4988M
    [32m[01/29 06:22:14 d2.utils.events]: [0meta: 0:03:00  iter: 719  total_loss: 1.391  loss_cls: 0.269  loss_box_reg: 0.437  loss_mask: 0.340  loss_rpn_cls: 0.068  loss_rpn_loc: 0.200  time: 0.6774  data_time: 0.1609  lr: 0.000180  max_mem: 4988M
    [32m[01/29 06:22:26 d2.utils.events]: [0meta: 0:02:46  iter: 739  total_loss: 1.292  loss_cls: 0.277  loss_box_reg: 0.470  loss_mask: 0.343  loss_rpn_cls: 0.070  loss_rpn_loc: 0.139  time: 0.6759  data_time: 0.1067  lr: 0.000185  max_mem: 4988M
    [32m[01/29 06:22:41 d2.utils.events]: [0meta: 0:02:34  iter: 759  total_loss: 1.303  loss_cls: 0.280  loss_box_reg: 0.424  loss_mask: 0.302  loss_rpn_cls: 0.059  loss_rpn_loc: 0.176  time: 0.6770  data_time: 0.1211  lr: 0.000190  max_mem: 4988M
    [32m[01/29 06:22:54 d2.utils.events]: [0meta: 0:02:21  iter: 779  total_loss: 1.174  loss_cls: 0.241  loss_box_reg: 0.418  loss_mask: 0.318  loss_rpn_cls: 0.050  loss_rpn_loc: 0.145  time: 0.6769  data_time: 0.1242  lr: 0.000195  max_mem: 4988M
    [32m[01/29 06:23:07 d2.utils.events]: [0meta: 0:02:08  iter: 799  total_loss: 1.161  loss_cls: 0.262  loss_box_reg: 0.447  loss_mask: 0.326  loss_rpn_cls: 0.047  loss_rpn_loc: 0.128  time: 0.6763  data_time: 0.1084  lr: 0.000200  max_mem: 4988M
    [32m[01/29 06:23:22 d2.utils.events]: [0meta: 0:01:56  iter: 819  total_loss: 1.191  loss_cls: 0.258  loss_box_reg: 0.408  loss_mask: 0.318  loss_rpn_cls: 0.039  loss_rpn_loc: 0.165  time: 0.6771  data_time: 0.1244  lr: 0.000205  max_mem: 4988M
    [32m[01/29 06:23:36 d2.utils.events]: [0meta: 0:01:43  iter: 839  total_loss: 1.192  loss_cls: 0.266  loss_box_reg: 0.381  loss_mask: 0.301  loss_rpn_cls: 0.058  loss_rpn_loc: 0.149  time: 0.6776  data_time: 0.1318  lr: 0.000210  max_mem: 4988M
    [32m[01/29 06:23:49 d2.utils.events]: [0meta: 0:01:30  iter: 859  total_loss: 1.123  loss_cls: 0.232  loss_box_reg: 0.380  loss_mask: 0.292  loss_rpn_cls: 0.037  loss_rpn_loc: 0.105  time: 0.6780  data_time: 0.1200  lr: 0.000215  max_mem: 4988M
    [32m[01/29 06:24:03 d2.utils.events]: [0meta: 0:01:18  iter: 879  total_loss: 1.139  loss_cls: 0.220  loss_box_reg: 0.432  loss_mask: 0.296  loss_rpn_cls: 0.039  loss_rpn_loc: 0.119  time: 0.6778  data_time: 0.0951  lr: 0.000220  max_mem: 4988M
    [32m[01/29 06:24:16 d2.utils.events]: [0meta: 0:01:05  iter: 899  total_loss: 1.111  loss_cls: 0.246  loss_box_reg: 0.414  loss_mask: 0.287  loss_rpn_cls: 0.045  loss_rpn_loc: 0.148  time: 0.6771  data_time: 0.1040  lr: 0.000225  max_mem: 4988M
    [32m[01/29 06:24:31 d2.utils.events]: [0meta: 0:00:52  iter: 919  total_loss: 1.142  loss_cls: 0.226  loss_box_reg: 0.375  loss_mask: 0.304  loss_rpn_cls: 0.062  loss_rpn_loc: 0.193  time: 0.6784  data_time: 0.1567  lr: 0.000230  max_mem: 4988M
    [32m[01/29 06:24:45 d2.utils.events]: [0meta: 0:00:39  iter: 939  total_loss: 1.133  loss_cls: 0.225  loss_box_reg: 0.394  loss_mask: 0.289  loss_rpn_cls: 0.032  loss_rpn_loc: 0.164  time: 0.6790  data_time: 0.1120  lr: 0.000235  max_mem: 4988M
    [32m[01/29 06:24:58 d2.utils.events]: [0meta: 0:00:26  iter: 959  total_loss: 1.153  loss_cls: 0.250  loss_box_reg: 0.390  loss_mask: 0.299  loss_rpn_cls: 0.056  loss_rpn_loc: 0.120  time: 0.6792  data_time: 0.1266  lr: 0.000240  max_mem: 4988M
    [32m[01/29 06:25:12 d2.utils.events]: [0meta: 0:00:13  iter: 979  total_loss: 1.176  loss_cls: 0.256  loss_box_reg: 0.434  loss_mask: 0.278  loss_rpn_cls: 0.038  loss_rpn_loc: 0.149  time: 0.6793  data_time: 0.1200  lr: 0.000245  max_mem: 4988M
    [32m[01/29 06:25:28 d2.utils.events]: [0meta: 0:00:00  iter: 999  total_loss: 1.104  loss_cls: 0.210  loss_box_reg: 0.342  loss_mask: 0.275  loss_rpn_cls: 0.043  loss_rpn_loc: 0.103  time: 0.6797  data_time: 0.1221  lr: 0.000250  max_mem: 4988M
    [32m[01/29 06:25:28 d2.engine.hooks]: [0mOverall training speed: 997 iterations in 0:11:18 (0.6804 s / it)
    [32m[01/29 06:25:28 d2.engine.hooks]: [0mTotal training time: 0:11:21 (0:00:02 on hooks)
    




    OrderedDict()



### Inferencing for new data

For inferencing, we need to load the final model created by the training and load the validation data set, or any kind of data set that you wish to inference on. Also, two folders have to be created within Google Colab. Be aware that depending on the amount of `iterations` and the chosen `threshold`, some or all images may show no predicted annotations. In that case, you have to adapt your configuration accordingly.


```python
!mkdir predictions
!mkdir output_images
```


```python
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
    0.70  # set the testing threshold for this model
)

# load the validation data
cfg.DATASETS.TEST = ("buildings_val",)
# create a predictor
predictor = DefaultPredictor(cfg)

start = datetime.now()

validation_folder = Path("/content/val")

for i, file in enumerate(validation_folder.glob("*.jpg")):
    # this loop opens the .jpg files from the val-folder, creates a dict with the file
    # information, plots visualizations and saves the result as .pkl files.
    file = str(file)
    file_name = file.split("/")[-1]
    im = cv2.imread(file)

    outputs = predictor(im)
    output_with_filename = {}
    output_with_filename["file_name"] = file_name
    output_with_filename["file_location"] = file
    output_with_filename["prediction"] = outputs
    with open(f"/content/predictions/predictions_{i}.pkl", "wb") as f:
        pickle.dump(output_with_filename, f)
    v = Visualizer(
        im[:, :, ::-1],
        metadata=building_metadata,
        scale=1,
        instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
    )

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.savefig(f"/content/output_images/{file_name}")
print("Time needed for inferencing:", datetime.now() - start)
```

    Time needed for inferencing: 0:02:10.990693
    


![png](output_34_1.png)


## Part 3 - Processing the prediction results

### Importing of additional packages

For processing our results, we need to import a few additional packages.


```python
import pickle
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
import pandas as pd
```

### Set some general colour and font settings

We download a font for displaying text on our images and set a few colors. They are in RGBA-format, so change values as you wish. If you do not require plotting the results as
images, set `plot_data` to False, thereby **decreasing computation time by 5x**.


```python
!wget https://github.com/google/fonts/raw/master/apache/roboto/Roboto-Regular.ttf
```

    --2020-01-29 06:28:03--  https://github.com/google/fonts/raw/master/apache/roboto/Roboto-Regular.ttf
    Resolving github.com (github.com)... 192.30.253.112
    Connecting to github.com (github.com)|192.30.253.112|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://raw.githubusercontent.com/google/fonts/master/apache/roboto/Roboto-Regular.ttf [following]
    --2020-01-29 06:28:03--  https://raw.githubusercontent.com/google/fonts/master/apache/roboto/Roboto-Regular.ttf
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 171676 (168K) [application/octet-stream]
    Saving to: ‘Roboto-Regular.ttf’
    
    Roboto-Regular.ttf  100%[===================>] 167.65K  --.-KB/s    in 0.03s   
    
    2020-01-29 06:28:03 (4.89 MB/s) - ‘Roboto-Regular.ttf’ saved [171676/171676]
    
    


```python
# define fonts and colors
font_id = ImageFont.truetype("/content/Roboto-Regular.ttf", 15)
font_result = ImageFont.truetype("/content/Roboto-Regular.ttf", 40)
text_color = (255, 255, 255, 128)
background_bbox_window = (0, 247, 255, 30)
background_bbox_building = (255, 167, 14, 30)
background_text = (0, 0, 0, 150)
background_mask_window = (0, 247, 255, 100)
background_mask_building = (255, 167, 14, 100)
device = "cpu"

# this variable is True if you want to plot the output images, False if you only need
# the CSV
plot_data = True
```

### Function to draw a bounding box

This function draws a bounding box as well as the window to facade percentage.


```python
def draw_bounding_box(img, bounding_box, text, category, id, draw_box=False):
    """Draws a bounding box onto the image as well as the building ID and the window 
    percentage."""

    x = bounding_box[0]
    y = bounding_box[3]
    text = str(round(text, 2))
    draw = ImageDraw.Draw(img, "RGBA")
    if draw_box:
        if category == 0:
            draw.rectangle(bounding_box, fill=background_bbox_window, outline=(0, 0, 0))
        elif category == 1:
            draw.rectangle(
                bounding_box, fill=background_bbox_building, outline=(0, 0, 0)
            )
    w, h = font_id.getsize(id)

    draw.rectangle((x, y, x + w, y - h), fill=background_text)
    draw.text((x, y - h), id, fill=text_color, font=font_id)
    # for buildings, add the window percentage value in the lower right corner
    if category == 1:
        w, h = font_result.getsize(text)
        draw.rectangle((x, y, x + w, y + h), fill=background_text)
        draw.text((x, y), text, fill=text_color, font=font_result)
```

### Function to draw masks

This function draws masks to the image. To do that, the binary numpy masks constisting of `True/False` for each pixel value of the image are converted to `RGBA` image files.


```python
def draw_mask(img, mask, category):
    """Draws a mask onto the image."""

    img = img.convert("RGBA")

    mask_RGBA = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    if category == 0:
        mask_RGBA[mask] = background_mask_window
    elif category == 1:
        mask_RGBA[mask] = background_mask_building
    mask_RGBA[~mask] = [0, 0, 0, 0]
    rgb_image = Image.fromarray(mask_RGBA).convert("RGBA")

    img = Image.alpha_composite(img, rgb_image)

    return img
```

### Calculate the percentage of window to facade

This function takes the results of the predictions and sums up all window-pixels within a building. The sum is then divided by the total amount of building pixel area of the respective building.


```python
def calculate_window_perc(dataset):
    """Takes a list of prediction dictionaries as input and calculates the percentage of 
    window to fassade for each building. The result is save to the dataset. For building
    data, the actual percentage is saved, for windows, 1.0 is put in."""
    with open("/content/val/via_region_data.json") as f:
        json_file = json.load(f)
    for i, data in enumerate(dataset):
        data["window_percentage"] = 0
        data["pixel_area"] = 0 
        data["tagged_id"] = 0
        # loop through building
        if data["category"] == 1:
            data = get_tagged_id(data, json_file)
            window_areas = []
            building_mask = data["mask"]
            building_area = np.sum(data["mask"])

            for x in dataset:
                # for each building, loop through each window
                if x["category"] == 0:
                    x["window_percentage"] = 1
                    pixels_overlapping = np.sum(x["mask"][building_mask])

                    window_areas.append(pixels_overlapping)

            window_percentage = sum(window_areas) / building_area
            
            data["window_percentage"] = window_percentage
            data["pixel_area"] = building_area
    
    return dataset
```

### Get Tagged IDs for the Buildings
The following function searches through the `via_region_data.json` file for any points that have an `tagged_id` category and if such a point matches with a predicted building mask, the `tagged_id` and point location is added to the building.


```python
def get_tagged_id(building, json_file):
    """Searches through the via_export_json.json of the images used for inferencing and 
    adds all tagged_ids to the dataset."""

    building["tagged_id"] = 0
    building["tagged_id_coords"] = 0
    # loop through the JSON annotations file
    for idx, v in enumerate(json_file.values()):
        annos = v["regions"]

        if v["filename"] == building["file_name"]:
            try:
                for annotation in annos:
                    anno = annotation["shape_attributes"]
                    # if the annotation is not a point, go to the next annotation
                    if anno["name"] != "point":
                        continue

                    if anno["name"] == "point":
                        tagged_id = annotation["region_attributes"]["tagged_id"]
                        px = anno["cx"]
                        py = anno["cy"]
                        point = [py, px]
                        # if the point location matches with the building mask, add the 
                        # id to the building data
                        if building["mask"][py][px]:
                            building["tagged_id"] = tagged_id
                            building["tagged_id_coords"] = point

            except KeyError as e:
                print("Error:", e)
                return building

    return building
```

### Save the building information to a CSV

This function saves the results for the buildings and their percentage values into a CSV-file, which can then be used for further processing or modelling.


```python
def create_csv(dataset):
    """Takes a list of lists of data dictionaries as input, flattens this list, creates
    a DataFrame, removes unnecessary information and saves it as a CSV."""

    # flatten list
    dataset = [item for sublist in dataset for item in sublist]
    df = pd.DataFrame(dataset)


    # calculate the percentage of the building compared to the total image size
    df["total_image_area"] = df["image_height"] * df["image_width"]
    df["building_area_perc_of_image"] = df["pixel_area"] / df["total_image_area"]
    # keep only specific columns
    df = df[
        [
            "file_name",
            "id",
            "tagged_id",
            "tagged_id_coords",
            "category",
            "pixel_area",
            "building_area_perc_of_image",
            "window_percentage",
        ]
    ]
    # only keep building information
    df = df[df["category"] == 1]
    

    df.to_csv("/content/result.csv")
    return(df)
```

### Function to process all the data

This function processes the results from the prediction and applies all the processing function from above. Bounding boxes and masks are drawn, the window percentage calculated and the resulting dicts with all the information is return.


```python
def process_data(file_path, plot_data=plot_data):
    """Takes an prediction result in form of a .pkl-file and draws the mask and bounding
    box information. From these, the percentage of windows to fassade for each building
    is calculated and plotted onto the image if plot_data=True."""
    with open(file_path, "rb") as f:
        # the following lines of code extract specific data from the prediction-dict
        prediction = pickle.load(f)

        image_height = prediction["prediction"]["instances"].image_size[0]
        image_width = prediction["prediction"]["instances"].image_size[1]
        
        # the data is still saved on the GPU and needs to be moved to the CPU
        boxes = (
            prediction["prediction"]["instances"]
            .get_fields()["pred_boxes"]
            .tensor.to(device)
            .numpy()
        )

        img = Image.open(prediction["file_location"])
        categories = (
            prediction["prediction"]["instances"]
            .get_fields()["pred_classes"]
            .to(device)
            .numpy()
        )
        masks = (
            prediction["prediction"]["instances"]
            .get_fields()["pred_masks"]
            .to(device)
            .numpy()
        )

        dataset = []
        counter_window = 0
        counter_building = 0
        # create a new data-dict as well as IDs for each building and window within an
        # image
        for i, box in enumerate(boxes):

            data = {}
            data["file_name"] = prediction["file_name"]
            data["file_location"] = prediction["file_location"]
            data["image_height"] = image_height
            data["image_width"] = image_width
            # category 0 is always a window
            if categories[i] == 0:
                data["id"] = f"w_{counter_window}"
                counter_window = counter_window + 1
            # category 1 is always a building
            elif categories[i] == 1:
                data["id"] = f"b_{counter_building}"
                counter_building = counter_building + 1

            data["bounding_box"] = box
            data["category"] = categories[i]
            data["mask"] = masks[i]
            dataset.append(data)

        dataset = calculate_window_perc(dataset)

        if plot_data:
            for i, data in enumerate(dataset):
                draw_bounding_box(
                    img,
                    data["bounding_box"],
                    data["window_percentage"],
                    data["category"],
                    data["id"],
                    draw_box=True,
                )
            for i, data in enumerate(dataset):
                img = draw_mask(img, data["mask"], data["category"])
            try:
              img.save(
                  f"/content/predictions/{data['file_name']}_prediction.png",
                  quality=95,
              )
            except UnboundLocalError as e:
              print("no annotations found, skipping")
    return dataset
```

### Use multiple CPU Cores for processing
This does not work well with Google Colab, but on a local machine with multiple CPU cores this would speed up processing quite a bit. 


```python
def apply_mp_progress(func, n_processes, prediction_list):
    """Applies multiprocessing to a list of data. Currently does not work well in Google
    Collab."""
    
    p = mp.Pool(n_processes)

    res_list = []
    with tqdm(total=len(prediction_list)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, prediction_list))):
            pbar.update()
            res_list.append(res)
        pbar.close()
    p.close()
    p.join()
    return res_list
```

### Run the processing

This last piece of code runs the processing of the results and saves it as a CSV file.


```python
prediction_folder = Path("/content/predictions/")
prediction_list = []
start = datetime.now()
for i, file in enumerate(prediction_folder.glob("*.pkl")):
    file = str(file)
    prediction_list.append(file)


# this is for processing on a single CPU in Colab
dataset = []

for file_location in tqdm(prediction_list):
  dataset_part = process_data(file_location)
  dataset.append(dataset_part)

# If you use this code on a local machine, comment out the four lines above and uncomment
# the line below
# dataset = apply_mp_progress(process_data, mp.cpu_count(), prediction_list)


df = create_csv(dataset)

print(datetime.now() - start)
df
```

    100%|██████████| 49/49 [02:06<00:00,  2.58s/it]

    0:02:06.347280
    

    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file_name</th>
      <th>id</th>
      <th>tagged_id</th>
      <th>tagged_id_coords</th>
      <th>category</th>
      <th>pixel_area</th>
      <th>building_area_perc_of_image</th>
      <th>window_percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>P2VBr3KZuYBOF1QA-DIRrQ.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>325987</td>
      <td>0.157208</td>
      <td>0.031940</td>
    </tr>
    <tr>
      <th>87</th>
      <td>P8eVIvM6OscXlR5SR6Z05w.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>502350</td>
      <td>0.242260</td>
      <td>0.108572</td>
    </tr>
    <tr>
      <th>183</th>
      <td>p8KcAK9bYM1MhB6m1LZBzA.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>203559</td>
      <td>0.098167</td>
      <td>0.015111</td>
    </tr>
    <tr>
      <th>216</th>
      <td>_6qHHdUHjbyLGdwqfTZgRA.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>491739</td>
      <td>0.237143</td>
      <td>0.020932</td>
    </tr>
    <tr>
      <th>251</th>
      <td>Pb73PiqChAM_VjnM6-mByA.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>376957</td>
      <td>0.181789</td>
      <td>0.011261</td>
    </tr>
    <tr>
      <th>260</th>
      <td>Pb73PiqChAM_VjnM6-mByA.jpg</td>
      <td>b_1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>178937</td>
      <td>0.086293</td>
      <td>0.103031</td>
    </tr>
    <tr>
      <th>272</th>
      <td>PAJ-MaBAsXToem1f8tQXQA.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2656813</td>
      <td>0.844578</td>
      <td>0.040326</td>
    </tr>
    <tr>
      <th>306</th>
      <td>_FczOQH2Gps-DKWQb9aLlw.jpg</td>
      <td>b_0</td>
      <td>id_1</td>
      <td>[319, 507]</td>
      <td>1</td>
      <td>402790</td>
      <td>0.194247</td>
      <td>0.133335</td>
    </tr>
    <tr>
      <th>322</th>
      <td>_YBDP9H7rtRL8ooFlNKGew.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>328583</td>
      <td>0.158460</td>
      <td>0.133817</td>
    </tr>
    <tr>
      <th>440</th>
      <td>_ObTdbCdQQTHL1kYnT7dyg.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>81129</td>
      <td>0.025790</td>
      <td>0.093025</td>
    </tr>
    <tr>
      <th>529</th>
      <td>_40EqDxSm7VfFa3loCybQA.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>476829</td>
      <td>0.229952</td>
      <td>0.038534</td>
    </tr>
    <tr>
      <th>558</th>
      <td>p0BLyIbSaWEhftL1Cuasaw.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>760542</td>
      <td>0.241770</td>
      <td>0.027758</td>
    </tr>
    <tr>
      <th>576</th>
      <td>p9LjGq-iOOegdkOiA4JRPQ.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>132229</td>
      <td>0.063768</td>
      <td>0.113470</td>
    </tr>
    <tr>
      <th>739</th>
      <td>p4zxq9hYv_OWvCwoy83HJA.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>145565</td>
      <td>0.070199</td>
      <td>0.006808</td>
    </tr>
    <tr>
      <th>760</th>
      <td>p4w0rP6eCcUI3w3htiMm3g.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>704624</td>
      <td>0.339807</td>
      <td>0.006323</td>
    </tr>
    <tr>
      <th>762</th>
      <td>p4w0rP6eCcUI3w3htiMm3g.jpg</td>
      <td>b_1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>216708</td>
      <td>0.104508</td>
      <td>0.045554</td>
    </tr>
    <tr>
      <th>822</th>
      <td>p4zCd5huZE35tEWakIhZv6.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>163759</td>
      <td>0.078973</td>
      <td>0.119212</td>
    </tr>
    <tr>
      <th>824</th>
      <td>p4zCd5huZE35tEWakIhZv6.jpg</td>
      <td>b_1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>371558</td>
      <td>0.179185</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>885</th>
      <td>_qOoozLWFdlXfwI4lbFKCw.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>214753</td>
      <td>0.103565</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>910</th>
      <td>p7Mu8DjZsfEomtAVPDy0hw.jpg</td>
      <td>b_0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>247912</td>
      <td>0.119556</td>
      <td>0.054652</td>
    </tr>
    <tr>
      <th>915</th>
      <td>p7Mu8DjZsfEomtAVPDy0hw.jpg</td>
      <td>b_1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>369672</td>
      <td>0.178275</td>
      <td>0.031777</td>
    </tr>
  </tbody>
</table>
</div>



### Downloading Results to the local Machine
You might want to download some of the results after all of this, especially due to the fact that all files created in this Colab notebook are temporary and deleted after 12 hours. The following code creates a zip file for a directory and all it's content and then opens up a download-dialog. For the download to work, you need to accept third party cookies and you also need to use the `Chrome` browser from Google.


```python
!zip -r predictions.zip predictions

```

      adding: predictions/ (stored 0%)
      adding: predictions/predictions_30.pkl (deflated 100%)
      adding: predictions/P8eVIvM6OscXlR5SR6Z05w.jpg_prediction.png (deflated 1%)
      adding: predictions/P2VBr3KZuYBOF1QA-DIRrQ.jpg_prediction.png (deflated 1%)
      adding: predictions/p8KcAK9bYM1MhB6m1LZBzA.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_7.pkl (deflated 100%)
      adding: predictions/predictions_26.pkl (deflated 100%)
      adding: predictions/PBwMzmcTJXTlPcpdemsZNA.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_1.pkl (deflated 100%)
      adding: predictions/predictions_45.pkl (deflated 100%)
      adding: predictions/predictions_16.pkl (deflated 100%)
      adding: predictions/predictions_22.pkl (deflated 100%)
      adding: predictions/predictions_34.pkl (deflated 100%)
      adding: predictions/P7zqBvtQSjv16d7s2MZjiA.jpg_prediction.png (deflated 1%)
      adding: predictions/_YBDP9H7rtRL8ooFlNKGew.jpg_prediction.png (deflated 1%)
      adding: predictions/p9r3VS5NDbvjErpX9tVxbQ.jpg_prediction.png (deflated 2%)
      adding: predictions/p9ZXCIT-rMlLwGC-BxRHNA.jpg_prediction.png (deflated 0%)
      adding: predictions/_FczOQH2Gps-DKWQb9aLlw.jpg_prediction.png (deflated 0%)
      adding: predictions/predictions_3.pkl (deflated 100%)
      adding: predictions/_hJfLUBz3M0MQXMz_5afWw.jpg_prediction.png (deflated 1%)
      adding: predictions/P7W1YnIB99tOroMW4BVQNQ.jpg_prediction.png (deflated 0%)
      adding: predictions/predictions_39.pkl (deflated 100%)
      adding: predictions/_lQKTfgy5oLfrnDqnb7_ng.jpg_prediction.png (deflated 2%)
      adding: predictions/predictions_46.pkl (deflated 100%)
      adding: predictions/predictions_18.pkl (deflated 100%)
      adding: predictions/predictions_23.pkl (deflated 100%)
      adding: predictions/predictions_8.pkl (deflated 100%)
      adding: predictions/predictions_4.pkl (deflated 100%)
      adding: predictions/predictions_15.pkl (deflated 100%)
      adding: predictions/_IiT38b3e9Sy5Pq1B49p4s.jpg_prediction.png (deflated 0%)
      adding: predictions/predictions_19.pkl (deflated 100%)
      adding: predictions/predictions_41.pkl (deflated 100%)
      adding: predictions/predictions_47.pkl (deflated 100%)
      adding: predictions/predictions_35.pkl (deflated 100%)
      adding: predictions/predictions_24.pkl (deflated 100%)
      adding: predictions/p0BLyIbSaWEhftL1Cuasaw.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_9.pkl (deflated 100%)
      adding: predictions/_w6T3cr3IYC2PmcGvmyvMg.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_33.pkl (deflated 100%)
      adding: predictions/predictions_44.pkl (deflated 100%)
      adding: predictions/P3dr3y0NZpnQsTcjIW-x9U.jpg_prediction.png (deflated 0%)
      adding: predictions/predictions_5.pkl (deflated 100%)
      adding: predictions/predictions_42.pkl (deflated 100%)
      adding: predictions/_QCaqWVWeYXZWIUK_0Wbag.jpg_prediction.png (deflated 1%)
      adding: predictions/P6PAuiLcfV7cWvpnMZ0ijg.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_13.pkl (deflated 100%)
      adding: predictions/predictions_11.pkl (deflated 100%)
      adding: predictions/_VOe8cqDxWY8vhZ6j_rMeA.jpg_prediction.png (deflated 0%)
      adding: predictions/P4we3TZeMBwNVD2Q8KyegQ.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_25.pkl (deflated 100%)
      adding: predictions/predictions_0.pkl (deflated 100%)
      adding: predictions/_Odc9YG8R9RXBkBE9vRNPw.jpg_prediction.png (deflated 1%)
      adding: predictions/_40EqDxSm7VfFa3loCybQA.jpg_prediction.png (deflated 1%)
      adding: predictions/PcEaOBL5k5V2-GV6TRcoIA.jpg_prediction.png (deflated 0%)
      adding: predictions/predictions_48.pkl (deflated 100%)
      adding: predictions/predictions_38.pkl (deflated 100%)
      adding: predictions/predictions_6.pkl (deflated 100%)
      adding: predictions/PBq3z7XcxIDwKfaRSSieqA.jpg_prediction.png (deflated 1%)
      adding: predictions/_0CfAgwU4GRi9S6yzBbnxQ.jpg_prediction.png (deflated 0%)
      adding: predictions/predictions_21.pkl (deflated 100%)
      adding: predictions/predictions_29.pkl (deflated 100%)
      adding: predictions/p_pGU7tYYIfdfimFqyD_Is.jpg_prediction.png (deflated 0%)
      adding: predictions/Pb73PiqChAM_VjnM6-mByA.jpg_prediction.png (deflated 1%)
      adding: predictions/p4yU2_tWslUKT24cW54oyg.jpg_prediction.png (deflated 1%)
      adding: predictions/p7Mu8DjZsfEomtAVPDy0hw.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_14.pkl (deflated 100%)
      adding: predictions/predictions_17.pkl (deflated 100%)
      adding: predictions/_qOoozLWFdlXfwI4lbFKCw.jpg_prediction.png (deflated 1%)
      adding: predictions/_1COpxhN8GOKkjj0fV8yrg.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_31.pkl (deflated 100%)
      adding: predictions/p9LjGq-iOOegdkOiA4JRPQ.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_36.pkl (deflated 100%)
      adding: predictions/_ObTdbCdQQTHL1kYnT7dyg.jpg_prediction.png (deflated 0%)
      adding: predictions/predictions_40.pkl (deflated 100%)
      adding: predictions/P5BoM4S8cmOBMS0kxjNgBQ.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_20.pkl (deflated 100%)
      adding: predictions/_CrNw7UUGzt7WknYd63I4w.jpg_prediction.png (deflated 0%)
      adding: predictions/predictions_43.pkl (deflated 100%)
      adding: predictions/p4zxq9hYv_OWvCwoy83HJA.jpg_prediction.png (deflated 1%)
      adding: predictions/p2YsGfPOCT8FXYvjta4row.jpg_prediction.png (deflated 1%)
      adding: predictions/P9x_OoS8_cN6gAFBfWqZ4Q.jpg_prediction.png (deflated 2%)
      adding: predictions/predictions_12.pkl (deflated 100%)
      adding: predictions/predictions_28.pkl (deflated 100%)
      adding: predictions/_Ujvl8m3THd78AyUC6we8w.jpg_prediction.png (deflated 1%)
      adding: predictions/PbQB6gupFxRsFwR9LljW_Q.jpg_prediction.png (deflated 1%)
      adding: predictions/p4w0rP6eCcUI3w3htiMm3g.jpg_prediction.png (deflated 0%)
      adding: predictions/p9pRPA0o-i7xU2EXelvqvA.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_10.pkl (deflated 100%)
      adding: predictions/predictions_2.pkl (deflated 100%)
      adding: predictions/p4X8gwA8Q15OI9ZUHrTJZg.jpg_prediction.png (deflated 1%)
      adding: predictions/_k84RoRK92PvZFj4AWcvWw.jpg_prediction.png (deflated 0%)
      adding: predictions/_6qHHdUHjbyLGdwqfTZgRA.jpg_prediction.png (deflated 1%)
      adding: predictions/p9MdBwk8aMwfvNYVdsrzwQ.jpg_prediction.png (deflated 1%)
      adding: predictions/predictions_37.pkl (deflated 100%)
      adding: predictions/Pbu3KVTnBvhRxFuRU7mQDg.jpg_prediction.png (deflated 1%)
      adding: predictions/p4zCd5huZE35tEWakIhZv6.jpg_prediction.png (deflated 0%)
      adding: predictions/predictions_32.pkl (deflated 100%)
      adding: predictions/predictions_27.pkl (deflated 100%)
      adding: predictions/PAJ-MaBAsXToem1f8tQXQA.jpg_prediction.png (deflated 0%)
    


```python
from google.colab import files
files.download('/content/predictions.zip')
```
