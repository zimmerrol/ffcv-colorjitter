<p align = 'center'>
<em><b>Fast Forward Computer Vision</b>: train models at <a href="#imagenet">1/10th the cost*</a> with accelerated data loading!</em>
</p>
<img src='assets/logo.svg' width='100%'/>
<p align = 'center'>
[<a href="https://ffcv.io">homepage</a>]
[<a href="#installation">install</a>]
[<a href="#quickstart">quickstart</a>]
[<a href="https://docs.ffcv.io">docs</a>]
[<a href="#imagenet">ImageNet</a>]
[<a href="https://join.slack.com/t/ffcv-workspace/shared_invite/zt-11olgvyfl-dfFerPxlm6WtmlgdMuw_2A">support slack</a>]
<br>
Maintainers:
<a href="https://twitter.com/gpoleclerc">Guillaume Leclerc</a>,
<a href="https://twitter.com/andrew_ilyas">Andrew Ilyas</a> and
<a href="https://twitter.com/logan_engstrom">Logan Engstrom</a>
</p>

`ffcv` dramatically increases data throughput in accelerated computing systems,
offering:
- <a href="#quickstart">Fast data loading and processing</a> (even in resource constrained environments)
- Efficient, simple, easy-to-understand, customizable training code for standard
   computer vision tasks

Install `ffcv` today and:
- ...train an ImageNet model on one GPU in TODO minutes (XX$ on AWS)
- ...train a CIFAR-10 model on one GPU in 36 seconds (XX$ on AWS)
- ...train a `$YOUR_DATASET` model `$REALLY_FAST` (for `$WAY_LESS`)

Compare our training and dataloading times to what you use now: 

<img src="assets/headline.svg"/>

Holding constant the same training routine and optimizing only the dataloading and data transfer routines with `ffcv`, we enable significantly faster training
(see [here](TODO) for further benchmark details).


## Install
With [Anaconda](https://docs.anaconda.com/anaconda/install/index.html):

```
conda install ffcv -c pytorch -c conda-forge -c ffcv
``` 

## Citation
If you use FFCV, please cite it as:

```
@misc{leclerc2022ffcv,
    author = {Guillaume Leclerc and Andrew Ilyas and Logan Engstrom and Sung Min Park and Hadi Salman and Aleksander Madry},
    title = {ffcv},
    year = {2021},
    howpublished = {\url{https://github.com/MadryLab/ffcv/}},
    note = {commit xxxxxxx}
}
```

## Index
- <a href="#quickstart"><b>Quickstart</b></a>: High level guide to `ffcv`.
- <a href="#features"><b>Features</b></a>: What can `ffcv` do for you?
- <a href="#prepackaged-computer-vision-benchmarks"><b>Fast Training Code</b></a>: Results, code, and training configs for ImageNet and CIFAR-10.

## Quickstart
Accelerate <a href="#features">*any*</a> learning system with `ffcv`.
First,
convert your dataset into `ffcv` format (`ffcv` converts both indexed PyTorch datasets and
<a href="https://github.com/webdataset/webdataset">WebDatasets</a>):
```python
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, NDArrayField
import numpy as np

# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
my_dataset = make_my_dataset() 
write_path = '/output/path/for/converted/ds.ffcv'

# Pass a type for each data field
writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time 
    'image': RGBImageField({
        max_resolution=256,
        jpeg_quality=jpeg_quality
    }),
    'label': IntField()
})

# Write dataset
writer.from_indexed_dataset(ds)
```
Then replace your old loader with the `ffcv` loader at train time (in PyTorch,
no other changes required!):
```python
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder

# Random resized crop
decoder = RandomResizedCropRGBImageDecoder((224, 224))

# Data decoding and augmentation
image_pipeline = [decoder, Cutout(), ToTensor(), ToTorchImage(), ToDevice(0)]
label_pipeline = [IntDecoder(), ToTensor(), ToDevice(0)]

# Pipeline for each data field
pipelines = {
    'image': image_pipeline,
    'label': label_pipeline
}

# Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
loader = Loader(train_path, batch_size=bs, num_workers=num_workers,
                order=OrderOption.RANDOM, pipelines=pipelines)

# rest of training / validation proceeds identically
for epoch in range(epochs):
    ...
```
[See here](TODO) for a more detailed guide to deploying `ffcv` for your dataset.

## Prepackaged Computer Vision Benchmarks
From gridding to benchmarking to fast research iteration, there are many reasons
to want faster model training. Below we present premade codebases for training 
on ImageNet and CIFAR, including both (a) extensible codebases and (b)
numerous premade training configurations.

### ImageNet
We provide a self-contained script for training ImageNet <it>fast</it>. Below we
plot the training time versus
accuracy frontier for 1-GPU ResNet-18 and 8-GPU ResNet-50 alongside
a few baselines:

[resnet18 plot] [resnet50 plot]

**Train your own ImageNet models!** <a href="https://github.com/MadryLab/ffcv/tree/new_ver/examples/imagenet">Use our training script and premade configurations</a> to train any model seen on the above graphs.

### CIFAR-10
We also include premade code for efficient training on CIFAR-10 in the `examples/`
directory, obtaining 93\% top1 accuracy in 36 seconds on a single A100 GPU
(without optimizations such as MixUp, Ghost BatchNorm, etc. which have the
potential to raise the accuracy even further). You can find the training script
<a href="https://github.com/MadryLab/ffcv/tree/new_ver/examples/cifar">here</a>. 

## Features
<img src='docs/_static/clippy-transparent-2.png' width='100%'/>

Computer vision or not, FFCV can help make training faster in a variety of
resource-constrained settings!
Our <a href="https://docs.ffcv.io/performance_guide.html">performance guide</a>
has a more detailed account of the ways in which FFCV can adapt to different
performance bottlenecks.

- **Plug-and-play with any existing training code**: Rather than changing
  aspects of model training itself, FFCV focuses on removing *data bottlenecks*,
  which turn out to be a problem everywhere from neural network training to
  linear regression. This means that:

    - FFCV can be introduced into any existing training code in just a few
      lines of code (e.g., just swapping out the data loader and optionally the
      augmentation pipeline);
    - you don't have to change the model itself to make it faster (e.g., feel
      free to analyze models *without* CutMix, Dropout, momentum scheduling, etc.);
    - FFCV can speed up a lot more than just neural network training---in
      fact, the more data-bottlenecked the application, the faster FFCV will make it!

  See our [Getting started]() guide, [Example walkthroughs](), and [Code
  examples]() to see how easy it is to get started!
- **Fast data processing without the pain**: FFCV automatically handle data
  reading, pre-fetching, caching, and transfer between devices in an extremely
  efficiently way, so that users don't have to think about it.
- **Automatically fused-and-compiled data processing**: By either using
  [](pre-written) FFCV transformations or [](easily writing them), users can
  take advantage of FFCV's compilation and pipelining abilities, which will
  automatically fuse and compile simple Python augmentations to machine code
  using [http://numba.org](Numba), and schedule them asynchronously to avoid
  loading delays.
- **Load data fast from RAM, SSD, or networked disk**: FFCV exposes
  user-friendly options that can be adjusted based on the resources
  available. For example, if a dataset fits into memory, FFCV can cache it
  at the OS level and ensure that multiple concurrent processes all get fast
  data access. Otherwise, FFCV can use fast process-level caching and will
  optimize data loading to minimize the underlying number of disk reads. See
  [The Bottleneck Doctor]() guide for more information!
- **Training multiple models per GPU**: Thanks to fully asynchronous
  thread-based data loading, you can now interleave training multiple models on
  the same GPU efficiently, without any data-loading overhead. See 
  [this guide]() for more info!  
- **Dedicated tools for image handling**: All the features above work are
  equally applicable to all sorts of machine learning models, but FFCV also
  offers some vision-specific features, such as fast JPEG encoding and decoding,
  storing datasets as mixtures of raw and compressed images to trade off I/O
  overhead and compute overhead, etc. See the [Working with images]() guide for
  more information!

<!-- for a 
more detailed look.
(`cv` denotes computer-vision specific features)

<p><b>Disk-read bottlenecks.</b> What if your GPUs sit idle from low disk throughput?
Maybe you're reading from a networked drive, maybe you have too many GPUs;
either way, try these features:
<ul>
<li><b><a href="TODO">Cache options</a></b>: use OS or process-level caching depending on whether your dataset fits in memory (or not).</li>
<li><b><a href="TODO">Use quasi-random data sampling</a></b>: </li>
<li><b><a href="TODO">Store resized images</a></b> (<code>cv</code>): Many datasets have gigantic images even though most pipelines crop and resize to smaller edge lengths before training.</li>
<li><b><a href="TODO">Store JPEGs</a></b> (<code>cv</code>): Store images as space-efficient JPEGs.</li>
<li><b><a href="TODO">Store lower quality JPEGs</a></b> (<code>cv</code>): Lower serialized JPEG quality to decrease storage sizes.</li>
</ul>
</p>

<p><b>CPU bottlenecks.</b> All CPUs at 100% and you're still not hitting maximal
GPU usage? Consider the following:
<ul>
<li><b><a href="TODO">Use premade, JIT-compiled augmentations</a></b>: `ffcv` comes with JIT-compiled equivalents of standard augmentation routines.</li>
<li><b><a href="TODO">Make your own JIT-compiled augmentations</a></b>: `ffcv` supports custom JIT-compiled augmentations.</li>
<li><b><a href="TODO">Store resized images</a></b> (<code>cv</code>): Smaller images require less compute to decode.</li>
<li><b><a href="TODO">Store lower quality JPEGs</a></b> (<code>cv</code>): Lower serialized JPEG quality to decrease CPU cycles spent decoding.</li>
<li><b><a href="TODO">Store a fraction of images as raw pixel data</a></b> (<code>cv</code>): Trade off storage and compute workload (raw pixels require no JPEG decoding) by randomly storing a specified fraction of the dataset as raw pixel data.</li>
</ul>
</p>

<p><b>GPU bottlenecks (any data).</b> Even if you're not bottlenecked by data
loading, <code>ffcv</code> can still accelerate your system:
<ul>
<li><b><a href="TODO">Asynchronous CPU-GPU data transfer</a></b>: While we always asynchronously transfer data, we also include tools for ensuring unblocked GPU execution.</li>
<li><b><a href="TODO">Train multiple models on the same GPU</a></b>: Fully
asynchronous dataloading means that different training processes won't block eachother.</li>
<li><b><a href="TODO">Offload compute to the CPU</a></b>: offload compute, like <a href="TODO">normalization</a> or <a href="">other augmentations</a>, onto the CPU.</li>
</ul>
This list is limited to what <code>ffcv</code> offers in data loading; check out
guides like <a href="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html">the PyTorch performance guide</a> for more ways to speed
up training.  -->

# Contributors

- Guillaume Leclerc
- Logan Engstrom
- Andrew Ilyas
- Sam Park
- Hadi Salman
