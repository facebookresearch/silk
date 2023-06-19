
# Method

## Why are **successful round-trip matches** keypoints ?

We essentially redefine keypoints from first principle (in the same spirit as [DISK](https://arxiv.org/pdf/2006.13566.pdf) / [GLAMpoints](https://arxiv.org/pdf/1908.06812.pdf)).

The initial goal of keypoints is to be **distinct** and **robust** to reasonable viewpoint / photometric changes, so that they can be tracked across multiple frames. The research has focused on corners (e.g. Harris, SuperPoint) for a long time since **corners** are known to have those properties.

In our work however, we focus on learning keypoints to have those properties directly instead of relying on a proxy objective (i.e. learning "cornerness"). By measuring the round-trip success, we are essentially measuring the ability of a position to become a good keypoint (i.e. having the two properties mentioned above). Descriptors that are neither distinct nor robust are unlikely to match correctly, and therefore, this is a good signal to regress the keypoint score on. By extending the definition of keypoints, we observe that our model can not only capture corners, but also more complex patterns (e.g. curves, complex textures, ...).

[[21](https://github.com/facebookresearch/silk/issues/21)]

## If **all** the pixels in the image pair have **successful round-trip matches**, are they all keypoints ?

In theory yes, but that doesn't happen in practice. Images often contain large areas of uniform colors, or repetitive patterns. Given the local nature of keypoint descriptors, they do not contain enough information to obtain perfect matching.

[[21](https://github.com/facebookresearch/silk/issues/21)]

## At **early** training, if there are no **successful round-trip matches**, will there be **no positive samples** to train the **keypoint head** ?

Yes, there are no successful matches initially. So the keypoint head essentially converges towards outputting 0 everywhere, and the keypoint loss descreases accordingly (i.e. it's doing a good job at predicting every keypoints will fail matching). However, after a short while, the descriptors start to become more discriminative, successful matches become more frequent, and the keypoint loss start to increase until it stabilizes (i.e. learning which keypoints are likely to match becomes a harder problem to solve).

[[21](https://github.com/facebookresearch/silk/issues/21)]

## How are descriptors **normalized** ?

Raw descriptors coming from the descriptor head are **L2-normalized**. We simply divide them by their norm (i.e. $\frac{D}{||D||}$), which makes them lie on a **unit hypersphere**.

[[18](https://github.com/facebookresearch/silk/issues/18)]

# Performance

## Is SiLK a lot **slower** than SuperPoint ?

**SuperPoint** reports **70 FPS**, while we do report our **VGGnp-4** runs at **12 FPS**.

It's difficult to get a fair comparison of FPS across papers since there are multiple factors that could affect speed (implementation quality, hardware, ...). Our released FPS numbers are given to be compared relatively to each other to get a sense of relative speed between backbones, but should not be taken as absolute since speed is often a function of engineering effort (e.g. SiLK could be put on a chip and become orders of magnitude faster) and hardware.

That being said, we've ran an **additional SiLK VGG-4 vs SuperPoint comparison** to get some specific numbers. On 480x269 images (and a different machine than the one used in the paper), **SuperPoint** runs at **83 FPS** while SiLK runs at **30 FPS**. The large gap is explained by the **lack of downsampling layers** in SiLK. We don't consider that to be too bad, and would likely benefit from further architectural investigations (as **future work**).

Additionally, an interesting consequence of our architecture is the ability to become **more accurate when using a smaller resolution** (c.f. [supplementary](https://arxiv.org/pdf/2304.06194.pdf) Tab. 10), given an error threshold of 3. This shows that SiLK can still beat SuperPoint on the @3 metrics even when reducing the resolution by a factor of three. When doing so, **SiLK** gets a FPS of **68**, which becomes a lot closer to **SuperPoint**'s numbers.

[[21](https://github.com/facebookresearch/silk/issues/21)]

## How to **tune** SiLK ?

As most keypoint methods, SiLK also requires some tuning. But unlike other methods, our search space is quite small. We recommend tuning for :

* `Image size`. Too small and positions won't be accurate enough. Too large and the top-k keypoints from both image are less likely to overlap. There is usually a sweet spot. A sweep from 240 to 720 tend to find that sweet spot.
* `Top-k`. Too small and the keypoints won't be repeatable. Too large and it will cost more memory and computation. A coarse sweep between 1k to 30k should be good enough.
* `Checkpoint`. We provide two checkpoints for our VGG-4 backbone : `pvgg-4.ckpt` and `coco-rgb-aug.ckpt` (trained with different photometric augmentations). Try both and see which one performs best.
* `Matching Algorighm`. Using MN, ratio-test or double-softmax (with different thresholds) will also affect performance. We refer to our [supplementary](https://arxiv.org/pdf/2304.06194.pdf) (Tab. 12, Fig. 7) to estimate the effect of the matching algorithm on results. On IMC, we used a double-softmax with threshold of 0.99, and temperature of 0.1.
* `RANSAC/MAGSAC Parameters` (mostly the inlier threshold and max iterations).

[[16](https://github.com/facebookresearch/silk/issues/16),[18](https://github.com/facebookresearch/silk/issues/18),[19](https://github.com/facebookresearch/silk/issues/19)]

## Why do I have **poor performance** when using **small number** of keypoints (< 1k) ?

SiLK tends to provide **many good keypoints**, but those keypoints might not be ranked similarly accross different images. For example, if we have two images, and each have 5k good keypoints with a score around 0.9, there is **no guarantee** that the keypoint ranking will be conserved between those two images. Therefore selecting only 100 (2%) best keypoints out of 5k will likely result in little overlapping between the two sets, thus causing erroneous matching.

[[17](https://github.com/facebookresearch/silk/issues/17)]

# Implementation

## What is the training **input size** ? Why does the size **differ** between backbones ?

During training, we do use the [RandomCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomCrop.html) transform to sample **fixed sized crops**. The input size is specified [here](https://github.com/facebookresearch/silk/blob/9d80ebd81212c75d3638bd2d77c18bc48af7f5b5/etc/mode/train-silk.yaml#L89) and change according to the selected backbone. The change is necessary to control for output size (148x148 by default). Since we **do not** use padding, using the same input size with different backbones would produce different output sizes.

[[3](https://github.com/facebookresearch/silk/issues/3)]

## Why do I get **CUDA out-of-memory error** during **inference** ?

When running the matching algorithm, using $N$ keypoints, it requires to compute a similarity matrix, of size $N^2$. The quadratic growth of that matrix is likely the factor causing the **out-of-memory error** when using a large amount of keypoints.

[[6](https://github.com/facebookresearch/silk/issues/6)]

## Why does my output (dense) have a **different spatial size** than my input ? What is the **coordinate mapping** ?

Since SiLK doesn't use padding, applying a succession of convolutional layers will **"eat" the borders**. This explains the size reduction between the input and the output, and this reduction is backbone-dependent (since different backbones apply different number of convolutions).

This change in size (and potentially resolution when up/downsampling layers are present) can be challenging to handle **position mapping** (i.e. what position in the output space correspond to what position in the input space) when using multiple backbones. To solve this, we've automated some of the work by forcing our backbones to provide a `LinearCoordinateMapping` object, which handles that position mapping (in both directions).

[[7](https://github.com/facebookresearch/silk/issues/7)]

## Where are the **model checkpoints** & **tensorboard** files ? How to change it ?

All training sessions are run in a timestamped folder specified [here](https://github.com/facebookresearch/silk/blob/09ded25ec7673ce2084996a2376ccf7b54e1c87d/etc/hydra/local.yaml). This is a feature of the [hydra](https://hydra.cc/) library.
Any saved checkpoints and tensorboard files should be located in `./var/silk-cli/run/...`. Additionally, the path of the saved checkpoint should be displayed in the logs at the end of the training session.

[[13](https://github.com/facebookresearch/silk/issues/13)]

## Why aren't all checkpoints saved to disk ?

We do use a PyTorch Lightning callback to only save the best 10 checkpoints (evaluated on the validation set). The `save_top_k` option can be modified to increase the number of checkpoints saved (e.g. [here](https://github.com/facebookresearch/silk/blob/b601cbc5458c0e5504e75bff4e138532c1ce683a/etc/mode/train-silk.yaml#L38) for SiLK).

[[13](https://github.com/facebookresearch/silk/issues/13)]

## What is `SILK_SCALE_FACTOR` (=1.41) ?

`SILK_SCALE_FACTOR` is a scaling factor of the descriptors (i.e. desc <- SILK_SCALE_FACTOR * desc) done after unit normalization. A value of 1.41 (square root of 2) will essentially bring the descriptors on the hypersphere surface of radius 2. This is here for **legacy reasons**, and has **little importance** (with the exception mentioned below). It can simply be treated as a constant and does not need any tuning.

Moreover, it's important to note that changing that value won't change the MNN matching, nor the ratio-test. **BUT**, it will affect the double-softmax matching, as changing `SILK_SCALE_FACTOR` is similar to **changing the softmax temperature**.

[[18](https://github.com/facebookresearch/silk/issues/18),[22](https://github.com/facebookresearch/silk/issues/22)]
