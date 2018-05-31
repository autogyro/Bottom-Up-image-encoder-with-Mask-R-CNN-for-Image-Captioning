# Bottom-Up-Attention with Mask R-CNN for Neural Image Captioner

This code merges Mask R-CNN into the Bottom-Up encoder for Bottom-Up and Top-Down (BUTD) Attention Neural Image Captioner. It is adapated from the following arts:
- Original BUTD Image Captioner paper: http://www.panderson.me/images/1707.07998-up-down.pdf
- BUTD Image Captioner implementation for the author: https://github.com/peteanderson80/bottom-up-attention
- Original Mask R-CNN paper:
https://arxiv.org/abs/1703.06870
- Mask R-CNN implementaion from the FacebookResearch: https://github.com/facebookresearch/Detectron

The main contributions of this work are:
- Merging [MS COCO Captioning](http://cocodataset.org/#home "MS COCO Captioning") into the [Visual Genome Dataset](http://visualgenome.org/ "Visual Genome Dataset").
- Integrates Mask R-CNN into the original BUTD bottom-up encoder (except for the Feature Pyramid Network).

The current milestone is:

- [x] Generates MS COCO mask along with the original Visual Genome dataset.
- [x] Dynamically loads MS COCO mask during runtime.
- [x] Dynamically filters MS COCO images and non-segmented area during runtime.
- [x] Migrates Region of Interest Align Layer into caffe.
- [x] Adds a Masked Cross Entropy Loss Layer into caffe.
- [ ] Trains the network with fine tuning the pre-trained BUTD model.
- [ ] Trains the network in a complete run.
- [ ] Trains the network on different dataset, e.g. with individual MS COCO dataset.
- [ ] Replace ResNet-101 with ResNeXt-101 for further experiments.

**Note1**: For now, unfortunately, I do not have any GPU resource for a complete model training. This code fine tunes the [BUTD trained model](https://github.com/peteanderson80/bottom-up-attention "BUTD trained model") with additional Mask Loss.
**Note2**: You might want to use the caffe implementation from this repository, as self-customized layers are included.


------------


### Reference
If you use this code, please reference this github link.

------------


### Disclaimer

This code is modified from the [BUTD Image Captioner](https://github.com/peteanderson80/bottom-up-attention "BUTD Image Captioner") and [Mask R-CNN](https://arxiv.org/abs/1703.06870 "and Mask R-CNN"). For the details of installation, implementation and references, please refer to the github repository from the original authors. This README.md mainly focuses on differtiating my code from original implementation.

------------

### Contents
1. [Mask Generator](#mask-generator)
2. [Mask Branch](#mask-branch)
3. [Caffe](#caffe)
4. [Expected Testing Result](#expected-testing-result)

### Mask Generator

The original dataset benchmarks the image captioner on MS COCO dataset, but the bottom-up encoder is separately trained on Visual Genome (VG) dataset. Comparing to MS COCO, it contains larger sets of object catagories (BUTD uses **1600** top-objects from VG v.s. **80** object classes from MS COCO). About half of the images in VG comes from MS COCO.

However, VG only provides bbox labels. To train Mask R-CNN with VG, I choose to merge MS COCO into VG.

0. **`data/genome/setup_vg.py`** is modified. It generates images into pascal voc format with additional 28x28 mask for each image.

1. **`lib/datasets/vg.py`** is modified. It loads a 28x28 mask for each image as well as corresponding MS COCO class index.

2. **`lib/roi_data_layer/layer.py`** is modified as a python data layer for caffe. It creates space for mask data. It also filters MS COCO dataset so only MS COCO images are used for training. Note that this filter must be removed before a complete training or testing.

3. **`lib/roi_data_layer/minibatch.py`** is modified to load data. It expands the original 28x28 mask to a 81x28x28 mask for each image.

4. **`lib/rpn/proposal_target_layer.py`** is modified. It filters non-segmented bbox. Note that this filter must be removed before a complete training.

5. **`lib/fast_rcnn/bbox_transform.py`** is modified. It slides mask depending on the bounding box prediction.


------------


### Mask Branch

**`models/vg/ResNet-101/faster_rcnn_end2end_final/train.prototxt`** is modified to creates a mask branch with 4 convolution layers and a mask loss output. Its architecture follows from the original Mask R-CNN paper.

Note that the original Mask R-CNN achieved the best performance with Feature Pyramid Network and ResXNet-101. In this implemenetation, only Region Align Layer and Mask Loss are included.

------------

### Caffe

To output appropriate loss, the original sigmoid cross entropy loss layer is not enough. Hence, self-customized layers are defined.

0. **`caffe/src/caffe/layers/roi_align_layer.cu`** is modified. It migrates the caffe2 implementation from the FacebookRearch.
1. **`caffe/src/caffe/layers/sigmoid_cross_entropy_loss_hongyuan_layer.cu`** is modified. It outputs a masked loss for the target MS COCO object class only.

### Expected Testing Result

Not available for now.
