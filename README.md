# LGSGMM

Official implementation of the paper: **“Two-Stage Multi-Drone Multi-Target Association: Laplacian-Guided Feature Enhancement and SGMM-Based Error Filtering Approach.”**

## Method update

We have substantially improved the method described in our paper. The feature enhancement network has been replaced with LightGlue, using the pretrained weight file `weights/superpoint_lightglue.pth`. The model is then trained on the MDMT dataset.

Our experiments show that the resulting model achieves an MDA score above 0.4 and has strong generalization performance. **Association results on our dataset** is a test image from one of our in-house engineering projects.

test_weight.pth  is the test weight file and can be directly used for evaluation.

## Association results on our dataset

![Association results on our dataset](<Association%20results%20on%20our%20dataset.png>)

## Training configuration

- Set the batch size to `bt=8` during training.
- Load the pretrained LightGlue weight file `weights/superpoint_lightglue.pth`.
- Keep the learning rate and decay schedule exactly as defined in the code. Modifying either setting can substantially reduce performance.

The [training loss curve](<train%20loss.png>) shows stable convergence and can be used as a reference.

## Usage

We welcome the use of this model in both academic and engineering projects. In our tests, the model can still associate previously unseen target categories that do not appear in the MDMT dataset, including military targets, with good accuracy.
