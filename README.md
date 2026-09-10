# LGSGMM

Official implementation of the paper: **“Two-Stage Multi-Drone Multi-Target Association: Laplacian-Guided Feature Enhancement and SGMM-Based Error Filtering Approach.”**

## Method update

We have substantially improved the method in our paper. The feature enhancement network has been replaced with LightGlue, using the pretrained weight file `weights/superpoint_lightglue.pth`. The model is then trained on the MDMT dataset.

Our experiments show that the resulting model achieves an MDA score above 0.4 and has strong generalization performance. The section “Association results on our dataset” presents a side-view scene from one chapter of our in-house engineering project.

We welcome the use of this model in academic and engineering projects. In our tests, it can still associate previously unseen target categories from the MDMT dataset, including military targets, with good accuracy.
