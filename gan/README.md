# Generative Adversarial Network

## Datasets
[LFW](http://vis-www.cs.umass.edu/lfw/). The input images are center cropped by 128 x 128, then resized to 64 x 64.

## GAN Setting in details
See [the model](gan/models)

## Discussion
- Adam with very low learning rate, e.g., 1. * 1e-5; otherwise easily fail to nan.
- Training G by maximizing log D(G(z)) instead could work well in my case.
- Generated images still remain blurring, we might need a more descriptive generator, e.g., increasing the number of parameters.
- Human face is very fine-grained, such that difficult to capture and to generate only uinsg random seed?

## Generated Image Examples

![Epoch=001](./images/001/00000.png "Epoch=001")
![Epoch=050](./images/050/00000.png "Epoch=050")
![Epoch=100](./images/100/00000.png "Epoch=100")
![Epoch=200](./images/200/00000.png "Epoch=200")
![Epoch=300](./images/300/00000.png "Epoch=300")
![Epoch=400](./images/400/00000.png "Epoch=400")
![Epoch=500](./images/500/00000.png "Epoch=500")


