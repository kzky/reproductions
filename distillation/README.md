# Distillation Example

In this example, Distillation technique is used for neural network compactization. the technique, Distiilation, is not limited to be used for this purpose; in other words, one could distil knowledges a network learned to other network, in other usecases, featurre learning, network quantizatoin, etc.

The follows show the workflow when one would like to compactize a network by using `Distillation`.

1. Train a teacher network.
2. Distil that knowledge to another network, usually called student.

`classification.py` corresponds to the first one, and `distillation.py` does the second.






