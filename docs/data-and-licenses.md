# Code and dataset licenses

The repository's code is [MIT licensed](../LICENSE) · Enrico Bianchi.
Third-party dependencies retain their own licenses.

### MNIST data

MNIST was created by Yann LeCun, Corinna Cortes and Christopher J. C. Burges
from NIST handwritten digits. Dataset licensing is documented inconsistently:
[TensorFlow's MNIST documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data)
attributes copyright to LeCun and Cortes and specifies
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), while the
[ylecun/mnist dataset card](https://huggingface.co/datasets/ylecun/mnist)
lists MIT. This project does not resolve that discrepancy or relicense MNIST.

The loader downloads the original IDX archives from the PyTorch-hosted mirror
and normalizes pixel values to [0, 1] in memory. Downloaded data is excluded from Git. Three locally trained model files are
included so the demo runs without retraining. Keep dataset attribution and applicable license terms when
redistributing MNIST or dataset adaptations; the code's MIT license does not
override those terms. This repository makes no separate licensing claim for
trained weights or generated images.
