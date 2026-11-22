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
and normalizes pixel values to [0, 1] in memory. Downloaded data is excluded from
Git. Three locally trained model files are included so the demo runs without
retraining. The code's MIT license does not override applicable dataset terms.

### Model weights and images

As a conservative sharing policy, the project author's rights, if any, in the
three files under `weights/`, generated sample images, and MNIST visualizations
retained in Git history are licensed under
[Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/).
This includes the historical `vqvae.png`, which combines MNIST examples,
reconstructions and visualizations. MNIST credit and underlying rights remain
with its creators, as described above.

When sharing these assets or adaptations, retain attribution to Enrico Bianchi
and the MNIST creators, link to the license, identify modifications, and comply
with applicable ShareAlike requirements. This grant covers only rights the
project author holds; it does not relicense third-party material, resolve the
conflicting MNIST license labels, or assert that every trained model or generated
image is necessarily copyrightable or an adaptation of the training data.
