ML-PePR (Beta)
==============

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contents:

   quick_start_guide
   privacy_attacks
   robustness_attacks
   attack_runner
   utilities

ML-PePR stands for Machine Learning Pentesting for Privacy and Robustness and is a Python library for evaluating machine
learning models. PePR is easily extensible and hackable. PePR's attack runner allows structured pentesting, and the
report generator produces straightforward privacy and robustness reports (LaTeX/PDF) from the attack results.

.. warning:: Caution, we cannot guarantee the correctness of PePR. Always do check the plausibility of your results!

Installation
------------

We offer various installation options. Follow the instructions below to perform the desired installation. If you want to
install the latest developer version, please use the `code-repository <https://github.com/hallojs/ml-pepr>`_ of this
library. The current release is only tested with Python 3.6.

Repository
~~~~~~~~~~

1. Clone the repository.
2. Cd to project directory: ``cd mlpepr``
3. Run in the terminal: ``pip install .``

PyPi
~~~~

`pypi <https://pypi.org/project/mlpepr/>`_-typical: ``pip install mlpepr``

Docker
~~~~~~

To use PePR inside a docker container, build a CPU or GPU image. Note that your system must be set up for GPU use:
`TensorFlow Docker Requirements <https://www.tensorflow.org/install/docker>`_.

Build the docker image:

1. Clone the repository: ``git clone https://github.com/hallojs/ml-pepr.git``
2. Cd to project directory: ``cd ml-pepr``
3. Build the docker image: ``docker build -t <image name> . -f Dockerfile-tf-<cpu or gpu>``

.. note:: The current version is not tested under Windows and only supports TensorFlow at the Moment!
