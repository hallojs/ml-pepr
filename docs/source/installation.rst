Installation
============

.. warning:: Caution, this is an alpha version. Always do check the plausibility of your results!

We offer various installation options. Follow the instructions below to perform the desired installation. If you want to
install the latest developer version please use the code-repository of this library. The current release is only tested
with Python 3.7.9.

Via Repository
--------------
1. Clone the repository.
2. Run in the terminal: ``pip install -e .``


Via PyPi
--------
Coming soon!


As Docker
---------
To use pepr inside a docker container build a cpu or gpu image. Note that your system must be set up for GPU use:
`TensorFlow Docker Requirements <https://www.tensorflow.org/install/docker>`_.

Build the docker image:

1. Clone the repository: ``git clone https://github.com/hallojs/ml-pepr.git``
2. Cd to project directory: ``cd ml-pepr``
3. Build the docker image: ``docker build -t <image name> . -f Dockerfile-tf-<cpu or gpu>``

.. note:: The current version is not tested under Windows and only supports TensorFlow at the Moment!
