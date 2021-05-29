ML-PePR: Pentesting Privacy and Robustness (Beta)
=================================================

|docs_pages_workflow| |publish_pypi_workflow| |black| |python_versions|

.. |docs_pages_workflow| image:: https://github.com/hallojs/ml-pepr/workflows/docs_pages_workflow/badge.svg?branch=master
    :target: https://github.com/hallojs/ml-pepr/actions/workflows/docs_pages_workflow.yml

.. |publish_pypi_workflow| image:: https://github.com/hallojs/ml-pepr/workflows/publish_pypi_workflow/badge.svg?branch=master
    :target: https://github.com/hallojs/ml-pepr/actions/workflows/publish_pypi_workflow.yml

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |python_versions| image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://www.python.org/downloads/release/python-360/

ML-PePR stands for Machine Learning Pentesting for Privacy and Robustness and is a Python library for evaluating machine
learning models. PePR is easily extensible and hackable. PePR's attack runner allows structured pentesting, and the
report generator produces straightforward privacy and robustness reports (LaTeX/PDF) from the attack results.

`Full documentation :books: including a quick-start guide :runner::dash: <https://hallojs.github.io/ml-pepr/>`_

**Caution, we cannot guarantee the correctness of PePR. Always do check the plausibility of your results!**

Installation
------------

We offer various installation options. Follow the instructions below to perform the desired installation. If you want to
install the latest developer version, please use the `code-repository <https://github.com/hallojs/ml-pepr>`_ of this
library. The current release is only tested with Python 3.6.

Repository
~~~~~~~~~~

1. Clone the repository.
2. Cd to project directory: ``cd ml-pepr``
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

Attack Catalog
--------------
PePR offers the following attacks:

+------------------------------------------------------------+------------------------+--------------+
| Attack                                                     | Type                   | Google Colab |
+============================================================+========================+==============+
| [1]_ Membership Inference Attack (mia)                     | Privacy (Black Box)    | |nb0|_       |
+------------------------------------------------------------+------------------------+--------------+
| [2]_ Direct Generalized Membership Inference Attack (gmia) | Privacy (Black Box)    | |nb1|_       |
+------------------------------------------------------------+------------------------+--------------+
| [3]_ Foolbox Attacks                                       | Robustness             | |nb2|_       |
+------------------------------------------------------------+------------------------+--------------+
| [4]_ Adversarial Robustness Toolbox (ART) Attacks          | Robustness             | |nb3|_       |
+------------------------------------------------------------+------------------------+--------------+

.. |nb0| image:: https://colab.research.google.com/assets/colab-badge.svg
.. _nb0: https://colab.research.google.com/github/hallojs/ml-pepr/blob/master/notebooks/mia_tutorial.ipynb

.. |nb1| image:: https://colab.research.google.com/assets/colab-badge.svg
.. _nb1: https://colab.research.google.com/github/hallojs/ml-pepr/blob/master/notebooks/direct_gmia_tutorial.ipynb

.. |nb2| image:: https://colab.research.google.com/assets/colab-badge.svg
.. _nb2: https://colab.research.google.com/github/hallojs/ml-pepr/blob/master/notebooks/foolbox_tutorial.ipynb

.. |nb3| image:: https://colab.research.google.com/assets/colab-badge.svg
.. _nb3: https://colab.research.google.com/github/hallojs/ml-pepr/blob/master/notebooks/art_tutorial.ipynb

License
-------
The entire content of the repository is licensed under GPLv3. The workflow for generating the documentation was
developed by `Michael Altfield <https://github.com/maltfield/rtd-github-pages>`_ and was modified and extended for our
purposes.

References
----------
.. [1] Shokri, Reza, et al. "Membership inference attacks against machine learning models." 2017 IEEE Symposium on
   Security and Privacy (SP). IEEE, 2017.

.. [2] Long, Yunhui, et al. "Understanding membership inferences on well-generalized learning models." arXiv preprint
   arXiv:1802.04889 (2018).

.. [3] `Foolbox <https://github.com/bethgelab/foolbox>`_: A Python toolbox to create adversarial examples that fool
    neural networks in PyTorch, TensorFlow, and JAX.

.. [4] `ART <https://github.com/Trusted-AI/adversarial-robustness-toolbox>`_: Adversarial Robustness Toolbox (ART)
    - Python Library for Machine Learning Security - Evasion, Poisoning, Extraction, Inference - Red and Blue Teams
