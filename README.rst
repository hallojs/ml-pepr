ML-PePR: Pentesting Privacy and Robustness [alpha]
=====================================================

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/hallojs/ml-pepr/master

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: pybadges/python_version.svg
    :target: https://www.python.org

PePR [ˈpɛpɚ] is a library for pentesting the privacy risk and robustness of machine learning models.

**Caution, this is a alpha version. Always check the plausibility of your results!**

Installation
------------
We offer various installation options. Follow the instructions below to perform the desired installation. If you want to
install the latest developer version please use the code-repository of this library. The current release is only tested
with Python 3.7.9 and only supports TensorFlow.

Via Repository
~~~~~~~~~~~~~~
1. Clone the repository: ``git clone https://github.com/hallojs/ml-pepr.git``
2. Cd to project directory: ``cd ml-pepr``
3. Run in the terminal: ``pip install -e .``

*Hint: If you want to use ``pip install -e .`` in Google Colab, you must restart the runtime before you can
``import pepr``.*

Via PyPi
~~~~~~~~
Coming soon!


As Docker
~~~~~~~~~
Coming soon!


Basic Usage
-----------
PePR offers the following options to structure privacy and/or robustness pentesting:

1. Run a single attack from the attack catalog with a single attack configuration

    1.1a Run the attack on a single target model

    1.1b Or run the attack on multiple target models to get a more general result, that is averaged over multiple target
    model instances and target training data sets (see also ``pepr.utilities.assign_record_ids_to_target_models`` for
    a utility function to generate appropriate training data sets)

    1.2a Generate a privacy and robustness report for just this attack configuration and the attacked target model

    1.2b Or generate a report section which can later be combined with other report sections to a more extensive report
    containing results of multiple attack types and attack configurations (see ``pepr.report.report_generator``)

2. Run multiple attacks from the attack catalog with a single or multiple attack configurations [work in progress]

    2.1 Write an attack configuration (YAML format) for the attack runner

    2.2a Run the attacks on a single target model

    2.2b Or run the attack on multiple target models to get a more general result, that is averaged over multiple target
    model instances and target training data sets (see also ``pepr.utilities.assign_record_ids_to_target_models`` for
    a utility function for generating appropriate training data sets)

    2.3 Generate a privacy and robustness report for all or a selection of the performed attacks

    2.4 Optional: Rerun a selection of attacks with a new attack configuration to optimize the attack results


Example Notebooks
-----------------
* direct_gmia_tutorial: A first really simple example notebook of using the direct membership inference attack on a
  | single target model with a single attack configuration.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://github.com/hallojs/ml-pepr/blob/master/notebooks/direct_gmia_tutorial.ipynb

Attack Catalog
--------------
PePR offers the following attacks:

+------------------------------------------------------------+---------------------+
| Attack                                                     | Type                |
+============================================================+=====================+
| [1]_ Membership Inference Attack (Coming Soon!)            | Privacy (Black Box) |
+------------------------------------------------------------+---------------------+
| [2]_ Direct Generalized Membership Inference Attack (gmia) | Privacy (Black Box) |
+------------------------------------------------------------+---------------------+

References
----------
.. [1] Shokri, Reza, et al. "Membership inference attacks against machine learning models." 2017 IEEE Symposium on
   Security and Privacy (SP). IEEE, 2017.

.. [2] Long, Yunhui, et al. "Understanding membership inferences on well-generalized learning models." arXiv preprint
   arXiv:1802.04889 (2018).
