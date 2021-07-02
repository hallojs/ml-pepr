Foolbox Attacks
===============

.. currentmodule:: pepr.robustness.foolbox_wrapper

.. note:: You can find a basic example notebook `here <https://colab.research.google.com/github/hallojs/ml-pepr/blob/master/notebooks/foolbox_tutorial.ipynb>`_.

Base Wrapper Class
******************
The base class does not implement any attack. The :ref:`Foolbox attack wrappers
<Foolbox Attack Wrappers>` inherit from the `BaseAttack` class and have the same
attributes.

.. autoclass:: BaseAttack

Foolbox Attack Wrappers
***********************
.. autosummary::
    :nosignatures:

    L2ContrastReductionAttack
    VirtualAdversarialAttack
    DDNAttack
    L2ProjectedGradientDescentAttack
    LinfProjectedGradientDescentAttack
    L2BasicIterativeAttack
    LinfBasicIterativeAttack
    L2FastGradientAttack
    LinfFastGradientAttack

    L2AdditiveGaussianNoiseAttack
    L2AdditiveUniformNoiseAttack
    L2ClippingAwareAdditiveGaussianNoiseAttack
    L2ClippingAwareAdditiveUniformNoiseAttack
    LinfAdditiveUniformNoiseAttack
    L2RepeatedAdditiveGaussianNoiseAttack
    L2RepeatedAdditiveUniformNoiseAttack
    L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack
    L2ClippingAwareRepeatedAdditiveUniformNoiseAttack
    LinfRepeatedAdditiveUniformNoiseAttack
    InversionAttack
    BinarySearchContrastReductionAttack
    LinearSearchContrastReductionAttack

    L2CarliniWagnerAttack
    NewtonFoolAttack
    EADAttack
    GaussianBlurAttack
    L2DeepFoolAttack
    LinfDeepFoolAttack
    SaltAndPepperNoiseAttack
    LinearSearchBlendedUniformNoiseAttack
    BinarizationRefinementAttack
    BoundaryAttack
    L0BrendelBethgeAttack
    L1BrendelBethgeAttack
    L2BrendelBethgeAttack
    LinfinityBrendelBethgeAttack

    FGM
    FGSM
    L2PGD
    LinfPGD
    PGD

.. note:: The ``DatasetAttack`` is currently not supported by PePR.

.. autoclass:: L2ContrastReductionAttack
.. autoclass:: VirtualAdversarialAttack
.. autoclass:: DDNAttack
.. autoclass:: L2ProjectedGradientDescentAttack
.. autoclass:: LinfProjectedGradientDescentAttack
.. autoclass:: L2BasicIterativeAttack
.. autoclass:: LinfBasicIterativeAttack
.. autoclass:: L2FastGradientAttack
.. autoclass:: LinfFastGradientAttack

.. autoclass:: L2AdditiveGaussianNoiseAttack
.. autoclass:: L2AdditiveUniformNoiseAttack
.. autoclass:: L2ClippingAwareAdditiveGaussianNoiseAttack
.. autoclass:: L2ClippingAwareAdditiveUniformNoiseAttack
.. autoclass:: LinfAdditiveUniformNoiseAttack
.. autoclass:: L2RepeatedAdditiveGaussianNoiseAttack
.. autoclass:: L2RepeatedAdditiveUniformNoiseAttack
.. autoclass:: L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack
.. autoclass:: L2ClippingAwareRepeatedAdditiveUniformNoiseAttack
.. autoclass:: LinfRepeatedAdditiveUniformNoiseAttack
.. autoclass:: InversionAttack
.. autoclass:: BinarySearchContrastReductionAttack
.. autoclass:: LinearSearchContrastReductionAttack

.. autoclass:: L2CarliniWagnerAttack
.. autoclass:: NewtonFoolAttack
.. autoclass:: EADAttack
.. autoclass:: GaussianBlurAttack
.. autoclass:: L2DeepFoolAttack
.. autoclass:: LinfDeepFoolAttack
.. autoclass:: SaltAndPepperNoiseAttack
.. autoclass:: LinearSearchBlendedUniformNoiseAttack
.. autoclass:: BinarizationRefinementAttack
.. autoclass:: BoundaryAttack
.. autoclass:: L0BrendelBethgeAttack
.. autoclass:: L1BrendelBethgeAttack
.. autoclass:: L2BrendelBethgeAttack
.. autoclass:: LinfinityBrendelBethgeAttack

.. autoclass:: FGM
.. autoclass:: FGSM
.. autoclass:: L2PGD
.. autoclass:: LinfPGD
.. autoclass:: PGD