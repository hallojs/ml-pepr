ART Evasion Attacks
===================

.. currentmodule:: pepr.robustness.art_wrapper

You can find a basic example notebook `here <https://colab.research.google.com/github/hallojs/ml-pepr/blob/master/notebooks/art_tutorial.ipynb>`_.

.. autosummary::
    :nosignatures:

    AdversarialPatch

    AutoAttack
    AutoProjectedGradientDescent
    BoundaryAttack
    BrendelBethgeAttack
    CarliniL2Method
    CarliniLInfMethod
    DeepFool
    ElasticNet
    FastGradientMethod
    FeatureAdversaries
    FrameSaliencyAttack
    HopSkipJump
    BasicIterativeMethod
    ProjectedGradientDescent
    NewtonFool
    PixelAttack
    ThresholdAttack
    SaliencyMapMethod
    SimBA
    SpatialTransformation
    SquareAttack
    TargetedUniversalPerturbation
    UniversalPerturbation
    VirtualAdversarialMethod
    ZooAttack

.. note:: PePR only supports ART attacks that can handle the ``KerasClassifier`` and
    image input. The Imperceptible ASR Attack for example is not supported because it
    expects a speech recognition estimator.

.. autoclass:: AdversarialPatch

.. autoclass:: AutoAttack
.. autoclass:: AutoProjectedGradientDescent
.. autoclass:: BoundaryAttack
.. autoclass:: BrendelBethgeAttack
.. autoclass:: CarliniL2Method
.. autoclass:: CarliniLInfMethod
.. autoclass:: DeepFool
.. autoclass:: ElasticNet
.. autoclass:: FastGradientMethod
.. autoclass:: FeatureAdversaries
.. autoclass:: FrameSaliencyAttack
.. autoclass:: HopSkipJump
.. autoclass:: BasicIterativeMethod
.. autoclass:: ProjectedGradientDescent
.. autoclass:: NewtonFool
.. autoclass:: PixelAttack
.. autoclass:: ThresholdAttack
.. autoclass:: SaliencyMapMethod
.. autoclass:: SimBA
.. autoclass:: SpatialTransformation
.. autoclass:: SquareAttack
.. autoclass:: TargetedUniversalPerturbation
.. autoclass:: UniversalPerturbation
.. autoclass:: VirtualAdversarialMethod
.. autoclass:: ZooAttack