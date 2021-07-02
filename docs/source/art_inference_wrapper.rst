ART Inference Attacks
=====================

.. currentmodule:: pepr.privacy.art_inference_wrapper

Base wrapper class
******************
The base class does not implement a any attack. The ART inference attack wrappers
inherit from the `BaseMembershipInferenceAttack` class and have the same attributes.

.. autoclass:: BaseMembershipInferenceAttack

ART Inference Attack wrappers
*****************************
.. autosummary::
    :nosignatures:

    MembershipInferenceBlackBox
    MembershipInferenceBlackBoxRuleBased
    LabelOnlyDecisionBoundary
    MIFace

.. autoclass:: MembershipInferenceBlackBox
.. autoclass:: MembershipInferenceBlackBoxRuleBased
.. autoclass:: LabelOnlyDecisionBoundary
.. autoclass:: MIFace