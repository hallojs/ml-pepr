ART Inference Attacks
=====================

.. currentmodule:: pepr.privacy.art_inference_wrapper

Base Wrapper Class
******************
The base class does not implement any attack. The :ref:`ART inference attack wrappers
<art_inference_wrappers>` inherit from the `BaseMembershipInferenceAttack` class
and have the same attributes.

.. autoclass:: BaseMembershipInferenceAttack

.. _art_inference_wrappers:

ART Inference Attack Wrappers
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