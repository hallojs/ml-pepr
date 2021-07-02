ART Extraction Attacks
======================

.. currentmodule:: pepr.privacy.art_extraction_wrapper

Base Wrapper Class
******************
The base class does not implement any attack. The :ref:`ART extraction attack wrappers
<ART Extraction Attack Wrappers>` inherit from the `BaseExtractionAttack` class
and have the same attributes.

.. autoclass:: BaseExtractionAttack

ART Extraction Attack Wrappers
******************************
.. autosummary::
    :nosignatures:

    CopycatCNN
    KnockoffNets

.. autoclass:: CopycatCNN
.. autoclass:: KnockoffNets