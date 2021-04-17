Attack Runner
=============

The attack runner is a script to execute multiple attacks at once. It creates one attack
report containing the results of all executed attacks.

.. seealso:: An attack runner tutorial notebook is available `here <https://colab.research.google.com/github/hallojs/ml-pepr/blob/master/notebooks/attack_runner_tutorial/attack_runner_tutorial.ipynb>`_.

Module Documentation
--------------------

.. automodule:: pepr.attack_runner
    :members:

Saved Attack Objects
--------------------
The attack runner saves all created attack objects into the specified directory
(``attack_obj_save_path``) using
`Python Pickle <https://docs.python.org/3.6/library/pickle.html>`_. When unpickling
these objects, make sure, that the corresponding folders with the "tm" suffix are
available in the same directory, because they include the target models.

.. note:: Currently, the attack runner has no functionality of reusing the attack
    objects for example to avoid redundant training when no parameters changed.

Attack Runner Configuration
---------------------------
The attack runner configuration file is an YAML file that specifies everything that is
needed to run an attack. There are two required main keys for every attack:

- ``attack_pars``
- ``data_conf``

Attack Parameters (``attack_pars``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Attack Independent Keys
"""""""""""""""""""""""
``attack_pars`` is a list of attack parameters for every attack to run. Every list entry
must have the following not attack dependent keys:

- ``attack_type``
- ``attack_alias``
- ``path_to_dataset_data``
- ``path_to_dataset_labels``
- ``target_models``

``attack_type`` specifies the type of the attack (e.g. ``"mia"`` or ``"gmia"``).
``attack_alias`` defines a unique name for the attack configuration. It is saved in the
attack object and is the heading of the subsection in the report.
``path_to_dataset_data`` and ``path_to_dataset_labels`` each defining a path to a
serialized numpy array. ``target_models`` defines an array of paths where the serialized
target models are saved.

Example:

.. code-block:: yaml
    :linenos:

    attack_pars:
    - attack_type: "mia"
      attack_alias: "MIA Default"
      path_to_dataset_data: "datasets/cifar100_data.npy"
      path_to_dataset_labels: "datasets/cifar100_labels.npy"
      target_model_paths:
      - "data/target_model_mia0"
      - "data/target_model_mia1"
      # ... attack dependent keys

Attack Dependent Keys
"""""""""""""""""""""
One list entry consists of a dictionary with all attack parameters for the desired
attack. All keys of the ``attack_pars`` dictionary that are passed to the corresponding
attack constructor, must be also defined in the list entry.

Listing all attack dependent keys per attack:

- MIA
    - ``number_shadow_models``
    - ``shadow_training_set_size``
    - ``number_classes``
    - ``create_compile_shadow_model``
    - ``shadow_epochs``
    - ``shadow_batch_size``
    - ``create_compile_attack_model``
    - ``attack_epochs``
    - ``attack_batch_size``
- GMIA
    - ``number_reference_models``
    - ``reference_training_set_size``
    - ``number_classes``
    - ``create_compile_model``
    - ``reference_epochs``
    - ``reference_batch_size``
    - ``hlf_metric``
    - ``hlf_layer_number``
    - ``number_target_records``

Data Configuration (``data_conf``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``data_conf`` is a list of data configurations for every attack. The order must be the
same as the order of ``attack_pars``. The required keys are equal to the ``data_conf``
dictionary that is passed to the attack constructor but with the suffix ``_path`` added
to them as they are paths to serialized numpy arrays.

Listing all keys per attack:

- MIA
    - ``shadow_indices_path``
    - ``target_indices_path``
    - ``evaluation_indices_path``
    - ``record_indices_per_target_path``
- GMIA
    - ``reference_indices_path``
    - ``target_indices_path``
    - ``evaluation_indices_path``
    - ``record_indices_per_target_path``

Example Configuration
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: yaml
    :linenos:

    # Attack Parameters
    attack_pars:
      - attack_type: "mia"                                            # --
        attack_alias: "MIA Tutorial"                                  #  |
        path_to_dataset_data: "datasets/cifar100_data.npy"            #  | Attack independent
        path_to_dataset_labels: "datasets/cifar100_labels.npy"        #  | parameters
        target_model_paths:                                           #  |
        - "data/target_model_mia"                                     # --
        number_shadow_models: 100                                     # --
        shadow_training_set_size: 2500                                #  |
        number_classes: 100                                           #  |
        create_compile_shadow_model: "create_compile_shadow_model"    #  |
        shadow_epochs: 100                                            #  | MIA
        shadow_batch_size: 50                                         #  | parameters
        create_compile_attack_model: "create_compile_attack_model"    #  |
        attack_epochs: 50                                             #  |
        attack_batch_size: 50                                         # --
      - attack_type: "gmia"                                           # --
        attack_alias: "GMIA Tutorial"                                 #  |
        path_to_dataset_data: "datasets/fmnist_data.npy"              #  | Attack independent
        path_to_dataset_labels: "datasets/fmnist_labels.npy"          #  | parameters
        target_model_paths:                                           #  |
        - "data/target_model_gmia"                                    # --
        number_reference_models: 100                                  # --
        reference_training_set_size: 10000                            #  |
        number_classes: 10                                            #  |
        create_compile_model: "create_compile_reference_model"        #  |
        reference_epochs: 50                                          #  | GMIA
        reference_batch_size: 50                                      #  | parameters
        hlf_metric: "cosine"                                          #  |
        hlf_layer_number: 10                                          #  |
        number_target_records: 25                                     # --

    # Data Configuration
    data_conf:
      - shadow_indices_path: "datasets/shadow_indices.npy"
        target_indices_path: "datasets/target_indices.npy"
        evaluation_indices_path: "datasets/evaluation_indices.npy"
        record_indices_per_target_path: "datasets/record_indices_per_target.npy"
      - reference_indices_path: "datasets/refenrence_indices.npy"
        target_indices_path: "datasets/target_indices.npy"
        evaluation_indices_path: "datasets/evaluation_indices.npy"
        record_indices_per_target_path: "datasets/record_indices_per_target.npy"