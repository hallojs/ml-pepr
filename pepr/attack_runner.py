"""
Attack runner for convenient pentesting with multiple configurations.
"""

import os
import logging
import pickle

import numpy as np
import tensorflow as tf
import yaml
from itertools import groupby

from pepr import report
from pepr.privacy import mia, gmia

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_attacks(yaml_path, report_save_path, attack_obj_save_path, pdf, functions):
    """
    Run multtiple attacks configured in a YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to attack configuration file in YAML format.
    report_save_path : str
        Path where to save the report files.
    attack_obj_save_path : str
        Path where to save the serialized attack objects.
    pdf : bool
        If true, the attack runner will build the LaTex report and save an PDF to the
        save path.
    functions : dict
        Dictionary containing the functions which return TensorFlow models. The keys
        should correspond to the strings in the configuration file.

    Returns
    -------
    dict
        Dictionary with attack alias and the corossponding path to the serialized attack
        object.
    """

    attack_object_paths = {}

    # Pickle helper
    def pickle_attack_obj(attack_obj):
        with open(
            f"{attack_obj_save_path}/{attack_obj.attack_alias}.pickle", "wb"
        ) as f:
            attack_obj.obj_save_path = attack_obj_save_path
            pickle.dump(attack_obj, f)
            attack_object_paths[
                f"{attack_obj.attack_alias}"
            ] = f"{attack_obj_save_path}/{attack_obj.attack_alias}.pickle"

    # Parse YAML
    with open(yaml_path) as f_stream:
        logger.info(f"Parse attack runner configuration file: {yaml_path}")
        data = yaml.load(f_stream)
        attack_obj_save_path

        os.makedirs(attack_obj_save_path, exist_ok=True)
        os.makedirs(report_save_path + "/fig", exist_ok=True)
        sections = []

        for i, yaml_attack_pars in enumerate(data["attack_pars"]):
            attack_type = yaml_attack_pars["attack_type"]
            logger.info(
                f"Attack number: {i+1}\n\n"
                "######################## Attack Run ########################"
            )
            if attack_type == "mia":
                # Parse MIA configuration
                logger.info("Setup configuration.")
                shadow_model_function = functions[
                    yaml_attack_pars["create_compile_shadow_model"]
                ]
                attack_model_function = functions[
                    yaml_attack_pars["create_compile_attack_model"]
                ]

                yaml_data_conf = data["data_conf"][i]
                shadow_indices = list(np.load(yaml_data_conf["shadow_indices_path"]))
                target_indices = list(np.load(yaml_data_conf["target_indices_path"]))
                evaluation_indices = list(
                    np.load(yaml_data_conf["evaluation_indices_path"])
                )
                record_indices_per_target = np.load(
                    yaml_data_conf["record_indices_per_target_path"]
                )

                target_models = []
                for path in yaml_attack_pars["target_model_paths"]:
                    target_models.append(tf.keras.models.load_model(path))

                # Setup dictionaries for attack
                attack_pars = {
                    "number_classes": yaml_attack_pars["number_classes"],
                    "number_shadow_models": yaml_attack_pars["number_shadow_models"],
                    "shadow_training_set_size": yaml_attack_pars[
                        "shadow_training_set_size"
                    ],
                    "create_compile_shadow_model": shadow_model_function,
                    "create_compile_attack_model": attack_model_function,
                    "shadow_epochs": yaml_attack_pars["shadow_epochs"],
                    "shadow_batch_size": yaml_attack_pars["shadow_batch_size"],
                    "attack_epochs": yaml_attack_pars["attack_epochs"],
                    "attack_batch_size": yaml_attack_pars["attack_batch_size"],
                }

                data_conf = {
                    "shadow_indices": shadow_indices,
                    "target_indices": target_indices,
                    "evaluation_indices": evaluation_indices,
                    "record_indices_per_target": record_indices_per_target,
                }

                # Create attack object
                logger.info(f"Attack: {yaml_attack_pars['attack_alias']}")
                attack = mia.Mia(
                    yaml_attack_pars["attack_alias"],
                    attack_pars,
                    np.load(yaml_attack_pars["path_to_dataset_data"]),
                    np.load(yaml_attack_pars["path_to_dataset_labels"]),
                    data_conf,
                    target_models,
                )

                attack.run()
                attack.create_attack_section(report_save_path)
                sections.append(attack.report_section)

                logger.debug("Serialize attack object.")
                pickle_attack_obj(attack)

            elif attack_type == "gmia":
                logger.info("Setup configuration.")
                # Parse MIA configuration
                model_function = functions[yaml_attack_pars["create_compile_model"]]

                yaml_data_conf = data["data_conf"][i]
                reference_indices = list(
                    np.load(yaml_data_conf["reference_indices_path"])
                )
                target_indices = list(np.load(yaml_data_conf["target_indices_path"]))
                evaluation_indices = list(
                    np.load(yaml_data_conf["evaluation_indices_path"])
                )
                record_indices_per_target = np.load(
                    yaml_data_conf["record_indices_per_target_path"]
                )

                target_models = []
                for path in yaml_attack_pars["target_model_paths"]:
                    target_models.append(tf.keras.models.load_model(path))

                # Setup dictionaries for attack
                attack_pars = {
                    "number_classes": yaml_attack_pars["number_classes"],
                    "number_reference_models": yaml_attack_pars[
                        "number_reference_models"
                    ],
                    "reference_training_set_size": yaml_attack_pars[
                        "reference_training_set_size"
                    ],
                    "create_compile_model": model_function,
                    "reference_epochs": yaml_attack_pars["reference_epochs"],
                    "reference_batch_size": yaml_attack_pars["reference_batch_size"],
                    "hlf_metric": yaml_attack_pars["hlf_metric"],
                    "hlf_layer_number": yaml_attack_pars["hlf_layer_number"],
                    "number_target_records": yaml_attack_pars["number_target_records"],
                }

                data_conf = {
                    "reference_indices": reference_indices,
                    "target_indices": target_indices,
                    "evaluation_indices": evaluation_indices,
                    "record_indices_per_target": record_indices_per_target,
                }

                # Create attack object
                logger.info(f"Attack: {yaml_attack_pars['attack_alias']}")
                attack = gmia.DirectGmia(
                    yaml_attack_pars["attack_alias"],
                    attack_pars,
                    np.load(yaml_attack_pars["path_to_dataset_data"]),
                    np.load(yaml_attack_pars["path_to_dataset_labels"]),
                    data_conf,
                    target_models,
                )

                attack.run()
                attack.create_attack_section(report_save_path)
                sections.append(attack.report_section)

                logger.debug("Serialize attack object.")
                pickle_attack_obj(attack)

        logger.info("Generate final report.")
        report.report_generator(report_save_path, sections, pdf=pdf)

    return attack_object_paths
