"""
Attack runner for convenient pentesting with multiple configurations.
"""

import os
import logging
import pickle

import numpy as np
import tensorflow as tf
import yaml

from pepr import report
from pepr.privacy import mia, gmia

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_attacks(yaml_path, attack_obj_save_path, functions):
    """
    Run multiple attacks configured in a YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to attack configuration file in YAML format.
    attack_obj_save_path : str
        Path where to save the serialized attack objects.
    functions : dict
        Dictionary containing the functions which return TensorFlow models. The keys
        should correspond to the strings in the configuration file.

    Returns
    -------
    dict
        Dictionary with attack alias and the corresponding path to the serialized attack
        object in the same order as specified in the YAML configuration.
    """

    def load_function_pointer(key, f_key) -> dict:
        return {key: functions[f_key]}

    def load_numpy(key, path) -> dict:
        return {key: np.load(path)}

    # Prefix to function mapping. If a key starts with one of these prefixes, the prefix
    # gets removed and the function will be called with the value.
    prefix_function_map = {
        "<fn>": load_function_pointer,
        "<np>": load_numpy,
    }

    # Attack type to attack constructor mapping. Add new attacks here.
    attack_constructor_map = {
        "mia": mia.Mia,
        "gmia": gmia.DirectGmia,
    }

    attack_object_paths = {}

    # Pickle helper
    def pickle_attack_obj(attack_obj):
        with open(
            f"{attack_obj_save_path}/{attack_obj.attack_alias}.pickle", "wb"
        ) as f:
            attack_obj.obj_save_path = attack_obj_save_path
            pickle.dump(attack_obj, f)
            attack_object_paths[
                attack_obj.attack_alias
            ] = f"{attack_obj_save_path}/{attack_obj.attack_alias}.pickle"

    # Parse YAML
    with open(yaml_path) as f_stream:
        logger.info(f"Parse attack runner configuration file: {yaml_path}")
        yaml_data = yaml.load(f_stream, Loader=yaml.SafeLoader)

        os.makedirs(attack_obj_save_path, exist_ok=True)

        for i, yaml_attack_pars in enumerate(yaml_data["attack_pars"]):
            yaml_data_conf = yaml_data["data_conf"][i]
            yaml_attack_pars_rem = yaml_attack_pars.copy()
            attack_type = yaml_attack_pars["attack_type"]
            logger.info(
                f"Attack number: {i+1}\n\n"
                "######################## Attack Run ########################"
            )

            attack_pars = {}
            data_conf = {}

            # Dictionary parsing
            logger.info("Setup configuration.")
            for k, func in prefix_function_map.items():
                for key, value in yaml_attack_pars.items():
                    if key.startswith(k):
                        real_key = key.replace(k, "", 1)
                        attack_pars.update(func(real_key, value))
                        del yaml_attack_pars_rem[key]
                for key, value in yaml_data_conf.items():
                    if key.startswith(k):
                        real_key = key.replace(k, "", 1)
                        data_conf.update(func(real_key, value))

            target_models = []
            for path in yaml_attack_pars["target_model_paths"]:
                target_models.append(tf.keras.models.load_model(path))
            del yaml_attack_pars_rem["target_model_paths"]

            data = np.load(yaml_attack_pars["path_to_dataset_data"])
            labels = np.load(yaml_attack_pars["path_to_dataset_labels"])
            del yaml_attack_pars_rem["path_to_dataset_data"]
            del yaml_attack_pars_rem["path_to_dataset_labels"]

            attack_alias = yaml_attack_pars["attack_alias"]
            del yaml_attack_pars_rem["attack_alias"]

            # Add remaining plaintext parameters
            del yaml_attack_pars_rem["attack_type"]
            attack_pars.update(yaml_attack_pars_rem)

            logger.info(f"Attack alias: {yaml_attack_pars['attack_alias']}")
            logger.debug(f"attack_pars = {attack_pars}")
            logger.debug(f"data_conf = {data_conf}")
            attack = attack_constructor_map[attack_type](
                attack_alias,
                attack_pars,
                data,
                labels,
                data_conf,
                target_models,
            )

            logger.info("Running attack.")
            attack.run()
            logger.debug("Serialize attack object.")
            pickle_attack_obj(attack)

    return attack_object_paths


def create_report(attack_object_paths, save_path, pdf=False):
    """
    Create an attack report for all attacks.

    Parameters
    ----------
    attack_object_paths : dict
        Dictionary with attack alias and the corresponding path to the serialized attack
        object.
    save_path : str
        Path where to save the report files.
    pdf : bool
        If true, the attack runner will build the LaTex report and save an PDF to the
        save path.
    """
    os.makedirs(save_path + "/fig", exist_ok=True)
    sections = []
    for key, attack_obj_path in attack_object_paths.items():
        with open(attack_obj_path, "rb") as f_obj:
            attack = pickle.load(f_obj)

            logger.info(f"Create report subsection: {attack.attack_alias}")
            attack.create_attack_section(save_path)
            sections.append(attack.report_section)

    logger.info("Generate final report.")
    report.report_generator(save_path, sections, pdf=pdf)
