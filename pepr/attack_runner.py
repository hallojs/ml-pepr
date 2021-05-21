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
from pepr.robustness import foolbox_wrapper, art_wrapper

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
        "FB_L2ContrastReductionAttack": foolbox_wrapper.L2ContrastReductionAttack,
        "FB_VirtualAdversarialAttack": foolbox_wrapper.VirtualAdversarialAttack,
        "FB_DDNAttack": foolbox_wrapper.DDNAttack,
        "FB_L2ProjectedGradientDescentAttack": foolbox_wrapper.L2ProjectedGradientDescentAttack,
        "FB_LinfProjectedGradientDescentAttack": foolbox_wrapper.LinfProjectedGradientDescentAttack,
        "FB_L2BasicIterativeAttack": foolbox_wrapper.L2BasicIterativeAttack,
        "FB_LinfBasicIterativeAttack": foolbox_wrapper.LinfBasicIterativeAttack,
        "FB_L2FastGradientAttack": foolbox_wrapper.L2FastGradientAttack,
        "FB_LinfFastGradientAttack": foolbox_wrapper.LinfFastGradientAttack,
        "FB_L2AdditiveGaussianNoiseAttack": foolbox_wrapper.L2AdditiveGaussianNoiseAttack,
        "FB_L2AdditiveUniformNoiseAttack": foolbox_wrapper.L2AdditiveUniformNoiseAttack,
        "FB_L2ClippingAwareAdditiveGaussianNoiseAttack": foolbox_wrapper.L2ClippingAwareAdditiveGaussianNoiseAttack,
        "FB_L2ClippingAwareAdditiveUniformNoiseAttack": foolbox_wrapper.L2ClippingAwareAdditiveUniformNoiseAttack,
        "FB_L2RepeatedAdditiveGaussianNoiseAttack": foolbox_wrapper.L2RepeatedAdditiveGaussianNoiseAttack,
        "FB_L2RepeatedAdditiveUniformNoiseAttack": foolbox_wrapper.L2RepeatedAdditiveUniformNoiseAttack,
        "FB_L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack": foolbox_wrapper.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack,
        "FB_L2ClippingAwareRepeatedAdditiveUniformNoiseAttack": foolbox_wrapper.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack,
        "FB_LinfAdditiveUniformNoiseAttack": foolbox_wrapper.LinfAdditiveUniformNoiseAttack,
        "FB_LinfRepeatedAdditiveUniformNoiseAttack": foolbox_wrapper.LinfRepeatedAdditiveUniformNoiseAttack,
        "FB_InversionAttack": foolbox_wrapper.InversionAttack,
        "FB_BinarySearchContrastReductionAttack": foolbox_wrapper.BinarySearchContrastReductionAttack,
        "FB_LinearSearchContrastReductionAttack": foolbox_wrapper.LinearSearchContrastReductionAttack,
        "FB_L2CarliniWagnerAttack": foolbox_wrapper.L2CarliniWagnerAttack,
        "FB_NewtonFoolAttack": foolbox_wrapper.NewtonFoolAttack,
        "FB_EADAttack": foolbox_wrapper.EADAttack,
        "FB_GaussianBlurAttack": foolbox_wrapper.GaussianBlurAttack,
        "FB_L2DeepFoolAttack": foolbox_wrapper.L2DeepFoolAttack,
        "FB_LinfDeepFoolAttack": foolbox_wrapper.LinfDeepFoolAttack,
        "FB_SaltAndPepperNoiseAttack": foolbox_wrapper.SaltAndPepperNoiseAttack,
        "FB_LinearSearchBlendedUniformNoiseAttack": foolbox_wrapper.LinearSearchBlendedUniformNoiseAttack,
        "FB_BinarizationRefinementAttack": foolbox_wrapper.BinarizationRefinementAttack,
        "FB_BoundaryAttack": foolbox_wrapper.BoundaryAttack,
        "FB_L0BrendelBethgeAttack": foolbox_wrapper.L0BrendelBethgeAttack,
        "FB_L1BrendelBethgeAttack": foolbox_wrapper.L1BrendelBethgeAttack,
        "FB_L2BrendelBethgeAttack": foolbox_wrapper.L2BrendelBethgeAttack,
        "FB_LinfinityBrendelBethgeAttack": foolbox_wrapper.LinfinityBrendelBethgeAttack,
        "ART_FastGradientMethod": art_wrapper.FastGradientMethod,
        "ART_AutoAttack": art_wrapper.AutoAttack,
        "ART_AutoProjectedGradientDescent": art_wrapper.AutoProjectedGradientDescent,
        "ART_BoundaryAttack": art_wrapper.BoundaryAttack,
        "ART_BrendelBethgeAttack": art_wrapper.BrendelBethgeAttack,
        "ART_CarliniL2Method": art_wrapper.CarliniL2Method,
        "ART_CarliniLInfMethod": art_wrapper.CarliniLInfMethod,
        "ART_DeepFool": art_wrapper.DeepFool,
        "ART_ElasticNet": art_wrapper.ElasticNet,
        "ART_FeatureAdversaries": art_wrapper.FeatureAdversaries,
        "ART_FrameSaliencyAttack": art_wrapper.FrameSaliencyAttack,
        "ART_HopSkipJump": art_wrapper.HopSkipJump,
        "ART_BasicIterativeMethod": art_wrapper.BasicIterativeMethod,
        "ART_ProjectedGradientDescent": art_wrapper.ProjectedGradientDescent,
        "ART_NewtonFool": art_wrapper.NewtonFool,
        "ART_PixelAttack": art_wrapper.PixelAttack,
        "ART_ThresholdAttack": art_wrapper.ThresholdAttack,
        "ART_SaliencyMapMethod": art_wrapper.SaliencyMapMethod,
        "ART_SimBA": art_wrapper.SimBA,
        "ART_SpatialTransformation": art_wrapper.SpatialTransformation,
        "ART_SquareAttack": art_wrapper.SquareAttack,
        "ART_TargetedUniversalPerturbation": art_wrapper.TargetedUniversalPerturbation,
        "ART_UniversalPerturbation": art_wrapper.UniversalPerturbation,
        "ART_VirtualAdversarialMethod": art_wrapper.VirtualAdversarialMethod,
        "ART_ZooAttack": art_wrapper.ZooAttack,
        "ART_AdversarialPatch": art_wrapper.AdversarialPatch,
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

            # Handle run arguments
            kwargs = {}
            if "run_args" in yaml_attack_pars_rem:
                kwargs = yaml_attack_pars_rem["run_args"][0]

            # Add remaining plaintext parameters
            del yaml_attack_pars_rem["attack_type"]
            attack_pars.update(yaml_attack_pars_rem)

            logger.info(f"Attack alias: {yaml_attack_pars['attack_alias']}")
            logger.debug(f"attack_pars = {attack_pars}")
            logger.debug(f"data_conf = {data_conf}")
            try:
                attack = attack_constructor_map[attack_type](
                    attack_alias,
                    attack_pars,
                    data,
                    labels,
                    data_conf,
                    target_models,
                )

                logger.info("Running attack.")
                attack.run(**kwargs)
                logger.debug("Serialize attack object.")
                pickle_attack_obj(attack)
            except Exception as e:
                logger.error(e)
                # traceback.print_exception(type(e), e, e.__traceback__)

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
        logger.debug(f"Try loading {key}")
        with open(attack_obj_path, "rb") as f_obj:
            attack = pickle.load(f_obj)

            logger.info(f"Create report subsection: {attack.attack_alias}")
            attack.create_attack_section(save_path)
            sections.append(attack.report_section)

    logger.info("Generate final report.")
    report.report_generator(save_path, sections, pdf=pdf)
