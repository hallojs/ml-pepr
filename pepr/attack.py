"""Attack module structuring how attacks are accessible in pepr."""

from abc import ABC, abstractmethod

import tensorflow as tf


class Attack(ABC):
    """Abstract base class for all attack implementations of pepr."""
    def __init__(
        self, attack_alias, attack_pars, data, labels, data_conf, target_models
    ):
        """Initialize an attack object in the pepr library.

        Parameters
        ----------
        attack_alias : str
            Alias for a specific instantiation of the attack class.
        attack_pars : dict
            Dictionary containing all needed parameters fo the attack.
        data : numpy.ndarray
            Dataset with all training samples used in the given pentesting setting.
        labels : numpy.ndarray
            Array of all labels used in the given pentesting setting.
        data_conf : dict
            Dictionary describing the data configuration of the given pentesting
            setting.
        target_models : iterable
            List of target models which should be tested.
        """
        self.attack_alias = attack_alias
        self.attack_pars = attack_pars
        self.data = data
        self.labels = labels
        self.labels_cat = tf.keras.utils.to_categorical(
            labels, num_classes=attack_pars["number_classes"]
        )
        self.data_conf = data_conf
        self.target_models = target_models
        self.attack_results = {}

    @abstractmethod
    def run(self):
        """Run an attack using a previously instantiated attack object."""
        pass

    @abstractmethod
    def create_attack_report(self, save_path):
        """Create an attack report just for the given attack instantiation.

        Parameters
        ----------
        save_path : str
            Path to save the tex and pdf file of the attack report.
        """
        pass

    @abstractmethod
    def create_attack_section(self, save_path):
        """Create an attack section for the given attack instantiation.

        Parameters
        ----------
        save_path :
            Path to save the report assets like figures.
        """
        pass
