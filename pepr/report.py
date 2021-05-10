"""Report module for generating attack reports describing the attack results."""

from pylatex.base_classes.containers import Container
from pylatex import Document, Command, Section, Subsection
from pylatex.utils import NoEscape

from itertools import groupby

descriptions = {
    "gmia": (
        "Implementation of the direct gmia from Long, Yunhui and Bindschaedler, "
        "Vincent and Wang, Lei and Bu, Diyue and Wang, Xiaofeng and Tang, Haixu and "
        "Gunter, Carl A and Chen, Kai (2018). Understanding membership inferences on "
        "well generalized learning models. arXiv preprint arXiv: 1802.04889."
    ),
    "mia": (
        "Implementation of the basic membership inference attack by Reza Shokri, Marco "
        "Stronati, Congzheng Song and Vitaly Shmatikov. Membership inference attacks "
        "against machine learning models 2017 IEEE Symposium on Security and Privacy "
        "(SP). IEEE, 2017."
    ),
    "L2ContrastReductionAttack": (
        "Reduces the contrast of the input using a perturbation of the given size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "VirtualAdversarialAttack": (
        "Second-order gradient-based attack on the logits. The attack calculate an "
        "untargeted adversarial perturbation by performing a approximated second order "
        "optimization step on the KL divergence between the unperturbed predictions "
        "and the predictions for the adversarial perturbation. This attack was "
        "originally introduced as the Virtual Adversarial Training method. "
        "Reference: Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae, Shin "
        "Ishii, “Distributional Smoothing with Virtual Adversarial Training”. arXiv "
        "preprint arXiv:1507.00677. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "DDNAttack": (
        "The Decoupled Direction and Norm L2 adversarial attack. "
        "Reference: Jérôme Rony, Luiz G. Hafemann, Luiz S. Oliveira, Ismail Ben Ayed, "
        "Robert Sabourin, Eric Granger, “Decoupling Direction and Norm for Efficient "
        "Gradient-Based L2 Adversarial Attacks and Defenses”. arXiv preprint "
        "arXiv:1811.09600. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2ProjectedGradientDescentAttack": (
        "L2 Projected Gradient Descent. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "LinfProjectedGradientDescentAttack": (
        "Linf Projected Gradient Descent. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2BasicIterativeAttack": (
        "L2 Basic Iterative Method. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "LinfBasicIterativeAttack": (
        "L-infinity Basic Iterative Method. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2FastGradientAttack": (
        "Fast Gradient Method (FGM). "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "LinfFastGradientAttack": (
        "Fast Gradient Sign Method (FGSM). "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2AdditiveGaussianNoiseAttack": (
        "Samples Gaussian noise with a fixed L2 size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2AdditiveUniformNoiseAttack": (
        "Samples uniform noise with a fixed L2 size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2ClippingAwareAdditiveGaussianNoiseAttack": (
        "Samples Gaussian noise with a fixed L2 size after clipping. "
        "Reference: Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware "
        "Normalization and Rescaling”. arXiv preprint arXiv:2007.07677. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2ClippingAwareAdditiveUniformNoiseAttack": (
        "Samples uniform noise with a fixed L2 size after clipping. "
        "Reference: Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware "
        "Normalization and Rescaling”. arXiv preprint arXiv:2007.07677. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "LinfAdditiveUniformNoiseAttack": (
        "Samples uniform noise with a fixed L-infinity size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2RepeatedAdditiveGaussianNoiseAttack": (
        "Repeatedly samples Gaussian noise with a fixed L2 size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2RepeatedAdditiveUniformNoiseAttack": (
        "Repeatedly samples uniform noise with a fixed L2 size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack": (
        "Repeatedly samples Gaussian noise with a fixed L2 size after clipping. "
        "Reference: Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware "
        "Normalization and Rescaling”. arXiv preprint arXiv:2007.07677. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2ClippingAwareRepeatedAdditiveUniformNoiseAttack": (
        "Repeatedly samples uniform noise with a fixed L2 size after clipping. "
        "Reference: Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware "
        "Normalization and Rescaling”. arXiv preprint arXiv:2007.07677. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "LinfRepeatedAdditiveUniformNoiseAttack": (
        "Repeatedly samples uniform noise with a fixed L-infinity size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "InversionAttack": (
        "Creates “negative images” by inverting the pixel values. "
        "Reference: Hossein Hosseini, Baicen Xiao, Mayoore Jaiswal, Radha Poovendran, "
        "“On the Limitation of Convolutional Neural Networks in Recognizing Negative "
        "Images”. arXiv preprint arXiv:1607.02533. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "BinarySearchContrastReductionAttack": (
        "Reduces the contrast of the input using a binary search to find the smallest "
        "adversarial perturbation. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "LinearSearchContrastReductionAttack": (
        "Reduces the contrast of the input using a linear search to find the smallest "
        "adversarial perturbation. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2CarliniWagnerAttack": (
        "Implementation of the Carlini & Wagner L2 Attack. "
        "Reference: Nicholas Carlini, David Wagner, “Towards evaluating the robustness "
        "of neural networks. In 2017 ieee symposium on security and privacy”. arXiv "
        "preprint arXiv:1608.04644. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "NewtonFoolAttack": (
        "Implementation of the NewtonFool Attack. "
        "Reference: Uyeong Jang et al., “Objective Metrics and Gradient Descent "
        "Algorithms for Adversarial Examples in Machine Learning” "
        "https://dl.acm.org/doi/10.1145/3134600.3134635. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "EADAttack": (
        "Implementation of the EAD Attack with EN Decision Rule. "
        "Reference: Pin-Yu Chen, Yash Sharma, Huan Zhang, Jinfeng Yi, Cho-Jui Hsieh, "
        "“EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples”, "
        "https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16893 "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "GaussianBlurAttack": (
        "Blurs the inputs using a Gaussian filter with linearly increasing standard "
        "deviation. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2DeepFoolAttack": (
        "A simple and fast gradient-based adversarial attack. Implements the DeepFool "
        "L2 attack. "
        "Reference: Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard, "
        "“DeepFool: a simple and accurate method to fool deep neural networks” arXiv "
        "preprint arXiv:1511.04599. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "LinfDeepFoolAttack": (
        "A simple and fast gradient-based adversarial attack. Implements the DeepFool "
        "L-Infinity attack. "
        "Reference: Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard, "
        "“DeepFool: a simple and accurate method to fool deep neural networks” arXiv "
        "preprint arXiv:1511.04599. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "SaltAndPepperNoiseAttack": (
        "Increases the amount of salt and pepper noise until the input is "
        "misclassified. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "LinearSearchBlendedUniformNoiseAttack": (
        "Blends the input with a uniform noise input until it is misclassified. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "BinarizationRefinementAttack": (
        "For models that preprocess their inputs by binarizing the inputs, this attack "
        "can improve adversarials found by other attacks. It does this by utilizing "
        "information about the binarization and mapping values to the corresponding "
        "value in the clean input or to the right side of the threshold. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "BoundaryAttack": (
        "A powerful adversarial attack that requires neither gradients nor "
        "probabilities. "
        "Reference: Wieland Brendel (*), Jonas Rauber (*), Matthias Bethge, "
        "“Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box "
        "Machine Learning Models”. arXiv preprint arXiv:1712.04248. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L0BrendelBethgeAttack": (
        "L0 variant of the Brendel & Bethge adversarial attack. This is a "
        "powerful gradient-based adversarial attack that follows the adversarial "
        "boundary (the boundary between the space of adversarial and non-adversarial "
        "images as defined by the adversarial criterion) to find the minimum distance "
        "to the clean image. "
        "Reference: Wieland Brendel, Jonas Rauber, Matthias Kümmerer, Ivan "
        "Ustyuzhaninov, Matthias Bethge, “Accurate, reliable and fast robustness "
        "evaluation”, 33rd Conference on Neural Information Processing Systems (2019). "
        "arXiv preprint arXiv:1907.01003. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L1BrendelBethgeAttack": (
        "L1 variant of the Brendel & Bethge adversarial attack. This is a "
        "powerful gradient-based adversarial attack that follows the adversarial "
        "boundary (the boundary between the space of adversarial and non-adversarial "
        "images as defined by the adversarial criterion) to find the minimum distance "
        "to the clean image. "
        "Reference: Wieland Brendel, Jonas Rauber, Matthias Kümmerer, Ivan "
        "Ustyuzhaninov, Matthias Bethge, “Accurate, reliable and fast robustness "
        "evaluation”, 33rd Conference on Neural Information Processing Systems (2019). "
        "arXiv preprint arXiv:1907.01003. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "L2BrendelBethgeAttack": (
        "L2 variant of the Brendel & Bethge adversarial attack. This is a "
        "powerful gradient-based adversarial attack that follows the adversarial "
        "boundary (the boundary between the space of adversarial and non-adversarial "
        "images as defined by the adversarial criterion) to find the minimum distance "
        "to the clean image. "
        "Reference: Wieland Brendel, Jonas Rauber, Matthias Kümmerer, Ivan "
        "Ustyuzhaninov, Matthias Bethge, “Accurate, reliable and fast robustness "
        "evaluation”, 33rd Conference on Neural Information Processing Systems (2019). "
        "arXiv preprint arXiv:1907.01003. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "LinfinityBrendelBethgeAttack": (
        "L-infinity variant of the Brendel & Bethge adversarial attack. This is a "
        "powerful gradient-based adversarial attack that follows the adversarial "
        "boundary (the boundary between the space of adversarial and non-adversarial "
        "images as defined by the adversarial criterion) to find the minimum distance "
        "to the clean image. "
        "Reference: Wieland Brendel, Jonas Rauber, Matthias Kümmerer, Ivan "
        "Ustyuzhaninov, Matthias Bethge, “Accurate, reliable and fast robustness "
        "evaluation”, 33rd Conference on Neural Information Processing Systems (2019). "
        "arXiv preprint arXiv:1907.01003. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
}


class ReportSection(Container):
    """Attack section for a specific attack instantiation in the attack report."""

    _latex_name = "reportsection"

    def __init__(self, attack_name, attack_alias, attack_type):
        """Initialize an attack section.

        Parameters
        ----------
        attack_name: str
            Name of the attack.
        attack_alias: str
            Alias for the attack instantiation used to structure the different attack
            instantiations in the report. For this Reason should the alias be unique for
            a report.
        attack_type: str
            Type of the attack (e.g. gmia).
        """
        super().__init__()
        self.attack_name = attack_name
        self.attack_alias = attack_alias
        self.append(Subsection(attack_alias))
        self.attack_type = attack_type

    def dumps(self):
        """Dump the content of a report section as raw latex code."""
        s = ""
        for child in self:
            if isinstance(child, str):
                s_ = child
            else:
                s_ = child.dumps()
            s += s_
        return s


def report_generator(save_path, attack_subsections, pdf=False):
    """Create a report out of multiple attack sections.

    Parameters
    ----------
    save_path : str
        Path to save the tex and pdf file of the report.
    attack_subsections : list
        List containing all attack subsections in the pentesting setting.
    pdf : bool
        If set, generate pdf out of latex file.
    """
    doc = Document(documentclass="article")
    doc.preamble.append(Command("usepackage", "graphicx"))
    doc.preamble.append(Command("usepackage", "subcaption"))
    doc.preamble.append(Command("usepackage", "geometry"))
    doc.preamble.append(Command("usepackage", "float"))
    doc.preamble.append(Command("geometry", "textwidth=180mm, textheight=240mm"))
    doc.preamble.append(Command("title", "Privacy Risk and Robustness Report"))
    doc.preamble.append(Command("date", NoEscape(r"\today")))
    doc.append(NoEscape(r"\maketitle"))

    # Group by attack type and preserve order
    order = [elm.attack_type for elm in attack_subsections if elm.attack_type]
    attack_subsections.sort(key=lambda elm: order.index(elm.attack_type))
    grouped_sections = [
        list(arr)
        for k, arr in groupby(attack_subsections, key=lambda elm: elm.attack_type)
    ]

    for i, attack in enumerate(grouped_sections):
        with doc.create(Section(attack[0].attack_name)):
            doc.append(descriptions[attack[0].attack_type])
            for subsection in attack:
                doc.append(subsection)

    if pdf:
        doc.generate_pdf(save_path + "/attack_report", clean_tex=False)
    else:
        doc.generate_tex(save_path + "/attack_report")
