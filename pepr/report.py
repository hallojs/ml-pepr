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
    "FB_L2ContrastReductionAttack": (
        "Reduces the contrast of the input using a perturbation of the given size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_VirtualAdversarialAttack": (
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
    "FB_DDNAttack": (
        "The Decoupled Direction and Norm L2 adversarial attack. "
        "Reference: Jérôme Rony, Luiz G. Hafemann, Luiz S. Oliveira, Ismail Ben Ayed, "
        "Robert Sabourin, Eric Granger, “Decoupling Direction and Norm for Efficient "
        "Gradient-Based L2 Adversarial Attacks and Defenses”. arXiv preprint "
        "arXiv:1811.09600. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2ProjectedGradientDescentAttack": (
        "L2 Projected Gradient Descent. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_LinfProjectedGradientDescentAttack": (
        "Linf Projected Gradient Descent. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2BasicIterativeAttack": (
        "L2 Basic Iterative Method. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_LinfBasicIterativeAttack": (
        "L-infinity Basic Iterative Method. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2FastGradientAttack": (
        "Fast Gradient Method (FGM). "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_LinfFastGradientAttack": (
        "Fast Gradient Sign Method (FGSM). "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2AdditiveGaussianNoiseAttack": (
        "Samples Gaussian noise with a fixed L2 size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2AdditiveUniformNoiseAttack": (
        "Samples uniform noise with a fixed L2 size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2ClippingAwareAdditiveGaussianNoiseAttack": (
        "Samples Gaussian noise with a fixed L2 size after clipping. "
        "Reference: Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware "
        "Normalization and Rescaling”. arXiv preprint arXiv:2007.07677. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2ClippingAwareAdditiveUniformNoiseAttack": (
        "Samples uniform noise with a fixed L2 size after clipping. "
        "Reference: Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware "
        "Normalization and Rescaling”. arXiv preprint arXiv:2007.07677. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_LinfAdditiveUniformNoiseAttack": (
        "Samples uniform noise with a fixed L-infinity size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2RepeatedAdditiveGaussianNoiseAttack": (
        "Repeatedly samples Gaussian noise with a fixed L2 size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2RepeatedAdditiveUniformNoiseAttack": (
        "Repeatedly samples uniform noise with a fixed L2 size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack": (
        "Repeatedly samples Gaussian noise with a fixed L2 size after clipping. "
        "Reference: Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware "
        "Normalization and Rescaling”. arXiv preprint arXiv:2007.07677. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2ClippingAwareRepeatedAdditiveUniformNoiseAttack": (
        "Repeatedly samples uniform noise with a fixed L2 size after clipping. "
        "Reference: Jonas Rauber, Matthias Bethge “Fast Differentiable Clipping-Aware "
        "Normalization and Rescaling”. arXiv preprint arXiv:2007.07677. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_LinfRepeatedAdditiveUniformNoiseAttack": (
        "Repeatedly samples uniform noise with a fixed L-infinity size. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_InversionAttack": (
        "Creates “negative images” by inverting the pixel values. "
        "Reference: Hossein Hosseini, Baicen Xiao, Mayoore Jaiswal, Radha Poovendran, "
        "“On the Limitation of Convolutional Neural Networks in Recognizing Negative "
        "Images”. arXiv preprint arXiv:1607.02533. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_BinarySearchContrastReductionAttack": (
        "Reduces the contrast of the input using a binary search to find the smallest "
        "adversarial perturbation. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_LinearSearchContrastReductionAttack": (
        "Reduces the contrast of the input using a linear search to find the smallest "
        "adversarial perturbation. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2CarliniWagnerAttack": (
        "Implementation of the Carlini & Wagner L2 Attack. "
        "Reference: Nicholas Carlini, David Wagner, “Towards evaluating the robustness "
        "of neural networks. In 2017 ieee symposium on security and privacy”. arXiv "
        "preprint arXiv:1608.04644. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_NewtonFoolAttack": (
        "Implementation of the NewtonFool Attack. "
        "Reference: Uyeong Jang et al., “Objective Metrics and Gradient Descent "
        "Algorithms for Adversarial Examples in Machine Learning” "
        "https://dl.acm.org/doi/10.1145/3134600.3134635. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_EADAttack": (
        "Implementation of the EAD Attack with EN Decision Rule. "
        "Reference: Pin-Yu Chen, Yash Sharma, Huan Zhang, Jinfeng Yi, Cho-Jui Hsieh, "
        "“EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples”, "
        "https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16893 "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_GaussianBlurAttack": (
        "Blurs the inputs using a Gaussian filter with linearly increasing standard "
        "deviation. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L2DeepFoolAttack": (
        "A simple and fast gradient-based adversarial attack. Implements the DeepFool "
        "L2 attack. "
        "Reference: Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard, "
        "“DeepFool: a simple and accurate method to fool deep neural networks” arXiv "
        "preprint arXiv:1511.04599. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_LinfDeepFoolAttack": (
        "A simple and fast gradient-based adversarial attack. Implements the DeepFool "
        "L-Infinity attack. "
        "Reference: Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard, "
        "“DeepFool: a simple and accurate method to fool deep neural networks” arXiv "
        "preprint arXiv:1511.04599. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_SaltAndPepperNoiseAttack": (
        "Increases the amount of salt and pepper noise until the input is "
        "misclassified. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_LinearSearchBlendedUniformNoiseAttack": (
        "Blends the input with a uniform noise input until it is misclassified. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_BinarizationRefinementAttack": (
        "For models that preprocess their inputs by binarizing the inputs, this attack "
        "can improve adversarials found by other attacks. It does this by utilizing "
        "information about the binarization and mapping values to the corresponding "
        "value in the clean input or to the right side of the threshold. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_BoundaryAttack": (
        "A powerful adversarial attack that requires neither gradients nor "
        "probabilities. "
        "Reference: Wieland Brendel (*), Jonas Rauber (*), Matthias Bethge, "
        "“Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box "
        "Machine Learning Models”. arXiv preprint arXiv:1712.04248. "
        "Attack implementation provided by the Foolbox toolbox."
    ),
    "FB_L0BrendelBethgeAttack": (
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
    "FB_L1BrendelBethgeAttack": (
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
    "FB_L2BrendelBethgeAttack": (
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
    "FB_LinfinityBrendelBethgeAttack": (
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
    "ART_FastGradientMethod": (
        "This attack was originally implemented by Goodfellow et al. (2015) with the "
        "infinity norm (and is known as the “Fast Gradient Sign Method”). This "
        "implementation extends the attack to other norms, and is therefore called the "
        "Fast Gradient Method. "
        "Reference: Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy, “Explaining "
        "and Harnessing Adversarial Examples” (2015). arXiv preprint arXiv:1412.6572. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_AutoAttack": (
        "Implementation of the AutoAttack attack. "
        "Reference: Francesco Croce, Matthias Hein, “Reliable evaluation of "
        "adversarial robustness with an ensemble of diverse parameter-free attacks” "
        "(2020). arXiv preprint arXiv:2003.01690. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_AutoProjectedGradientDescent": (
        "Implementation of the Auto Projected Gradient Descent attack. "
        "Reference: Francesco Croce, Matthias Hein, “Reliable evaluation of "
        "adversarial robustness with an ensemble of diverse parameter-free attacks” "
        "(2020). arXiv preprint arXiv:2003.01690. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_BoundaryAttack": (
        "Implementation of the boundary attack from Brendel et al. (2018). This is a "
        "powerful black-box attack that only requires final class prediction. "
        "Reference: Wieland Brendel, Jonas Rauber, Matthias Bethge, “Decision-Based "
        "Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning "
        "Models” (2018). arXiv preprint arXiv:1712.04248. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_BrendelBethgeAttack": (
        "Base class for the Brendel & Bethge adversarial attack, a powerful "
        "gradient-based adversarial attack that follows the adversarial boundary (the "
        "boundary between the space of adversarial and non-adversarial images "
        "as defined by the adversarial criterion) to find the minimum distance to the "
        "clean image. "
        "Reference: Wieland Brendel, Jonas Rauber, Matthias Kümmerer, Ivan "
        "Ustyuzhaninov, Matthias Bethge, “Accurate, reliable and fast robustness "
        "evaluation”, 33rd Conference on Neural Information Processing Systems (2019). "
        "arXiv preprint arXiv:1907.01003. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_CarliniL2Method": (
        "The L_2 optimized attack of Carlini and Wagner (2016). This attack is among "
        "the most effective and should be used among the primary attacks to evaluate "
        "potential defences. "
        "Reference: Nicholas Carlini, David Wagner, “Towards Evaluating the Robustness "
        "of Neural Networks” (2017). arXiv preprint arXiv:1608.04644. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_CarliniLInfMethod": (
        "This is a modified version of the L_2 optimized attack of Carlini and Wagner "
        "(2016). It controls the L_Inf norm, i.e. the maximum perturbation applied to "
        "each pixel. "
        "Reference: Nicholas Carlini, David Wagner, “Towards Evaluating the Robustness "
        "of Neural Networks” (2017). arXiv preprint arXiv:1608.04644. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_DeepFool": (
        "Implementation of the attack from Moosavi-Dezfooli et al. (2015). "
        "Reference: Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard, "
        "“DeepFool: a simple and accurate method to fool deep neural networks” (2016). "
        "arXiv preprint arXiv:1511.04599. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_ElasticNet": (
        "The elastic net attack of Pin-Yu Chen et al. (2018). "
        "Reference: Pin-Yu Chen, Yash Sharma, Huan Zhang, Jinfeng Yi, Cho-Jui Hsieh, "
        "“EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples” "
        "(2018). arXiv preprint arXiv:1709.04114. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_FeatureAdversaries": (
        "This class represent a Feature Adversaries evasion attack. "
        "Reference: Sara Sabour, Yanshuai Cao, Fartash Faghri, David J. Fleet, "
        "“Adversarial Manipulation of Deep Representations” (2016). arXiv preprint "
        "arXiv:1511.05122. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_FrameSaliencyAttack": (
        "Implementation of the attack framework proposed by Inkawhich et al. (2018). "
        "Prioritizes the frame of a sequential input to be adversarially perturbed "
        "based on the saliency score of each frame. "
        "Reference: Nathan Inkawhich, Matthew Inkawhich, Yiran Chen, Hai Li, "
        "“Adversarial Attacks for Optical Flow-Based Action Recognition Classifiers” "
        "(2018). arXiv preprint arXiv:1811.11875. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_HopSkipJump": (
        "Implementation of the HopSkipJump attack from Jianbo et al. (2019). This is a "
        "powerful black-box attack that only requires final class prediction, and is "
        "an advanced version of the boundary attack. "
        "Reference: Jianbo Chen, Michael I. Jordan, Martin J. Wainwright, "
        "“HopSkipJumpAttack: A Query-Efficient Decision-Based Attack” (2020). arXiv "
        "preprint arXiv:1904.02144. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_BasicIterativeMethod": (
        "The Basic Iterative Method is the iterative version of FGM and FGSM. "
        "Reference: Alexey Kurakin, Ian Goodfellow, Samy Bengio, “Adversarial examples "
        "in the physical world” (2017). arXiv preprint arXiv:1607.02533. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_ProjectedGradientDescent": (
        "The Projected Gradient Descent attack is an iterative method in which, after "
        "each iteration, the perturbation is projected on an lp-ball of specified "
        "radius (in addition to clipping the values of the adversarial sample so that "
        "it lies in the permitted data range). This is the attack proposed by Madry et "
        "al. for adversarial training. "
        "Reference: Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris "
        "Tsipras, Adrian Vladu, “Towards Deep Learning Models Resistant to Adversarial "
        "Attacks” (2019). arXiv preprint arXiv:1706.06083. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_NewtonFool": (
        "Implementation of the attack from Uyeong Jang et al. (2017). "
        "Reference: Uyeong Jang, Xi Wu, and Somesh Jha. 2017. Objective Metrics and "
        "Gradient Descent Algorithms for Adversarial Examples in Machine Learning. In "
        "Proceedings of the 33rd Annual Computer Security Applications Conference "
        "(ACSAC 2017). Association for Computing Machinery, New York, NY, USA, "
        "262–277. DOI:https://doi.org/10.1145/3134600.3134635 "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_PixelAttack": (
        "This attack was originally implemented by Vargas et al. (2019). It is "
        "generalisation of One Pixel Attack originally implemented by Su et al. "
        "(2019). "
        "Reference: J. Su, D. V. Vargas and K. Sakurai, “One Pixel Attack for Fooling "
        "Deep Neural Networks”, in IEEE Transactions on Evolutionary Computation, vol. "
        "23, no. 5, pp. 828-841, Oct. 2019, doi: 10.1109/TEVC.2019.2890858. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_ThresholdAttack": (
        "This attack was originally implemented by Vargas et al. (2019). "
        "Reference: Shashank Kotyan, Danilo Vasconcellos Vargas, “Adversarial "
        "Robustness Assessment: Why both L_0 and L_inf Attacks Are Necessary” (2019). "
        "arXiv preprint arXiv:1906.06026. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_SaliencyMapMethod": (
        "Implementation of the Jacobian-based Saliency Map Attack (Papernot et al. "
        "2016). "
        "Reference: Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, "
        "Z. Berkay Celik, Ananthram Swami, “The Limitations of Deep Learning in "
        "Adversarial Settings” (2016). arXiv preprint arXiv:1511.07528. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_SimBA": (
        "This class implements the black-box attack SimBA. "
        "Reference: Chuan Guo, Jacob R. Gardner, Yurong You, Andrew Gordon Wilson, "
        "Kilian Q. Weinberger, “Simple Black-box Adversarial Attacks” (2019). "
        "arXiv preprint arXiv:1905.07121. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_SpatialTransformation": (
        "Implementation of the spatial transformation attack using translation and "
        "rotation of inputs. The attack conducts black-box queries to the target model "
        "in a grid search over possible translations and rotations to find optimal "
        "attack parameters. "
        "Reference: Logan Engstrom, Brandon Tran, Dimitris Tsipras, Ludwig Schmidt, "
        "Aleksander Madry, “Exploring the Landscape of Spatial Robustness” (2019). "
        "arXiv preprint arXiv:1712.02779. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_SquareAttack": (
        "This class implements the SquareAttack attack. "
        "Reference: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, "
        "Matthias Hein, “Square Attack: a query-efficient black-box adversarial attack "
        "via random search” (2020). arXiv preprint arXiv:1912.00049. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_TargetedUniversalPerturbation": (
        "Implementation of the attack from Hirano and Takemoto (2019). Computes a "
        "fixed perturbation to be applied to all future inputs. To this end, it can "
        "use any adversarial attack method. "
        "Reference: Hokuto Hirano, Kazuhiro Takemoto, “Simple iterative method for "
        "generating targeted universal adversarial perturbations” (2019). arXiv "
        "preprint arXiv:1911.06502. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_UniversalPerturbation": (
        "Implementation of the attack from Moosavi-Dezfooli et al. (2016). Computes a "
        "fixed perturbation to be applied to all future inputs. To this end, it can "
        "use any adversarial attack method. "
        "Reference: Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, Pascal "
        "Frossard, “Universal adversarial perturbations” (2017). arXiv preprint "
        "arXiv:1610.08401. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_VirtualAdversarialMethod": (
        "This attack was originally proposed by Miyato et al. (2016) and was used for "
        "virtual adversarial training. "
        "Reference: Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae, Shin "
        "Ishii, “Distributional Smoothing with Virtual Adversarial Training” (2016). "
        "arXiv preprint arXiv:1507.00677. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_ZooAttack": (
        "The black-box zeroth-order optimization attack from Pin-Yu Chen et al. "
        "(2018). This attack is a variant of the C&W attack which uses ADAM coordinate "
        "descent to perform numerical estimation of gradients. "
        "Reference: Pin-Yu Chen, Huan Zhang, Yash Sharma, Jinfeng Yi, Cho-Jui Hsieh, "
        "“ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural "
        "Networks without Training Substitute Models” (2017). arXiv preprint "
        "arXiv:1708.03999. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
    ),
    "ART_AdversarialPatch": (
        "Implementation of the adversarial patch attack for square and rectangular "
        "images and videos. "
        "Reference: Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin "
        "Gilmer, “Adversarial Patch” (2018). arXiv preprint arXiv:1712.09665. "
        "Attack implementation provided by the Adversarial Robustness Toolbox (ART)."
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
