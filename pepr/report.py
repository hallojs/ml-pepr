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
