from pylatex.base_classes.containers import Container
from pylatex import Document, Command, Section, Subsection
from pylatex.utils import NoEscape

descriptions = {
    'gmia': (
        "Implementation of the direct gmia from Long, Yunhui and Bindschaedler, "
        "Vincent and Wang, Lei and Bu, Diyue and Wang, Xiaofeng and Tang, Haixu and "
        "Gunter, Carl A and Chen, Kai (2018). Understanding membership inferences on "
        "well generalized learning models.arXiv preprint arXiv: 1802.04889."
    )
}


class ReportSection(Container):
    """Attack section for a specific attack instantiation in the attack report."""
    attack_type: str
    attack_name: str
    attack_alias: str
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


def report_generator(attack_sections):
    """Create a report out of multiple attack sections."""
    doc = Document(documentclass='article')
    doc.preamble.append(Command("usepackage", "subcaption"))
    doc.preamble.append(Command("usepackage", "geometry"))
    doc.preamble.append(Command("geometry", "textwidth=180mm, textheight=240mm"))
    doc.preamble.append(Command("title", "Privacy Risk and Robustness Report"))
    doc.preamble.append(Command("date", NoEscape(r"\today")))
    doc.append(NoEscape(r"\maketitle"))

    for section in attack_sections:
        with doc.create(Section(section.attack_name)):
            doc.append(descriptions[section.attack_type])
            doc.append(section)

    doc.generate_pdf("attack_report", clean_tex=False)
