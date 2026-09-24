"""
Cover letter for the IEEE Sensors Letters submission.

Follows the structure of the authors' JESTECH cover letter: salutation, the
submission sentence, one paragraph stating the gap and the numbered outputs with
validation, one paragraph on fit to the journal, one paragraph of declarations,
and the corresponding-author block.

One addition to that template: an explicit disclosure of the companion
Meas. Sci. Technol. paper this Letter builds on. It is the authors' own recent
work and is cited as [1] throughout, so naming it up front is the honest way to
pre-empt a self-overlap query from the editorial office.
"""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "submission", "Cover_Letter_LSENS.docx")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

doc = Document()
for s in doc.sections:
    s.top_margin = s.bottom_margin = Inches(1.0)
    s.left_margin = s.right_margin = Inches(1.1)

st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(10)
st.paragraph_format.line_spacing = 1.15


def para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, after=None,
         italic=False):
    p = doc.add_paragraph()
    p.alignment = align
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


para("Dear Editor-in-Chief,", WD_ALIGN_PARAGRAPH.LEFT, after=10)

para(
    "We are pleased to submit our manuscript entitled “A Closed-Form "
    "Minimum-Window Rule for Sensor-Driven Remaining-Useful-Life Prediction” "
    "for consideration as a Regular Letter in IEEE Sensors Letters."
)

para(
    "Sample-complexity theory for remaining useful life (RUL) prediction "
    "quantifies how many complete run-to-failure trajectories a model needs for "
    "training, but is silent on a question a sensor engineer must answer at "
    "design time: how much history the deployed sensor has to retain. This "
    "Letter supplies the missing deployment-side bound and yields three "
    "practical outputs: (1) a windowed Cramér–Rao floor, showing that "
    "deployment accuracy is limited, independently of the training set, by the "
    "sensor noise level and the width of the trailing buffer feeding the "
    "predictor; (2) a single increasing function, the normalized cumulative "
    "Fisher information of the degradation class, whose increments give the "
    "windowing efficiency and whose inverse gives the minimum buffer in closed "
    "form for exponential, power-law and stretched-exponential degradation "
    "alike, expressed in the unit a designer provisions — a buffer length in "
    "samples; and (3) a saturation theorem identifying the training-set size "
    "past which further failure trajectories stop being the binding constraint, "
    "so that accuracy must instead come from a wider buffer or a quieter sensor."
)

para(
    "Validation on the run-to-failure bearing trajectories of the PRONOSTIA "
    "platform confirms that the measured age-error spread tracks the predicted "
    "floor over two decades, and locates the radius within which the "
    "linearization underlying the bound holds. The finding we expect to interest "
    "this readership most is empirical: the exponent that sets the buffer "
    "requirement proves to be a property of the feature-extraction chain rather "
    "than of the machine, so indicators built from different bands of one and "
    "the same accelerometer impose sharply different storage requirements."
)

para(
    "We believe the work suits IEEE Sensors Letters because its subject is what "
    "a sampled sensor stream can support. The observation model is stated in "
    "sampled form with an explicit sampling rate and measurement-noise scale, "
    "the central result converts a normalized window width into a retained "
    "sample count, and the empirical conclusion is a statement about sensor "
    "signal processing: buffer provisioning cannot be specified without naming "
    "the feature the processing chain computes. We suggest the subject category "
    "Sensor Signal Processing."
)

para(
    "This manuscript is original, has not been published previously, and is not "
    "under consideration for publication elsewhere. All authors have approved "
    "the submission and agree with its content, and we declare no conflict of "
    "interest. We wish to disclose that the Letter builds on our recent paper "
    "“Generalization bounds and sample complexity for remaining useful life "
    "prediction from complete degradation trajectories” (Meas. Sci. Technol., "
    "vol. 37, 226203, 2026), cited as reference [1] throughout. That work is "
    "entirely training-side; the present Letter derives the deployment-side "
    "bound it does not contain, and shows the two to be increments of one "
    "information measure differing only in where the observation interval is "
    "anchored. The PRONOSTIA dataset used here is publicly available, and the "
    "analysis code will be released upon acceptance. ORCID identifiers are "
    "provided for both authors."
)

para("Yours sincerely,", WD_ALIGN_PARAGRAPH.LEFT, after=16)

for line, bold in [("Kim-Anh Nguyen, Ph.D.", True),
                   ("On behalf of the authors", False),
                   ("Faculty of Electrical Engineering", False),
                   ("The University of Danang – University of Science and Technology",
                    False),
                   ("Da Nang 550000, Vietnam", False),
                   ("E-mail: nkanh@dut.udn.vn", False),
                   ("ORCID: 0000-0003-3408-847X", False)]:
    para(line, WD_ALIGN_PARAGRAPH.LEFT, bold=bold, after=0)

doc.save(OUT)
print(f"wrote {OUT}")

# read back and report, since there is no renderer available here
d2 = Document(OUT)
paras = [p.text for p in d2.paragraphs if p.text.strip()]
words = sum(len(p.split()) for p in paras)
print(f"{len(paras)} paragraphs, {words} words")
for p in paras[:3]:
    print("  |", p[:95])
