"""Credit Risk Predictor on Hugging Face Spaces.

Loads the same scikit-learn pipeline as the Streamlit app (models/credit_risk_model.joblib),
predicts the probability of default for one applicant, and explains the prediction by walking
each tree of the HistGradientBoostingClassifier (Saabas path attribution). The contributions
plus the baseline add up exactly to the model's raw output.
"""

from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
saved = joblib.load(HERE / "models" / "credit_risk_model.joblib")
PIPELINE = saved["pipeline"]
THRESHOLD = float(saved["threshold"])
DEFAULTS = joblib.load(HERE / "models" / "app_defaults.joblib")
if isinstance(DEFAULTS, pd.Series):
    DEFAULTS = DEFAULTS.to_dict()
elif isinstance(DEFAULTS, pd.DataFrame):
    DEFAULTS = DEFAULTS.iloc[0].to_dict()

PRE, CLF = PIPELINE[0], PIPELINE[-1]
OUT_NAMES = list(PRE.get_feature_names_out())
COLUMNS = list(PIPELINE.feature_names_in_)


def source_column(name: str) -> str:
    """Map a transformed feature (num__X or cat__X_value) back to its input column."""
    body = name.split("__", 1)[1]
    if name.startswith("num__"):
        return body
    matches = [c for c in COLUMNS if body == c or body.startswith(c + "_")]
    return max(matches, key=len) if matches else body


SOURCE = [source_column(n) for n in OUT_NAMES]


def expectations(nodes) -> np.ndarray:
    """Expected raw output at every node: leaf values averaged by training counts."""
    exp = np.zeros(len(nodes))

    def walk(i: int) -> float:
        n = nodes[i]
        if n["is_leaf"]:
            exp[i] = n["value"]
        else:
            l, r = n["left"], n["right"]
            vl, vr = walk(l), walk(r)
            cl, cr = nodes[l]["count"], nodes[r]["count"]
            exp[i] = (cl * vl + cr * vr) / (cl + cr)
        return exp[i]

    walk(0)
    return exp


TREES = [(p[0].nodes, expectations(p[0].nodes)) for p in CLF._predictors]
BASELINE = float(np.ravel(CLF._baseline_prediction)[0])


def explain(x: np.ndarray):
    """Per-feature contributions (log-odds) for one transformed row."""
    contrib = np.zeros(len(x))
    base = BASELINE
    for nodes, exp in TREES:
        base += exp[0]
        i = 0
        while not nodes[i]["is_leaf"]:
            n = nodes[i]
            v = x[n["feature_idx"]]
            go_left = bool(n["missing_go_to_left"]) if np.isnan(v) else v <= n["num_threshold"]
            j = n["left"] if go_left else n["right"]
            contrib[n["feature_idx"]] += exp[j] - exp[i]
            i = j
    return base, contrib


LABELS = {
    "EXT_SOURCE_1": "External credit score",
    "EXT_SOURCE_2": "External credit score",
    "EXT_SOURCE_3": "External credit score",
    "DAYS_BIRTH": "Age",
    "AMT_INCOME_TOTAL": "Yearly income",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Yearly repayment",
    "AMT_GOODS_PRICE": "Price of goods",
    "CODE_GENDER": "Gender",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_EDUCATION_TYPE": "Education",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns property",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_HOUSING_TYPE": "Housing",
}


def predict(gender, age, family, education, income, credit, annuity, goods, car, realty, income_type, housing, know_ext, ext):
    if annuity > income:
        return "Yearly repayment can't be more than yearly income.", "", None
    row = dict(DEFAULTS)
    row.update(
        CODE_GENDER=gender, DAYS_BIRTH=-(age * 365), NAME_FAMILY_STATUS=family, NAME_EDUCATION_TYPE=education,
        AMT_INCOME_TOTAL=income, AMT_CREDIT=credit, AMT_ANNUITY=annuity, AMT_GOODS_PRICE=goods,
        FLAG_OWN_CAR=car, FLAG_OWN_REALTY=realty, NAME_INCOME_TYPE=income_type, NAME_HOUSING_TYPE=housing,
    )
    if know_ext:
        row.update(EXT_SOURCE_1=ext, EXT_SOURCE_2=ext, EXT_SOURCE_3=ext)
    df = pd.DataFrame([row])[COLUMNS]
    p = float(PIPELINE.predict_proba(df)[0, 1])

    x = np.asarray(PRE.transform(df), dtype=float)[0]
    base, contrib = explain(x)
    by_input: dict[str, float] = {}
    for c, src in zip(contrib, SOURCE):
        key = LABELS.get(src, "Everything else (typical values)")
        if src.startswith("EXT_SOURCE") and not know_ext:
            key = "Everything else (typical values)"
        by_input[key] = by_input.get(key, 0.0) + float(c)
    raw = base + sum(by_input.values())
    check = abs(1 / (1 + np.exp(-raw)) - p)

    flagged = p >= THRESHOLD
    verdict = (
        f"## {p:.1%} chance of default\n"
        f"**{'Flag for review' if flagged else 'Below the review line'}**: the model flags applicants at "
        f"{THRESHOLD:.0%} or more, a line chosen to catch more real defaulters than the default 50% would."
    )
    rows = sorted(by_input.items(), key=lambda kv: -abs(kv[1]))
    table = pd.DataFrame(
        [{"Input": k, "Effect on risk": ("raises" if v > 0 else "lowers"), "Log-odds": round(v, 3)} for k, v in rows if abs(v) >= 0.001]
    )
    note = (
        f"Starting point (average applicant): {base:+.3f} log-odds. The contributions add up to the model's raw output "
        f"{raw:+.3f}, which is {1 / (1 + np.exp(-raw)):.2%} (difference from predict_proba: {check:.1e})."
    )
    return verdict, note, table


MODEL_CARD = """
**Model**: scikit-learn `HistGradientBoostingClassifier` in a pipeline with its preprocessing, trained on the
Home Credit Default Risk application data.

| Metric (test set) | Score |
|---|---:|
| ROC-AUC | 0.760 |
| Recall at 15% threshold | 41.8% |
| Precision at 15% threshold | 25.0% |
| Accuracy | 85.2% |

**Why 15%?** Defaults are rare, so 85% accuracy is easy and says little. Missing a defaulter costs more than a second
look at a safe applicant, so the line sits at 15% instead of 50%: it catches 41.8% of real defaulters, and about one in
four flagged applicants really defaults.

**Limits**: a learning project, not a lending tool. Inputs you don't set here are fixed at typical values from the
training data. The training data includes gender, which a real lender would have to check for fair-lending reasons.
"""

with gr.Blocks(title="Credit Risk Predictor") as demo:
    gr.Markdown(
        "# Credit Risk Predictor\n"
        "How likely is this applicant to default, and which inputs pushed the estimate up or down? "
        "By [Sangam Gupta](https://www.sangamgupta.in/work/credit-risk) · "
        "[Code on GitHub](https://github.com/TheSangamX/credit_risk_predictor)"
    )
    with gr.Row():
        with gr.Column():
            gender = gr.Radio(["F", "M"], value="F", label="Gender")
            age = gr.Slider(18, 70, value=35, step=1, label="Age")
            family = gr.Dropdown(["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], value="Married", label="Family status")
            education = gr.Dropdown(["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"], value="Secondary / secondary special", label="Education")
        with gr.Column():
            income = gr.Number(180000, label="Yearly income")
            credit = gr.Number(500000, label="Loan amount")
            annuity = gr.Number(25000, label="Yearly repayment")
            goods = gr.Number(450000, label="Price of the goods the loan is for")
        with gr.Column():
            car = gr.Radio(["Y", "N"], value="N", label="Owns a car")
            realty = gr.Radio(["Y", "N"], value="Y", label="Owns property")
            income_type = gr.Dropdown(["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"], value="Working", label="Income type")
            housing = gr.Dropdown(["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"], value="House / apartment", label="Housing")
    with gr.Row():
        know_ext = gr.Checkbox(False, label="I know the applicant's external credit score")
        ext = gr.Slider(0, 1, value=0.5, step=0.01, label="External credit score (0 = worst, 1 = best)")
    go = gr.Button("Estimate risk", variant="primary")
    verdict = gr.Markdown()
    table = gr.Dataframe(label="What moved the estimate (largest first)", interactive=False)
    note = gr.Markdown()
    with gr.Accordion("Model card", open=False):
        gr.Markdown(MODEL_CARD)
    inputs = [gender, age, family, education, income, credit, annuity, goods, car, realty, income_type, housing, know_ext, ext]
    go.click(predict, inputs, [verdict, note, table])
    demo.load(predict, inputs, [verdict, note, table])

if __name__ == "__main__":
    demo.launch()
