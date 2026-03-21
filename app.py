import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('X_train_sample.pkl', 'rb') as f:
        X_train = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, feature_names, X_train, explainer

model, feature_names, X_train, explainer = load_artifacts()

st.title("Student Pass/Fail Predictor")
st.caption("Adjust the sliders to see how each factor affects the prediction in real time.")

st.header("Student profile")
col1, col2 = st.columns(2)

with col1:
    G1 = st.slider("First term grade (G1)", 0, 20, 10,
        help="Grade 0–20. Passing threshold is 10.")
    G2 = st.slider("Second term grade (G2)", 0, 20, 10,
        help="Strongest predictor in this model.")
    failures = st.slider("Past class failures", 0, 3, 0,
        help="Number of years previously failed.")
    absences = st.slider("Number of absences", 0, 75, 5,
        help="Risk increases significantly above 15 days.")

with col2:
    studytime = st.slider("Weekly study time", 1, 4, 2,
        help="1=<2hrs  2=2-5hrs  3=5-10hrs  4=>10hrs")
    Medu = st.slider("Mother's education", 0, 4, 2,
        help="0=none  1=primary  2=middle  3=secondary  4=higher")
    goout = st.slider("Going out with friends", 1, 5, 3,
        help="1=very low  5=very high")
    health = st.slider("Health status", 1, 5, 3,
        help="1=very bad  5=very good")

st.divider()

def build_input_vector():
    row = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )
    row['G1']        = G1
    row['G2']        = G2
    row['failures']  = failures
    row['absences']  = absences
    row['studytime'] = studytime
    row['Medu']      = Medu
    row['goout']     = goout
    row['health']    = health
    row['age']       = 17
    row['Fedu']      = 2
    row['traveltime']= 1
    row['famrel']    = 3
    row['freetime']  = 3
    row['Dalc']      = 1
    row['Walc']      = 1
    return row

input_df = build_input_vector()

proba = model.predict_proba(input_df)[0][1]
prediction = "PASS" if proba >= 0.5 else "FAIL"

col_pred, col_prob = st.columns(2)
with col_pred:
    st.metric(
        label="Prediction",
        value=prediction,
        delta="above threshold" if proba >= 0.5 else "below threshold"
    )
with col_prob:
    st.metric(
        label="Pass probability",
        value=f"{proba:.1%}",
        delta=f"{proba - 0.67:+.1%} vs average student"
    )

st.progress(float(proba))

if proba >= 0.8:
    st.success("Strong pass prediction.")
elif proba >= 0.5:
    st.info("Borderline pass — some risk factors present.")
elif proba >= 0.3:
    st.warning("At risk — consider intervention.")
else:
    st.error("High failure risk — immediate support recommended.")

st.divider()
st.header("Why this prediction?")

shap_vals = explainer.shap_values(input_df)
if isinstance(shap_vals, list):
    sv = shap_vals[1][0]
else:
    sv = shap_vals[:, :, 1][0]

top_features = ['G2', 'G1', 'failures', 'absences',
                'studytime', 'Medu', 'goout', 'health']
top_indices  = [feature_names.index(f) for f in top_features]
top_shap     = [sv[i] for i in top_indices]
colors       = ['#2ecc71' if v > 0 else '#e74c3c' for v in top_shap]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(top_features, top_shap, color=colors)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('SHAP value  (green = toward Pass,  red = toward Fail)')
ax.set_title('What is driving this prediction?')

for bar, val in zip(bars, top_shap):
    ax.text(
        val + (0.003 if val >= 0 else -0.003),
        bar.get_y() + bar.get_height() / 2,
        f'{val:+.3f}',
        va='center',
        ha='left' if val >= 0 else 'right',
        fontsize=9
    )

plt.tight_layout()
st.pyplot(fig)
plt.close()

dominant_idx = int(np.argmax(np.abs(top_shap)))
dominant     = top_features[dominant_idx]
dominant_val = top_shap[dominant_idx]

if dominant_val > 0:
    st.info(f"Biggest positive factor: **{dominant}** "
            f"(pushing toward Pass by {dominant_val:+.3f})")
else:
    st.warning(f"Biggest risk factor: **{dominant}** "
               f"(pushing toward Fail by {dominant_val:.3f})")

st.divider()
st.header("Simulate an intervention")
st.write("What if this student changed one thing?")

intervention = st.selectbox("Choose an intervention:", [
    "No change",
    "Study 1 level more (+1 studytime)",
    "Reduce absences by 5 days",
    "Improve G2 by 2 points (tutoring)",
    "Reduce going out by 1 level",
])

intervened_df = input_df.copy()

if intervention == "Study 1 level more (+1 studytime)":
    intervened_df['studytime'] = min(studytime + 1, 4)
elif intervention == "Reduce absences by 5 days":
    intervened_df['absences'] = max(absences - 5, 0)
elif intervention == "Improve G2 by 2 points (tutoring)":
    intervened_df['G2'] = min(G2 + 2, 20)
elif intervention == "Reduce going out by 1 level":
    intervened_df['goout'] = max(goout - 1, 1)

if intervention != "No change":
    new_proba = model.predict_proba(intervened_df)[0][1]
    delta     = new_proba - proba

    col_b, col_a = st.columns(2)
    with col_b:
        st.metric("Before", f"{proba:.1%}")
    with col_a:
        st.metric("After intervention", f"{new_proba:.1%}",
                  delta=f"{delta:+.1%}")

    if delta >= 0.1:
        st.success(f"This intervention meaningfully helps — "
                   f"pass probability rises by {delta:.1%}.")
    elif delta > 0:
        st.info(f"Small positive effect — {delta:.1%} improvement.")
    elif delta == 0:
        st.warning("No effect for this student profile.")
    else:
        st.warning("This change slightly reduces the probability.")

st.divider()
st.caption("Model: Random Forest (max_depth=10) | "
           "Dataset: UCI Student Performance | "
           "Features: 41 | Training accuracy: ~93%")