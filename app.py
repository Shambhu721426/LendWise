
import os
import json
import joblib
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="LendWise | Loan Prediction",
    page_icon="🏦",
    layout="wide"
)

MODEL_PATH = "models/loan_model.joblib"
METRICS_PATH = "models/metrics.json"


# ==========================================
# CUSTOM CSS - DARK MODERN UI
# ==========================================

st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #0E1117 !important;
    color: #FFFFFF !important;
}

.block-container {
    max-width: 1200px;
    padding: 2rem 2rem 3rem 2rem !important;
}

/* HEADER */
.main-title {
    font-size: 38px !important;
    font-weight: 900 !important;
    color: #FFFFFF !important;
    margin-bottom: 5px;
}

.sub-title {
    color: #D1D5DB !important;
    font-size: 17px !important;
}

/* SECTION HEADINGS */
.section-title {
    color: #FFFFFF !important;
    font-size: 23px !important;
    font-weight: 800 !important;
    margin-top: 15px;
    margin-bottom: 15px;
}

h1, h2, h3 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

/* NORMAL TEXT */
p, li {
    color: #E5E7EB;
}

/* DASHBOARD CARDS */
.info-card {
    background: #191D27 !important;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #393D47;
    text-align: center;
    min-height: 110px;
}

.info-label {
    color: #A7B0C0 !important;
    font-size: 15px !important;
}

.info-value {
    color: #FFFFFF !important;
    font-size: 21px !important;
    font-weight: 800 !important;
}

/* INPUT LABELS */
.stSelectbox label,
.stNumberInput label {
    color: #FFFFFF !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* SELECT BOXES */
div[data-baseweb="select"] > div {
    background-color: #262932 !important;
    border: 1px solid #393D47 !important;
    border-radius: 9px !important;
    min-height: 43px !important;
}

div[data-baseweb="select"] * {
    color: #FFFFFF !important;
    font-size: 16px !important;
}

/* NUMBER INPUTS */
div[data-testid="stNumberInput"] input {
    background-color: #262932 !important;
    color: #FFFFFF !important;
    border: 1px solid #393D47 !important;
    border-radius: 9px !important;
    font-size: 16px !important;
    min-height: 43px !important;
}

div[data-testid="stNumberInput"] button {
    background-color: #262932 !important;
    color: #FFFFFF !important;
    border: none !important;
}

/* DROPDOWN MENU */
div[role="listbox"],
div[role="option"] {
    background-color: #262932 !important;
    color: #FFFFFF !important;
}

div[role="option"]:hover {
    background-color: #394150 !important;
}

/* INPUT FOCUS */
div[data-baseweb="select"]:focus-within > div,
div[data-testid="stNumberInput"] input:focus {
    border: 1px solid #60A5FA !important;
    box-shadow: 0 0 0 1px #60A5FA !important;
}

/* PREDICTION BUTTON */
div.stButton > button {
    background-color: #171B24 !important;
    color: #FFFFFF !important;
    border: 1px solid #555B68 !important;
    border-radius: 9px !important;
    padding: 10px 22px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    min-height: 43px !important;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #303744 !important;
    color: #FFFFFF !important;
    border: 1px solid #FFFFFF !important;
}

/* SUCCESS MESSAGE */
div[data-testid="stAlert"][kind="success"] {
    background-color: #123C2B !important;
    border: 1px solid #1E6B45 !important;
    border-radius: 9px !important;
}

div[data-testid="stAlert"][kind="success"] p {
    color: #4ADE80 !important;
    font-size: 16px !important;
}

/* ERROR MESSAGE */
div[data-testid="stAlert"][kind="error"] {
    background-color: #431D25 !important;
    border: 1px solid #8B303B !important;
    border-radius: 9px !important;
}

div[data-testid="stAlert"][kind="error"] p {
    color: #F87171 !important;
    font-size: 16px !important;
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background-color: #191D27 !important;
    border: 1px solid #393D47 !important;
    border-radius: 10px !important;
    padding: 15px !important;
}

div[data-testid="stMetricLabel"] {
    color: #D1D5DB !important;
    font-size: 15px !important;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 27px !important;
    font-weight: 800 !important;
}

/* CREDIT RISK */
.risk-low {
    color: #4ADE80 !important;
    font-size: 18px;
    font-weight: 800;
    padding: 8px 0;
}

.risk-high {
    color: #F87171 !important;
    font-size: 18px;
    font-weight: 800;
    padding: 8px 0;
}

/* CIRCULAR PREDICTION RESULT */
.result-heading {
    text-align: center;
    color: #FFFFFF !important;
    font-size: 25px !important;
    font-weight: 800 !important;
    margin-top: 20px;
}

.result-circle {
    width: 220px;
    height: 220px;
    border-radius: 50%;
    margin: 25px auto;
    display: flex;
    align-items: center;
    justify-content: center;
}

.result-inner {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: #171B2E;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.result-percentage {
    font-size: 38px !important;
    font-weight: 900 !important;
    color: #FFFFFF !important;
}

.result-label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #D1D5DB !important;
}

.result-status {
    text-align: center;
    font-size: 27px;
    font-weight: 900;
    margin-top: 15px;
}

.result-description {
    text-align: center;
    color: #D1D5DB !important;
    font-size: 16px;
}

/* DIVIDERS */
hr {
    border-color: #30343D !important;
}

/* CAPTIONS */
.stCaption {
    color: #A7B0C0 !important;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #9CA3AF !important;
    font-size: 14px;
    padding: 20px;
}


/* ===== CIRCULAR PREDICTION RESULT ===== */

.result-circle {
    width: 220px;
    height: 220px;
    border-radius: 50%;
    margin: 25px auto;
    display: flex;
    align-items: center;
    justify-content: center;
}

.result-inner {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: #171B2E;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.result-percentage {
    font-size: 38px !important;
    font-weight: 900 !important;
    color: #FFFFFF !important;
}

.result-label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #D1D5DB !important;
}

.result-status {
    text-align: center;
    font-size: 27px;
    font-weight: 900;
    margin-top: 15px;
}

.result-description {
    text-align: center;
    color: #D1D5DB !important;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL AND METRICS
# ==========================================

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as file:
            return json.load(file)
    return None


model = load_model()
metrics = load_metrics()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<p class="main-title">🏦 LendWise</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Smart Loan Approval Prediction System</p>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# MODEL AVAILABILITY
# ==========================================

if model is None:
    st.error("Trained model not found!")

    st.info(
        "Please add train.csv inside the data folder "
        "and run python train.py in your terminal."
    )

    st.stop()


# ==========================================
# DASHBOARD - THREE CARDS
# ==========================================

st.markdown(
    '<p class="section-title">Loan Prediction Dashboard</p>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">SECTION 01</div>
        <div class="info-value">👤 Applicant Details</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">SECTION 02</div>
        <div class="info-value">💰 Loan Details</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">SECTION 03</div>
        <div class="info-value">📊 Credit History</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# ==========================================
# APPLICANT DETAILS
# ==========================================

st.markdown(
    '<p class="section-title">01. Applicant Details</p>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Marital Status",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Number of Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

with col2:
    employment = st.selectbox(
        "Employment Type",
        ["Salaried", "Self Employed"]
    )

    applicant_income = st.number_input(
        "Applicant Income (₹)",
        min_value=0,
        value=50000,
        step=1000
    )

    coapplicant_income = st.number_input(
        "Co-applicant Income (₹)",
        min_value=0,
        value=0,
        step=1000
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )


st.divider()


# ==========================================
# LOAN DETAILS
# ==========================================

st.markdown(
    '<p class="section-title">02. Loan Details</p>',
    unsafe_allow_html=True
)

loan_intent = st.selectbox(
    "Loan Intent",
    [
        "Home Purchase",
        "Home Renovation",
        "Education",
        "Medical",
        "Vehicle Purchase",
        "Personal Use",
        "Business"
    ]
)

col1, col2 = st.columns(2)

with col1:
    loan_amount = st.number_input(
        "Loan Amount (₹ thousands)",
        min_value=1,
        value=150,
        step=5
    )

with col2:
    loan_term = st.number_input(
        "Loan Term (Days)",
        min_value=1,
        value=360,
        step=30
    )


st.divider()


# ==========================================
# CREDIT HISTORY AND RISK
# ==========================================

st.markdown(
    '<p class="section-title">03. Credit History & Risk Assessment</p>',
    unsafe_allow_html=True
)

st.info(
    "The training dataset contains Credit_History as a 0/1 indicator, "
    "not an actual CIBIL score."
)

credit_history = st.selectbox(
    "Credit History",
    options=[1.0, 0.0],
    format_func=lambda x: (
        "Good Credit History (1) - Low Risk"
        if x == 1.0
        else "Poor Credit History (0) - High Risk"
    )
)

if credit_history == 1.0:
    st.markdown(
        '<p class="risk-low">🟢 LOW RISK - Good Credit History</p>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<p class="risk-high">🔴 HIGH RISK - Poor Credit History</p>',
        unsafe_allow_html=True
    )


st.divider()


# ==========================================
# 04. LOAN PREDICTION
# ==========================================

st.markdown(
    '<p class="section-title">04. Loan Prediction</p>',
    unsafe_allow_html=True
)

if st.button("PREDICT LOAN APPROVAL"):

    # Prepare input data
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": (
            "Yes" if employment == "Self Employed" else "No"
        ),
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    # Get prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    percentage = probability * 100
    angle = percentage * 3.6

    if prediction == 1:
        status = "LOAN APPROVED"
        status_color = "#22C55E"
        circle_color = "#22C55E"
    else:
        status = "LOAN NOT APPROVED"
        status_color = "#EF4444"
        circle_color = "#EF4444"

    # Display circular result using HTML component
    prediction_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            background: transparent;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 10px;
            color: white;
        }}

        .result-container {{
            text-align: center;
            padding: 15px;
        }}

        .result-heading {{
            font-size: 25px;
            font-weight: bold;
            color: white;
            margin-bottom: 25px;
        }}

        .result-circle {{
            width: 220px;
            height: 220px;
            border-radius: 50%;
            margin: auto;
            display: flex;
            justify-content: center;
            align-items: center;
            background: conic-gradient(
                {circle_color} 0deg {angle}deg,
                #343A46 {angle}deg 360deg
            );
        }}

        .result-inner {{
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: #171B2E;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}

        .percentage {{
            color: white;
            font-size: 38px;
            font-weight: 900;
        }}

        .result-label {{
            color: #D1D5DB;
            font-size: 15px;
            font-weight: bold;
            margin-top: 8px;
        }}

        .result-status {{
            color: {status_color};
            font-size: 27px;
            font-weight: 900;
            margin-top: 22px;
        }}

        .result-description {{
            color: #D1D5DB;
            font-size: 15px;
            margin-top: 12px;
        }}
    </style>
    </head>

    <body>
        <div class="result-container">

            <div class="result-heading">
                Your Prediction Result
            </div>

            <div class="result-circle">
                <div class="result-inner">

                    <div class="percentage">
                        {percentage:.2f}%
                    </div>

                    <div class="result-label">
                        Approval Probability
                    </div>

                </div>
            </div>

            <div class="result-status">
                {status}
            </div>

            <div class="result-description">
                AI-based loan prediction result
            </div>

        </div>
    </body>
    </html>
    """

    components.html(
        prediction_html,
        height=430,
        scrolling=False
    )

    st.caption(
        "Educational machine learning prediction only. "
        "This is not a guaranteed bank approval decision."
    )


# ==========================================
# MODEL PERFORMANCE
# ==========================================

if metrics is not None:

    st.divider()

    st.markdown(
        '<p class="section-title">05. Model Performance</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{metrics['f1_score'] * 100:.2f}%"
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🏦 LendWise | Loan Approval Prediction System<br>
        Developed using Python, Pandas, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
