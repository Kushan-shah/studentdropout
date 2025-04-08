import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ---------------------------
# Set Page Config & Custom CSS for a polished look
# ---------------------------
st.set_page_config(page_title="Student Dropout Prediction", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Load Saved Artifacts for Prediction (using caching for performance)
# ---------------------------
@st.cache_resource
def load_artifacts():
    with open("model/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/onehot_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("model/training_columns.pkl", "rb") as f:
        training_columns = pickle.load(f)
    return model, scaler, encoder, training_columns

model, scaler, encoder, training_columns = load_artifacts()

# Lists for preprocessing (must match training)
numerical_columns = [
    "Previous_qualification_grade",
    "Admission_grade",
    "Curricular_units_1st_sem_credited",
    "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited",
    "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
    "Curricular_units_2nd_sem_without_evaluations",
    "Age_at_enrollment",
    "Unemployment_rate",
    "Inflation_rate",
    "GDP",
    "Grade_Progression",        # Engineered feature
    "Attendance_Consistency"    # Engineered feature
]

categorical_columns = [
    "Marital_status",
    "Application_mode",
    "Course",
    "Previous_qualification",
    "Nacionality",
    "Mothers_qualification",
    "Fathers_qualification",
    "Mothers_occupation",
    "Fathers_occupation",
    "Age_Group"                # Engineered feature
]

# ---------------------------
# App Title & Introduction
# ---------------------------
st.markdown("<h1 style='text-align: center;'>🎓 Student Dropout Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Enter the student's details below for prediction</h4>", unsafe_allow_html=True)

# ---------------------------
# Sidebar: Grouped User Input with Expanders
# ---------------------------
st.sidebar.header("Input Student Details")

def extract_numeric(s):
    s = s.strip()
    if " – " in s:
        return int(s.split(" – ")[0])
    elif " - " in s:
        return int(s.split(" - ")[0])
    else:
        return int(s)

with st.sidebar.expander("Demographics & Application Details", expanded=True):
    marital_status = st.selectbox(
        "Marital Status",
        ["1 – Single", "2 – Married", "3 – Widower", "4 – Divorced", "5 – Facto Union", "6 – Legally Separated"]
    )
    application_mode = st.selectbox(
        "Application Mode",
        [
            "1 - 1st phase - general contingent",
            "2 - Ordinance No. 612/93",
            "5 - 1st phase - special contingent (Azores Island)",
            "7 - Holders of other higher courses",
            "10 - Ordinance No. 854-B/99",
            "15 - International student (bachelor)",
            "16 - 1st phase - special contingent (Madeira Island)",
            "17 - 2nd phase - general contingent",
            "18 - 3rd phase - general contingent",
            "26 - Ordinance No. 533-A/99, item b2) (Different Plan)",
            "27 - Ordinance No. 533-A/99, item b3 (Other Institution)",
            "39 - Over 23 years old",
            "42 - Transfer",
            "43 - Change of course",
            "44 - Technological specialization diploma holders",
            "51 - Change of institution/course",
            "53 - Short cycle diploma holders",
            "57 - Change of institution/course (International)"
        ]
    )
    application_order = st.number_input("Application Order (0 to 9)", min_value=0, max_value=9, value=0, step=1)
    course = st.selectbox(
        "Course",
        [
            "33 - Biofuel Production Technologies",
            "171 - Animation and Multimedia Design",
            "8014 - Social Service (evening attendance)",
            "9003 - Agronomy",
            "9070 - Communication Design",
            "9085 - Veterinary Nursing",
            "9119 - Informatics Engineering",
            "9130 - Equinculture",
            "9147 - Management",
            "9238 - Social Service",
            "9254 - Tourism",
            "9500 - Nursing",
            "9556 - Oral Hygiene",
            "9670 - Advertising and Marketing Management",
            "9773 - Journalism and Communication",
            "9853 - Basic Education",
            "9991 - Management (evening attendance)"
        ]
    )
    daytime_evening_attendance = st.selectbox("Daytime/Evening Attendance", ["1 – Daytime", "0 - Evening"])

with st.sidebar.expander("Educational Background", expanded=False):
    previous_qualification = st.selectbox(
        "Previous Qualification",
        [
            "1 - Secondary education",
            "2 - Higher education - bachelor's degree",
            "3 - Higher education - degree",
            "4 - Higher education - master's",
            "5 - Higher education - doctorate",
            "6 - Frequency of higher education",
            "9 - 12th year of schooling - not completed",
            "10 - 11th year of schooling - not completed",
            "12 - Other - 11th year of schooling",
            "14 - 10th year of schooling",
            "15 - 10th year of schooling - not completed",
            "19 - Basic education 3rd cycle (9th/10th/11th year) or equiv.",
            "38 - Basic education 2nd cycle (6th/7th/8th year) or equiv.",
            "39 - Technological specialization course",
            "40 - Higher education - degree (1st cycle)",
            "42 - Professional higher technical course",
            "43 - Higher education - master (2nd cycle)"
        ]
    )
    previous_qualification_grade = st.slider("Previous Qualification Grade (0-200)", 0.0, 200.0, 150.0)
    admission_grade = st.slider("Admission Grade (0-200)", 0.0, 200.0, 150.0)
    nacionality = st.selectbox(
        "Nacionality",
        [
            "1 - Portuguese", "2 - German", "6 - Spanish", "11 - Italian", "13 - Dutch", "14 - English",
            "17 - Lithuanian", "21 - Angolan", "22 - Cape Verdean", "24 - Guinean", "25 - Mozambican",
            "26 - Santomean", "32 - Turkish", "41 - Brazilian", "62 - Romanian", "100 - Moldova (Republic of)",
            "101 - Mexican", "103 - Ukrainian", "105 - Russian", "108 - Cuban", "109 - Colombian"
        ]
    )

with st.sidebar.expander("Family Education & Occupation", expanded=False):
    mothers_qualification = st.selectbox(
        "Mother's Qualification",
        [
            "1 - Secondary Education - 12th Year of Schooling or Eq.",
            "2 - Higher Education - Bachelor's Degree",
            "3 - Higher Education - Degree",
            "4 - Higher Education - Master's",
            "5 - Higher Education - Doctorate",
            "6 - Frequency of Higher Education",
            "9 - 12th Year of Schooling - Not Completed",
            "10 - 11th Year of Schooling - Not Completed",
            "11 - 7th Year (Old)",
            "12 - Other - 11th Year of Schooling",
            "14 - 10th Year of Schooling",
            "18 - General commerce course",
            "19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
            "22 - Technical-professional course",
            "26 - 7th year of schooling",
            "27 - 2nd cycle of the general high school course",
            "29 - 9th Year of Schooling - Not Completed",
            "30 - 8th year of schooling",
            "34 - Unknown",
            "35 - Can't read or write",
            "36 - Can read without having a 4th year of schooling",
            "37 - Basic education 1st cycle (4th/5th year) or Equiv.",
            "38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
            "39 - Technological specialization course",
            "40 - Higher education - degree (1st cycle)",
            "41 - Specialized higher studies course",
            "42 - Professional higher technical course",
            "43 - Higher Education - Master (2nd cycle)",
            "44 - Higher Education - Doctorate (3rd cycle)"
        ]
    )
    fathers_qualification = st.selectbox(
        "Father's Qualification",
        [
            "1 - Secondary Education - 12th Year of Schooling or Eq.",
            "2 - Higher Education - Bachelor's Degree",
            "3 - Higher Education - Degree",
            "4 - Higher Education - Master's",
            "5 - Higher Education - Doctorate",
            "6 - Frequency of Higher Education",
            "9 - 12th Year of Schooling - Not Completed",
            "10 - 11th Year of Schooling - Not Completed",
            "11 - 7th Year (Old)",
            "12 - Other - 11th Year of Schooling",
            "13 - 2nd year complementary high school course",
            "14 - 10th Year of Schooling",
            "18 - General commerce course",
            "19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
            "20 - Complementary High School Course",
            "22 - Technical-professional course",
            "25 - Complementary High School Course - not concluded",
            "26 - 7th year of schooling",
            "27 - 2nd cycle of the general high school course",
            "29 - 9th Year of Schooling - Not Completed",
            "30 - 8th year of schooling",
            "31 - General Course of Administration and Commerce",
            "33 - Supplementary Accounting and Administration",
            "34 - Unknown",
            "35 - Can't read or write",
            "36 - Can read without having a 4th year of schooling",
            "37 - Basic education 1st cycle (4th/5th year) or Equiv.",
            "38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
            "39 - Technological specialization course",
            "40 - Higher education - degree (1st cycle)",
            "41 - Specialized higher studies course",
            "42 - Professional higher technical course",
            "43 - Higher Education - Master (2nd cycle)",
            "44 - Higher Education - Doctorate (3rd cycle)"
        ]
    )
    mothers_occupation = st.selectbox(
        "Mother's Occupation",
        [
            "0 - Student",
            "1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
            "2 - Specialists in Intellectual and Scientific Activities",
            "3 - Intermediate Level Technicians and Professions",
            "4 - Administrative staff",
            "5 - Personal Services, Security and Safety Workers and Sellers",
            "6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
            "7 - Skilled Workers in Industry, Construction and Craftsmen",
            "8 - Installation and Machine Operators and Assembly Workers",
            "9 - Unskilled Workers",
            "10 - Armed Forces Professions",
            "90 - Other Situation",
            "99 - (blank)",
            "122 - Health professionals",
            "123 - Teachers",
            "125 - Specialists in information and communication technologies (ICT)",
            "131 - Intermediate level science and engineering technicians and professions",
            "132 - Technicians and professionals, of intermediate level of health",
            "134 - Intermediate level technicians from legal, social, sports, cultural and similar services",
            "141 - Office workers, secretaries in general and data processing operators",
            "143 - Data, accounting, statistical, financial services and registry-related operators",
            "144 - Other administrative support staff"
        ]
    )
    fathers_occupation = st.selectbox(
        "Father's Occupation",
        [
            "0 - Student",
            "1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
            "2 - Specialists in Intellectual and Scientific Activities",
            "3 - Intermediate Level Technicians and Professions",
            "4 - Administrative staff",
            "5 - Personal Services, Security and Safety Workers and Sellers",
            "6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
            "7 - Skilled Workers in Industry, Construction and Craftsmen",
            "8 - Installation and Machine Operators and Assembly Workers",
            "9 - Unskilled Workers",
            "10 - Armed Forces Professions",
            "90 - Other Situation",
            "99 - (blank)",
            "101 - Armed Forces Officers",
            "102 - Armed Forces Sergeants",
            "103 - Other Armed Forces personnel",
            "112 - Directors of administrative and commercial services",
            "114 - Hotel, catering, trade and other services directors",
            "121 - Specialists in the physical sciences, mathematics, engineering and related techniques",
            "122 - Health professionals",
            "123 - Teachers",
            "124 - Specialists in finance, accounting, administrative organization, public and commercial relations",
            "131 - Intermediate level science and engineering technicians and professions",
            "132 - Technicians and professionals, of intermediate level of health",
            "134 - Intermediate level technicians from legal, social, sports, cultural and similar services",
            "135 - Information and communication technology technicians",
            "141 - Office workers, secretaries in general and data processing operators",
            "143 - Data, accounting, statistical, financial services and registry-related operators",
            "144 - Other administrative support staff"
        ]
    )

with st.sidebar.expander("Academic Performance & Economic Indicators", expanded=False):
    curricular_units_1st_sem_credited = st.number_input("Curricular Units 1st Sem (Credited)", min_value=0, max_value=100, value=30, step=1)
    curricular_units_1st_sem_enrolled = st.number_input("Curricular Units 1st Sem (Enrolled)", min_value=0, max_value=100, value=30, step=1)
    curricular_units_1st_sem_evaluations = st.number_input("Curricular Units 1st Sem (Evaluations)", min_value=0, max_value=100, value=30, step=1)
    curricular_units_1st_sem_approved = st.number_input("Curricular Units 1st Sem (Approved)", min_value=0, max_value=100, value=30, step=1)
    curricular_units_1st_sem_grade = st.slider("Curricular Units 1st Sem (Grade) (0-20)", 0.0, 20.0, 10.0)
    curricular_units_1st_sem_without_evaluations = st.number_input("Curricular Units 1st Sem (Without Evaluations)", min_value=0, max_value=100, value=0, step=1)
    curricular_units_2nd_sem_credited = st.number_input("Curricular Units 2nd Sem (Credited)", min_value=0, max_value=100, value=30, step=1)
    curricular_units_2nd_sem_enrolled = st.number_input("Curricular Units 2nd Sem (Enrolled)", min_value=0, max_value=100, value=30, step=1)
    curricular_units_2nd_sem_evaluations = st.number_input("Curricular Units 2nd Sem (Evaluations)", min_value=0, max_value=100, value=30, step=1)
    curricular_units_2nd_sem_approved = st.number_input("Curricular Units 2nd Sem (Approved)", min_value=0, max_value=100, value=30, step=1)
    curricular_units_2nd_sem_grade = st.slider("Curricular Units 2nd Sem (Grade) (0-20)", 0.0, 20.0, 10.0)
    curricular_units_2nd_sem_without_evaluations = st.number_input("Curricular Units 2nd Sem (Without Evaluations)", min_value=0, max_value=100, value=0, step=1)
    unemployment_rate = st.number_input("Unemployment Rate (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    inflation_rate = st.number_input("Inflation Rate (%)", min_value=-10.0, max_value=50.0, value=1.0, step=0.1)
    GDP = st.number_input("GDP", min_value=-10.0, max_value=50.0, value=1.0, step=0.1)

with st.sidebar.expander("Binary Options", expanded=False):
    displaced = st.selectbox("Displaced", ["1 – Yes", "0 – No"])
    educational_special_needs = st.selectbox("Educational Special Needs", ["1 – Yes", "0 – No"])
    debtor = st.selectbox("Debtor", ["1 – Yes", "0 – No"])
    tuition_fees_up_to_date = st.selectbox("Tuition Fees Up to Date", ["1 – Yes", "0 – No"])
    scholarship_holder = st.selectbox("Scholarship Holder", ["1 – Yes", "0 – No"])
    international = st.selectbox("International", ["1 – Yes", "0 – No"])
    age_at_enrollment = st.number_input("Age at Enrollment", min_value=17, max_value=100, value=18, step=1)
    gender = st.selectbox("Gender", ["1 – Male", "0 – Female"])

# ---------------------------
# Build Input Feature DataFrame & Feature Engineering
# ---------------------------
data = {
    "Marital_status": extract_numeric(marital_status),
    "Application_mode": extract_numeric(application_mode),
    "Application_order": application_order,
    "Course": extract_numeric(course),
    "Daytime_evening_attendance": extract_numeric(daytime_evening_attendance),
    "Previous_qualification": extract_numeric(previous_qualification),
    "Previous_qualification_grade": previous_qualification_grade,
    "Nacionality": extract_numeric(nacionality),
    "Mothers_qualification": extract_numeric(mothers_qualification),
    "Fathers_qualification": extract_numeric(fathers_qualification),
    "Mothers_occupation": extract_numeric(mothers_occupation),
    "Fathers_occupation": extract_numeric(fathers_occupation),
    "Admission_grade": admission_grade,
    "Displaced": 1 if displaced.split(" – ")[0] == "1" or displaced.split(" - ")[0] == "1" else 0,
    "Educational_special_needs": 1 if educational_special_needs.split(" – ")[0] == "1" or educational_special_needs.split(" - ")[0] == "1" else 0,
    "Debtor": 1 if debtor.split(" – ")[0] == "1" or debtor.split(" - ")[0] == "1" else 0,
    "Tuition_fees_up_to_date": 1 if tuition_fees_up_to_date.split(" – ")[0] == "1" or tuition_fees_up_to_date.split(" - ")[0] == "1" else 0,
    "Gender": extract_numeric(gender),
    "Scholarship_holder": 1 if scholarship_holder.split(" – ")[0] == "1" or scholarship_holder.split(" - ")[0] == "1" else 0,
    "Age_at_enrollment": age_at_enrollment,
    "International": 1 if international.split(" – ")[0] == "1" or international.split(" - ")[0] == "1" else 0,
    "Curricular_units_1st_sem_credited": curricular_units_1st_sem_credited,
    "Curricular_units_1st_sem_enrolled": curricular_units_1st_sem_enrolled,
    "Curricular_units_1st_sem_evaluations": curricular_units_1st_sem_evaluations,
    "Curricular_units_1st_sem_approved": curricular_units_1st_sem_approved,
    "Curricular_units_1st_sem_grade": curricular_units_1st_sem_grade,
    "Curricular_units_1st_sem_without_evaluations": curricular_units_1st_sem_without_evaluations,
    "Curricular_units_2nd_sem_credited": curricular_units_2nd_sem_credited,
    "Curricular_units_2nd_sem_enrolled": curricular_units_2nd_sem_enrolled,
    "Curricular_units_2nd_sem_evaluations": curricular_units_2nd_sem_evaluations,
    "Curricular_units_2nd_sem_approved": curricular_units_2nd_sem_approved,
    "Curricular_units_2nd_sem_grade": curricular_units_2nd_sem_grade,
    "Curricular_units_2nd_sem_without_evaluations": curricular_units_2nd_sem_without_evaluations,
    "Unemployment_rate": unemployment_rate,
    "Inflation_rate": inflation_rate,
    "GDP": GDP
}
input_df = pd.DataFrame(data, index=[0])
# Engineered Features
input_df["Grade_Progression"] = input_df["Curricular_units_2nd_sem_grade"] - input_df["Curricular_units_1st_sem_grade"]
input_df["Attendance_Consistency"] = input_df["Curricular_units_1st_sem_approved"] / (input_df["Curricular_units_1st_sem_enrolled"] + 1e-5)
input_df["Age_Group"] = pd.cut(input_df["Age_at_enrollment"], bins=[15,20,25,30,35,100],
                               labels=["15-20", "21-25", "26-30", "31-35", "36+"])

# ---------------------------
# Main Section: Prediction & Processed Input Review
# ---------------------------
st.markdown("## Prediction Section")
if st.button("Predict Dropout"):
    # One-Hot Encode categorical columns (including engineered Age_Group)
    cat_cols = [
        "Marital_status",
        "Application_mode",
        "Course",
        "Previous_qualification",
        "Nacionality",
        "Mothers_qualification",
        "Fathers_qualification",
        "Mothers_occupation",
        "Fathers_occupation",
        "Age_Group"
    ]
    input_encoded = encoder.transform(input_df[cat_cols])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(cat_cols), index=input_df.index)
    
    # Concatenate with the rest of the features
    input_proc = input_df.drop(columns=cat_cols)
    input_proc = pd.concat([input_proc.reset_index(drop=True), input_encoded_df.reset_index(drop=True)], axis=1)
    
    # Scale numerical features and reorder columns
    input_proc[numerical_columns] = scaler.transform(input_proc[numerical_columns])
    input_proc = input_proc[training_columns]
    
    # Make the prediction
    prediction = model.predict(input_proc)
    status_map = {0: "Dropout", 1: "Graduate", 2: "Enrolled"}
    
    # Layout the prediction result and processed input side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction Result")
        st.success(f"Predicted Academic Status: **{status_map[prediction[0]]}**")
    with col2:
        st.subheader("Processed Input Features")
        st.dataframe(input_proc)

# ---------------------------
# Optional: Model Evaluation & Visualizations in Tabs
# ---------------------------
if st.sidebar.checkbox("Show Model Evaluation & Visualizations"):
    st.markdown("## Model Evaluation & Visualizations")
    tab1, tab2, tab3 = st.tabs(["Target & Marital Distribution", "Age & Gender", "Advanced Metrics"])
    
    # Tab 1: Pie Charts
    with tab1:
        try:
            df_raw = pd.read_csv("data.csv", delimiter=";")
            df_raw.columns = (
                df_raw.columns.str.strip()
                .str.replace(" ", "_")
                .str.replace("/", "_")
                .str.replace(r"\(", "", regex=True)
                .str.replace(r"\)", "", regex=True)
                .str.replace("'", "", regex=True)
            )
            target_counts = df_raw["Target"].value_counts().sort_index()
            fig1, ax1 = plt.subplots(figsize=(6,6))
            ax1.pie(target_counts, labels=["Dropout", "Graduate", "Enrolled"], autopct="%1.1f%%", startangle=140)
            ax1.set_title("Distribution of Target")
            st.pyplot(fig1)
            
            marital_counts = df_raw["Marital_status"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6,6))
            ax2.pie(marital_counts, labels=marital_counts.index, autopct="%1.1f%%", startangle=140)
            ax2.set_title("Marital Status Distribution")
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error in Tab 1 visualizations: {e}")
    
    # Tab 2: Histograms & Bar Charts
    with tab2:
        try:
            fig3, ax3 = plt.subplots(figsize=(8,6))
            ax3.hist(df_raw["Age_at_enrollment"], bins=20, color="skyblue", edgecolor="black")
            ax3.set_title("Age Distribution of Students")
            ax3.set_xlabel("Age at Enrollment")
            ax3.set_ylabel("Count")
            st.pyplot(fig3)
            
            gender_map = {1: "Male", 0: "Female"}
            gender_counts = df_raw["Gender"].map(gender_map).value_counts()
            fig4, ax4 = plt.subplots(figsize=(6,4))
            sns.barplot(x=gender_counts.index, y=gender_counts.values, palette="pastel", ax=ax4)
            ax4.set_title("Gender Count of Students")
            ax4.set_ylabel("Count")
            st.pyplot(fig4)
        except Exception as e:
            st.error(f"Error in Tab 2 visualizations: {e}")
    
    # Tab 3: Advanced Visualizations (Courses, Financial, Correlation, ROC, etc.)
    with tab3:
        try:
            course_counts = df_raw["Course"].value_counts()
            fig5, ax5 = plt.subplots(figsize=(10,6))
            sns.barplot(x=course_counts.index.astype(str), y=course_counts.values, palette="coolwarm", ax=ax5)
            ax5.set_title("Courses Enrolled by Students")
            ax5.set_xlabel("Course")
            ax5.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig5)
            
            # Correlation Heatmap
            numeric_df = df_raw.select_dtypes(include=["number"])
            fig6, ax6 = plt.subplots(figsize=(14,12))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax6)
            ax6.set_title("Correlation Heatmap")
            st.pyplot(fig6)
            
            # ROC Curve for Multi-class Classification
            def preprocess_data(df_raw):
                columns_to_keep = [
                    "Marital_status", "Application_mode", "Application_order", "Course", "Daytime_evening_attendance",
                    "Previous_qualification", "Previous_qualification_grade", "Nacionality", "Mothers_qualification",
                    "Fathers_qualification", "Mothers_occupation", "Fathers_occupation", "Admission_grade",
                    "Displaced", "Educational_special_needs", "Debtor", "Tuition_fees_up_to_date", "Gender",
                    "Scholarship_holder", "Age_at_enrollment", "International", "Curricular_units_1st_sem_credited",
                    "Curricular_units_1st_sem_enrolled", "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved",
                    "Curricular_units_1st_sem_grade", "Curricular_units_1st_sem_without_evaluations", "Curricular_units_2nd_sem_credited",
                    "Curricular_units_2nd_sem_enrolled", "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
                    "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_without_evaluations", "Unemployment_rate",
                    "Inflation_rate", "GDP", "Target"
                ]
                df_proc = df_raw[columns_to_keep].copy()
                df_proc["Grade_Progression"] = df_proc["Curricular_units_2nd_sem_grade"] - df_proc["Curricular_units_1st_sem_grade"]
                df_proc["Attendance_Consistency"] = df_proc["Curricular_units_1st_sem_approved"] / (df_proc["Curricular_units_1st_sem_enrolled"] + 1e-5)
                df_proc["Age_Group"] = pd.cut(df_proc["Age_at_enrollment"], bins=[15,20,25,30,35,100],
                                             labels=["15-20", "21-25", "26-30", "31-35", "36+"])
                cat_cols = [
                    "Marital_status", "Application_mode", "Course", "Previous_qualification", "Nacionality",
                    "Mothers_qualification", "Fathers_qualification", "Mothers_occupation", "Fathers_occupation", "Age_Group"
                ]
                input_encoded = encoder.transform(df_proc[cat_cols])
                encoded_df = pd.DataFrame(
                    input_encoded,
                    columns=encoder.get_feature_names_out(cat_cols),
                    index=df_proc.index,
                )
                df_proc = pd.concat([df_proc.drop(columns=cat_cols), encoded_df], axis=1)
                num_cols = [
                    "Previous_qualification_grade", "Admission_grade", "Curricular_units_1st_sem_credited",
                    "Curricular_units_1st_sem_enrolled", "Curricular_units_1st_sem_evaluations",
                    "Curricular_units_1st_sem_approved", "Curricular_units_1st_sem_grade",
                    "Curricular_units_1st_sem_without_evaluations", "Curricular_units_2nd_sem_credited",
                    "Curricular_units_2nd_sem_enrolled", "Curricular_units_2nd_sem_evaluations",
                    "Curricular_units_2nd_sem_approved", "Curricular_units_2nd_sem_grade",
                    "Curricular_units_2nd_sem_without_evaluations", "Age_at_enrollment", "Unemployment_rate",
                    "Inflation_rate", "GDP", "Grade_Progression", "Attendance_Consistency"
                ]
                df_proc[num_cols] = scaler.transform(df_proc[num_cols])
                y_true = df_raw["Target"].map({"Dropout": 0, "Graduate": 1, "Enrolled": 2})
                df_proc = df_proc.drop(columns=["Target"])
                X_proc = df_proc[training_columns]
                return X_proc, y_true

            X_all, y_all = preprocess_data(df_raw)
            y_score = model.predict_proba(X_all)
            y_all_bin = label_binarize(y_all, classes=[0, 1, 2])
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(3):
                fpr[i], tpr[i], _ = roc_curve(y_all_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            fig7, ax7 = plt.subplots(figsize=(8,6))
            colors = ["blue", "green", "red"]
            for i, color in zip(range(3), colors):
                ax7.plot(fpr[i], tpr[i], color=color,
                         label=f"ROC curve for class {i} (area = {roc_auc[i]:.2f})")
            ax7.plot([0,1], [0,1], "k--")
            ax7.set_xlim([0.0,1.0])
            ax7.set_ylim([0.0,1.05])
            ax7.set_xlabel("False Positive Rate")
            ax7.set_ylabel("True Positive Rate")
            ax7.set_title("ROC Curve for Multi-class Classification")
            ax7.legend(loc="lower right")
            st.pyplot(fig7)
        except Exception as e:
            st.error(f"Error in Advanced Metrics: {e}")
