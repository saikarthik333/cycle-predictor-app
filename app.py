import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- App UI Configuration ---
st.set_page_config(page_title="Intelligent Cycle Predictor", page_icon="🧠", layout="wide")
st.title("🧠 Intelligent Cycle Prediction Engine")


# --- Data Loading and Preprocessing ---
@st.cache_data # Cache the data for performance
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None, None

    features_to_use = [
        'ClientID', 'CycleNumber', 'Age', 'Yearsmarried', 'Schoolyears',
        'Height', 'Weight', 'BMI', 'Numberpreg', 'Livingkids',
        'LengthofLutealPhase', 'TotalMensesScore', 'LengthofCycle'
    ]
    df_processed = df[features_to_use].copy()

    for col in df_processed.columns:
        if col != 'ClientID':
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    df_processed.dropna(subset=['LengthofCycle'], inplace=True)
    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            
    # Define features (X) and target (y) for the general model
    X_all = df_processed.drop(['ClientID', 'LengthofCycle'], axis=1)
    y_all = df_processed['LengthofCycle']
    
    return df_processed, (X_all, y_all)

# --- Model Training ---
@st.cache_resource # Cache the trained model for performance
def train_general_model(X, y):
    """Trains a general model on the entire dataset."""
    model = RandomForestRegressor(n_estimators=150, random_state=42, oob_score=True)
    model.fit(X, y)
    return model

# --- Main App ---
df, all_data = load_and_preprocess_data('FedCycleData071012 (2).csv')

if df is None:
    st.error("🚨 **Error:** `FedCycleData071012 (2).csv` not found. Please place the data file in the same folder as `app.py`.")
else:
    # Train the general model that will be used for manual predictions
    X_all, y_all = all_data
    general_model = train_general_model(X_all, y_all)

    # --- UI: Mode Selection Sidebar ---
    st.sidebar.header("Select Application Mode")
    app_mode = st.sidebar.radio(
        "Choose what you want to do:",
        ("Explore Predictions for Kaggle Users", "Get a Prediction with My Own Data")
    )

    # --- MODE 1: Explore Kaggle Data (Your original feature) ---
    if app_mode == "Explore Predictions for Kaggle Users":
        st.header("Mode 1: Explore Predictions for Kaggle Users")
        st.markdown("Here, a unique model is trained for *each selected user* to demonstrate hyper-personalization.")
        
        user_cycle_counts = df['ClientID'].value_counts()
        valid_users = user_cycle_counts[user_cycle_counts > 5].index.tolist()

        selected_user = st.selectbox("Select a User ID to see their prediction:", valid_users)

        if selected_user:
            user_df = df[df['ClientID'] == selected_user].sort_values(by='CycleNumber').copy()
            X_user = user_df.drop(['ClientID', 'LengthofCycle'], axis=1)
            y_user = user_df['LengthofCycle']
            
            # Train model specifically for this user
            user_model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
            user_model.fit(X_user.iloc[:-1], y_user.iloc[:-1])
            
            # Predict the last cycle
            prediction = user_model.predict(X_user.iloc[-1:])
            predicted_length = round(prediction[0])
            y_test_actual = y_user.iloc[-1].item()

            col1, col2 = st.columns(2)
            col1.metric("Predicted Cycle Length", f"{predicted_length} days", f"{round(predicted_length - y_test_actual, 1)} days vs Actual", delta_color="inverse")
            col2.metric("Actual Cycle Length", f"{y_test_actual} days")
            st.info(f"This user-specific model's accuracy score was: {user_model.oob_score_:.2%}", icon="✅")

    # --- MODE 2: Manual User Input (Your new feature) ---
    elif app_mode == "Get a Prediction with My Own Data":
        st.header("Mode 2: Get a Prediction with Your Own Data")
        st.markdown("Enter your information below. A **general model**, trained on all users, will make a prediction.")

        with st.form("user_input_form"):
            st.subheader("Your Health & Demographic Info")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=10, max_value=60, value=30)
                height_cm = st.number_input("Height (cm)", min_value=140, max_value=200, value=165)
                weight_kg = st.number_input("Weight (kg)", min_value=40, max_value=150, value=60)
            with col2:
                number_preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
                living_kids = st.number_input("Number of Living Children", min_value=0, max_value=20, value=1)
                school_years = st.number_input("Years of Schooling", min_value=0, max_value=25, value=16)
            with col3:
                years_married = st.number_input("Years Married", min_value=0, max_value=50, value=5)
                bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
                st.metric("Your Calculated BMI", bmi)

            st.subheader("Your Last Cycle's Info")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                last_cycle_number = st.number_input("What cycle number was your last one? (e.g., 5th)", min_value=1, value=10)
            with col_b:
                luteal_phase = st.number_input("Last Luteal Phase Length (days)", min_value=5, max_value=25, value=14)
            with col_c:
                menses_score = st.number_input("Total Menses Score (if known)", min_value=0, max_value=50, value=15)

            submitted = st.form_submit_button("🚀 Predict My Next Cycle Length")

        if submitted:
            # Create a dataframe from the user's input
            input_data = {
                'CycleNumber': [last_cycle_number], 'Age': [age], 'Yearsmarried': [years_married],
                'Schoolyears': [school_years], 'Height': [height_cm], 'Weight': [weight_kg],
                'BMI': [bmi], 'Numberpreg': [number_preg], 'Livingkids': [living_kids],
                'LengthofLutealPhase': [luteal_phase], 'TotalMensesScore': [menses_score]
            }
            input_df = pd.DataFrame(input_data)
            
            # Ensure the order of columns matches the training data
            input_df = input_df[X_all.columns]
            
            # Use the general model to make a prediction
            prediction = general_model.predict(input_df)
            predicted_length = round(prediction[0])
            
            st.success(f"### The model predicts your next cycle will be **{predicted_length} days** long.")