import streamlit as st
import pandas as pd
import joblib
import requests, csv
from io import StringIO

# ---------- CONFIG ---------- #
BREED_CSV_URL = "https://raw.githubusercontent.com/tmfilho/akcdata/master/data/akc-data-latest.csv"

# ---------- LOADING DATA ---------- #
@st.cache_data
def load_breed_data():
    df = pd.read_csv(BREED_CSV_URL)
    if "breed" not in df.columns:
        df.rename(columns={df.columns[0]: "breed"}, inplace=True)
    return df[["breed", "min_weight", "max_weight", "min_height", "max_height"]]

@st.cache_resource
def load_models():
    model   = joblib.load("models/chewy_recommender_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return model, encoder

breed_df         = load_breed_data()
model, encoder   = load_models()

# ---------- LOADING CHEWY PRODUCTS ---------- #
@st.cache_data
def load_products():
    url  = "https://raw.githubusercontent.com/abhinavatp2k/chewy-pet-recommender/main/data/chewy_products.csv"
    txt  = requests.get(url).text
    f    = StringIO(txt)
    r    = csv.reader(f)
    header = next(r)
    rows   = []
    for row in r:
        if len(row) > 3:
            row = row[:2] + [",".join(row[2:]).strip('"')]
        rows.append(row)
    return pd.DataFrame(rows, columns=header)

prod_df = load_products()

# Map category -> list of product dicts
category2prods = (
    prod_df.groupby("category")[["product_name", "description"]]
    .apply(lambda x: x.to_dict("records"))
    .to_dict()
)

# ---------- STREAMLIT UI ---------- #
st.set_page_config(page_title="Dog Product Recommender 🐾", page_icon="🐶")
st.title("🐶 Dog Product Recommender")

# --- User Inputs
breed   = st.selectbox("Select Breed", sorted(breed_df["breed"].unique()))
weight  = st.number_input("Weight (kg)", min_value=1.0, step=0.5)
height  = st.number_input("Height at shoulder (cm)", min_value=5.0, step=0.5)
symptom = st.text_area("Describe your dog's main symptom / issue", placeholder="e.g. itchy skin and shedding")

if st.button("Get Recommendations"):
    if symptom.strip() == "":
        st.error("Please describe a symptom or health issue.")
        st.stop()

    # ----- Prediction
    # Very tiny encoders: convert breed + symptom into encoded features
    breed_enc   = encoder.transform([model.classes_[0]])[0]  # placeholder not used
    symptom_enc = encoder.transform([encoder.classes_[0]])[0]  # likewise

    # Quick re-encode with fit classes (simulate)
    try:
        symptom_enc = encoder.transform([encoder.classes_[encoder.classes_.tolist().index(symptom.strip().split()[0])]])[0]
    except:
        symptom_enc = 0  # fallback

    X_input = pd.DataFrame(
        [[breed_enc, weight, height, symptom_enc]],
        columns=["breed_enc", "weight", "height", "symptom_enc"]
    )
    pred_label_enc = model.predict(X_input)[0]
    pred_category  = encoder.inverse_transform([pred_label_enc])[0]

    st.success(f"### 🏷 Predicted Need: **{pred_category}**")

    # ----- Recommend products
    prods = category2prods.get(pred_category, [])
    st.markdown("#### 🛒 Recommended Products")
    if not prods:
        st.info("No matching products found.")
    for p in prods:
        st.markdown(f"- **{p['product_name']}** — {p['description']}")

    # ----- Health Summary
    breed_row = breed_df[breed_df["breed"] == breed].iloc[0]
    low_high  = f"{breed_row['min_weight']:.0f}–{breed_row['max_weight']:.0f} kg"
    weight_note = "✅ Weight is within typical range." if breed_row["min_weight"] <= weight <= breed_row["max_weight"] else "⚠️ Weight outside typical range — consider diet/exercise."

    st.markdown("#### 🩺 Personalized Health Summary")
    st.markdown(
        f"{breed} typically weighs **{low_high}**. Your dog is **{weight} kg**. {weight_note}  \n\n"
        f"Based on the symptom, we detected **{pred_category.lower()}** concerns. "
        f"Consider the products above and consult your veterinarian if problems persist."
    )
