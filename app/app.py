import streamlit as st
import pandas as pd
import joblib
import requests, csv
from io import StringIO

# ---------- CONFIG ---------- #
BREED_CSV_URL = "https://raw.githubusercontent.com/tmfilho/akcdata/master/data/akc-data-latest.csv"

# ---------- CUSTOM CSS ---------- #
st.markdown("""
    <style>
        body {
            background-image: url("https://images.unsplash.com/photo-1601758064222-6ec2a0fef803");
            background-size: cover;
            background-attachment: fixed;
            color: white;
        }
        .stApp {
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 1rem;
        }
        .css-1d391kg {color: white;}
        h1 {
            text-align: center;
            color: #FFB800;
            text-shadow: 1px 1px 2px black;
        }
        .stButton>button {
            color: white;
            background-color: #FF7F50;
            font-weight: bold;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ---------- #
@st.cache_data
def load_breed_data():
    df = pd.read_csv(BREED_CSV_URL)
    if "breed" not in df.columns:
        df.rename(columns={df.columns[0]: "breed"}, inplace=True)
    return df[["breed", "min_weight", "max_weight", "min_height", "max_height"]]

@st.cache_resource
def load_models():
    model = joblib.load("models/chewy_recommender_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return model, encoder

breed_df = load_breed_data()
model, encoder = load_models()

@st.cache_data
def load_products():
    url = "https://raw.githubusercontent.com/abhinavatp2k/chewy-pet-recommender/main/data/chewy_products.csv"
    txt = requests.get(url).text
    f = StringIO(txt)
    r = csv.reader(f)
    header = next(r)
    rows = []
    for row in r:
        if len(row) > 3:
            row = row[:2] + [",".join(row[2:]).strip('"')]
        rows.append(row)
    return pd.DataFrame(rows, columns=header)

prod_df = load_products()

# Map category -> product records
category2prods = (
    prod_df.groupby("category")[["product_name", "description"]]
    .apply(lambda x: x.to_dict("records"))
    .to_dict()
)

# ---------- UI ---------- #
st.set_page_config(page_title="Dog Product Recommender 🐶", page_icon="🐾")
st.title("🐾 Dog Product Recommender")

# Input
breeds = encoder.classes_.tolist()
breed = st.selectbox("🐕 Select Breed", breeds)
weight_kg = st.number_input("🐾 Weight (kg)", min_value=1.0, step=0.5)
height_cm = st.number_input("📏 Height at shoulder (cm)", min_value=5.0, step=0.5)
symptom = st.text_area("🩺 Describe your dog's main symptom", placeholder="e.g. itchy skin, shedding")

if st.button("🦴 Get Recommendations"):
    if symptom.strip() == "":
        st.error("Please describe a symptom or health issue.")
        st.stop()

    # Encode input
    try:
        breed_enc = encoder.transform([breed])[0]
    except:
        st.error("Breed not recognized!")
        st.stop()

    try:
        symptom_enc = encoder.transform([encoder.classes_[encoder.classes_.tolist().index(symptom.strip().split()[0])]])[0]
    except:
        symptom_enc = 0

    # Build input
    X_input = pd.DataFrame(
        [[breed_enc, weight_kg, height_cm, symptom_enc]],
        columns=["breed_enc", "weight", "height", "symptom_enc"]
    )

    pred_label_enc = model.predict(X_input)[0]
    pred_category = encoder.inverse_transform([pred_label_enc])[0]

    st.success(f"🎯 **Predicted Category:** {pred_category}")

    st.markdown("### 🛒 Recommended Products")
    prods = category2prods.get(pred_category, [])
    if not prods:
        st.warning("No product recommendations found.")
    else:
        for p in prods:
            st.markdown(f"🐶 **{p['product_name']}** — {p['description']}")

    # --- Health Summary
    breed_row = breed_df[breed_df["breed"] == breed].iloc[0]
    min_wt, max_wt = breed_row["min_weight"], breed_row["max_weight"]
    low_high = f"{min_wt:.0f}–{max_wt:.0f} kg"
    wt_status = "✅ Within normal range." if min_wt <= weight_kg <= max_wt else "⚠️ Check with vet — unusual weight."

    st.markdown("### 🩺 Personalized Health Summary")
    st.markdown(
        f"- 🐾 {breed} typically weighs **{low_high}**.\n"
        f"- 📦 Your dog weighs **{weight_kg} kg** → {wt_status}\n"
        f"- 💡 Based on symptoms, we suggest products from the **{pred_category}** category.\n\n"
        f"🩺 Consult a vet for further advice if symptoms persist."
    )

