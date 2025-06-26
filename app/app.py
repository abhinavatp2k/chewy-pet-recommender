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

# ---------- STYLING ---------- #
st.set_page_config(page_title="Dog Product Recommender 🐶", page_icon="🐾")
st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1619983081563-dff3f9da3f71');
        background-size: cover;
        background-attachment: fixed;
    }
    .main {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 10px;
    }
    h1, h2, h3, .st-emotion-cache-1v0mbdj {
        color: #fff;
        text-shadow: 1px 1px 2px #000;
    }
    .stButton > button {
        background-color: #ff914d;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff7b00;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- STREAMLIT UI ---------- #
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🐶 Dog Product Recommender")

breeds = encoder.classes_.tolist()
breed = st.selectbox("🐕 Select Breed", breeds)
weight = st.slider("⚖️ Weight (kg)", min_value=1.0, max_value=80.0, value=25.0)
height = st.slider("📏 Height at shoulder (cm)", min_value=5.0, max_value=100.0, value=30.0)
symptom = st.text_area("💬 Describe your dog’s main symptom / issue", placeholder="e.g. itchy skin and shedding")

if st.button("🎯 Get Recommendations"):
    if symptom.strip() == "":
        st.error("🚨 Please describe a symptom or health issue.")
        st.stop()

    try:
        breed_enc = encoder.transform([breed])[0]
    except:
        st.error("🚨 Breed not recognized.")
        st.stop()

    try:
        symptom_key = encoder.classes_[encoder.classes_.tolist().index(symptom.strip().split()[0])]
        symptom_enc = encoder.transform([symptom_key])[0]
    except:
        symptom_enc = 0

    X_input = pd.DataFrame(
        [[breed_enc, weight, height, symptom_enc]],
        columns=["breed_enc", "weight", "height", "symptom_enc"]
    )
    pred_label_enc = model.predict(X_input)[0]
    pred_category  = encoder.inverse_transform([pred_label_enc])[0]

    st.success(f"🏷 **Predicted Health Need:** {pred_category}")

    # ----- Recommend products
    prods = category2prods.get(pred_category, [])
    st.markdown("#### 🛒 Recommended Products")
    if not prods:
        st.info("No matching products found.")
    for p in prods:
        st.markdown(f"- 🐾 **{p['product_name']}** — {p['description']}")

    # ----- Health Summary
    breed_row = breed_df[breed_df["breed"] == breed].iloc[0]
    low_high_kg = f"{breed_row['min_weight']:.0f}–{breed_row['max_weight']:.0f} kg"
    low_high_lb = f"{breed_row['min_weight']*2.2:.0f}–{breed_row['max_weight']*2.2:.0f} lbs"
    weight_note = "✅ Weight is within typical range." if breed_row["min_weight"] <= weight <= breed_row["max_weight"] else "⚠️ Weight outside typical range — consider diet/exercise."

    st.markdown("#### 🩺 Personalized Health Summary")
    st.markdown(
        f"🐕 **{breed}** typically weighs **{low_high_kg}** ({low_high_lb}).  \n"
        f"Your dog weighs **{weight} kg** ({weight*2.2:.0f} lbs). {weight_note}  \n\n"
        f"Based on your symptom, our system identified **{pred_category.lower()}** as the area of concern. "
        f"🧠 Consider the above recommendations and consult your vet if symptoms persist."
    )
st.markdown('</div>', unsafe_allow_html=True)

