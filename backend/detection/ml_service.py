import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
import zipfile


# ==========================================
# MODEL PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

RF_MODEL_PATH = MODEL_DIR / "random_forest.pkl"
XGB_MODEL_PATH = MODEL_DIR / "xgboost.json"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"

RF_ZIP_PATH = MODEL_DIR / "random_forest.zip"
ENCODER_ZIP_PATH = MODEL_DIR / "encoder.zip"


# ==========================================
# LOAD MODEL
# ==========================================

if not RF_MODEL_PATH.exists():
    with zipfile.ZipFile(RF_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

if not ENCODER_PATH.exists():
    with zipfile.ZipFile(ENCODER_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)


rf_model = joblib.load(RF_MODEL_PATH)

xgb_model = XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)

encoder = joblib.load(ENCODER_PATH)


# ==========================================
# SINGLE TRANSACTION PREDICTION
# ==========================================

def predict_transaction(
    from_address,
    to_address,
    value,
    block_height,
    hour,
    model_name
):
    """
    Melakukan prediction terhadap satu transaksi Ethereum.
    """

    # ------------------------------------------
    # Encode From dan To address
    # ------------------------------------------

    address_data = pd.DataFrame({
        "From": [from_address],
        "To": [to_address],
    })

    encoded_addresses = encoder.transform(address_data)

    from_encoded = encoded_addresses[0][0]
    to_encoded = encoded_addresses[0][1]


    # ------------------------------------------
    # Pilih model dan feature
    # ------------------------------------------

    if model_name == "random_forest":

        model = rf_model

        input_data = pd.DataFrame([
            {
                "From_encoded": from_encoded,
                "To_encoded": to_encoded,
                "Value": float(value),
                "BlockHeight": int(block_height),
            }
        ])

    elif model_name == "xgboost":

        model = xgb_model

        input_data = pd.DataFrame([
            {
                "From_encoded": from_encoded,
                "To_encoded": to_encoded,
                "Value": float(value),
                "BlockHeight": int(block_height),
                "Hour": int(hour),
            }
        ])

    else:

        raise ValueError("Model tidak dikenali.")


    # ------------------------------------------
    # Prediction
    # ------------------------------------------

    prediction = model.predict(input_data)[0]


    # ------------------------------------------
    # Probability
    # ------------------------------------------

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        probability = float(max(probabilities))


    # ------------------------------------------
    # Label
    # ------------------------------------------

    if int(prediction) == 1:

        label = "Anomaly-indicating"

    else:

        label = "Normal"


    # ------------------------------------------
    # Return result
    # ------------------------------------------

    return {
        "prediction": int(prediction),
        "label": label,
        "model": model_name,
        "probability": probability,
    }


# ==========================================
# BATCH PREDICTION
# ==========================================

def predict_batch(df, model_name):
    """
    Melakukan prediction terhadap banyak transaksi
    yang berasal dari sebuah DataFrame.
    """


    # ------------------------------------------
    # Validasi model
    # ------------------------------------------

    if model_name not in ["random_forest", "xgboost"]:

        raise ValueError("Model tidak dikenali.")


    # ------------------------------------------
    # Validasi kolom
    # ------------------------------------------

    required_columns = [
        "From",
        "To",
        "Value",
        "BlockHeight",
        "Hour",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Kolom berikut tidak ditemukan: "
            + ", ".join(missing_columns)
        )


    # ------------------------------------------
    # Copy data
    # ------------------------------------------

    data = df.copy()


    # ------------------------------------------
    # Encode From dan To
    # ------------------------------------------

    address_data = data[["From", "To"]]

    encoded_addresses = encoder.transform(address_data)

    data["From_encoded"] = encoded_addresses[:, 0]

    data["To_encoded"] = encoded_addresses[:, 1]


    # ------------------------------------------
    # Pilih model dan feature
    # ------------------------------------------

    if model_name == "random_forest":

        model = rf_model

        input_data = data[
            [
                "From_encoded",
                "To_encoded",
                "Value",
                "BlockHeight",
            ]
        ]

    else:

        model = xgb_model

        input_data = data[
            [
                "From_encoded",
                "To_encoded",
                "Value",
                "BlockHeight",
                "Hour",
            ]
        ]


    # ------------------------------------------
    # Prediction
    # ------------------------------------------

    predictions = model.predict(input_data)


    # ------------------------------------------
    # Probability
    # ------------------------------------------

    probabilities = model.predict_proba(input_data)

    max_probabilities = probabilities.max(axis=1)


    # ------------------------------------------
    # Add prediction results
    # ------------------------------------------

    data["Prediction"] = predictions.astype(int)

    data["Label"] = [
        "Anomaly-indicating"
        if prediction == 1
        else "Normal"
        for prediction in predictions
    ]

    data["Probability"] = max_probabilities


    # ------------------------------------------
    # Return result
    # ------------------------------------------

    return data