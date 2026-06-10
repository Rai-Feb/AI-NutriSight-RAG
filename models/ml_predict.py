import joblib
import pandas as pd
import os

# Path otomatis ke folder models
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "Prediksi_Stunting.pkl")
LE_PATH = os.path.join(BASE_DIR, "LabelEncoder_Gender.pkl")


def predict_stunting(age_months, weight_kg, height_cm, gender):
    """
    Prediksi status stunting anak berdasarkan model Random Forest.

    Parameters:
    - age_months (int/float): Usia dalam bulan
    - weight_kg (float): Berat badan dalam kg
    - height_cm (float): Tinggi badan dalam cm
    - gender (str): Jenis kelamin ('Laki-laki' atau 'Perempuan')

    Returns:
    - dict: Berisi status prediksi, confidence, detail perhitungan, dll.
    """
    try:
        # Cek keberadaan file model dan encoder
        if not os.path.exists(MODEL_PATH):
            return {
                "success": False,
                "message": "File model (Prediksi_Stunting.pkl) tidak ditemukan di folder models/.",
            }

        if not os.path.exists(LE_PATH):
            return {
                "success": False,
                "message": "File encoder (LabelEncoder_Gender.pkl) tidak ditemukan di folder models/.",
            }

        # Load model dan label encoder
        model = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)

        # Encode jenis kelamin (sesuai dengan data training)
        try:
            gender_encoded = le.transform([gender])[0]
        except ValueError:
            return {
                "success": False,
                "message": f"Jenis kelamin '{gender}' tidak valid. Gunakan 'Laki-laki' atau 'Perempuan'.",
            }

        # 4. Feature Engineering (menyesuaikan colab)
        who_median_height = 50 + age_months * 0.7
        who_sd_height = 3.0 + age_months * 0.03

        z_score_tinggi = (height_cm - who_median_height) / who_sd_height
        bmi = weight_kg / ((height_cm / 100) ** 2)
        tinggi_per_usia = height_cm / (age_months + 1)

        # Dataframe
        input_data = pd.DataFrame(
            {
                "Jenis Kelamin Encoded": [gender_encoded],
                "Umur (bulan)": [age_months],
                "Berat Badan (kg)": [weight_kg],
                "Tinggi Badan (cm)": [height_cm],
                "Z-Score Tinggi": [z_score_tinggi],
                "BMI": [bmi],
                "Tinggi per Usia": [tinggi_per_usia],
            }
        )

        # Lakukan prediksi
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(probabilities.max())

        # Kembalikan hasil dalam format dictionary (sesuai kebutuhan app.py)
        return {
            "success": True,
            "status": str(prediction),
            "confidence": confidence,
            "details": {
                "bmi": round(bmi, 2),
                "z_score": round(z_score_tinggi, 2),
                "tinggi_per_usia": round(tinggi_per_usia, 2),
                "usia": age_months,
                "berat": weight_kg,
                "tinggi": height_cm,
                "gender": gender,
            },
            "message": f"Status: {prediction}\nConfidence: {confidence:.2%}",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Terjadi kesalahan saat memprediksi: {str(e)}",
        }
