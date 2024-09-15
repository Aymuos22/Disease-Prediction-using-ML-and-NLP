from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('disease_prediction_model.pkl')

# Extract correct feature names from the model (if available)
try:
    correct_feature_names = model.feature_names_in_
except AttributeError:
    # Manually specify or handle missing feature names if not directly available
    correct_feature_names = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 
        'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
        'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
        'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
        'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
        'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
        'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
        'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
        'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
        'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
        'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
        'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
        'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
        'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
        'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
        'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
        'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising',
        'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
        'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
        'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
        'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
        'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
        'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
        'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
        'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
        'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
        'altered_sensorium', 'red_spots_over_body', 'belly_pain',
        'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
        'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
        'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
        'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
        'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
        'fluid_overload.1',  # Included as seen during training
        'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
        'painful_walking', 'pus_filled_pimples', 'blackheads',
        'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
        'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
    ]

# Full mapping of numerical codes to prognosis labels (ensure all mappings are complete)
prognosis_mapping = {
    0: 'Fungal infection', 1: 'Allergy', 2: 'GERD', 3: 'Chronic cholestasis',
    4: 'Drug Reaction', 5: 'Peptic ulcer disease', 6: 'AIDS', 7: 'Diabetes',
    8: 'Gastroenteritis', 9: 'Bronchial Asthma', 10: 'Hypertension', 11: 'Migraine',
    12: 'Cervical spondylosis', 13: 'Paralysis (brain hemorrhage)', 14: 'Jaundice',
    15: 'Malaria', 16: 'Chicken pox', 17: 'Dengue', 18: 'Typhoid', 19: 'Hepatitis A',
    20: 'Hepatitis B', 21: 'Hepatitis C', 22: 'Hepatitis D', 23: 'Hepatitis E',
    24: 'Alcoholic hepatitis', 25: 'Tuberculosis', 26: 'Common Cold',
    27: 'Pneumonia', 28: 'Dimorphic hemmorhoids(piles)', 29: 'Heart attack',
    30: 'Varicose veins', 31: 'Hypothyroidism', 32: 'Hyperthyroidism',
    33: 'Hypoglycemia', 34: 'Osteoarthristis', 35: 'Arthritis', 36: 'Vertigo',
    37: 'Acne', 38: 'Urinary tract infection', 39: 'Psoriasis', 40: 'Impetigo',
    41: 'Hyperlipidemia', 42: 'Influenza', 43: 'Metabolic syndrome', 44: 'Obesity',
    45: 'Pulmonary embolism', 46: 'Stroke', 47: 'Tuberculosis', 48: 'Cystic fibrosis',
    49: 'Emphysema', 50: 'Chronic bronchitis', 51: 'Asthma', 52: 'COPD', 53: 'Lung cancer',
    54: 'Mesothelioma', 55: 'Bronchiolitis', 56: 'Lung abscess', 57: 'Pleurisy',
    58: 'Pneumothorax', 59: 'Pulmonary hypertension', 60: 'Sleep apnea', 
    61: 'Respiratory syncytial virus', 62: 'Whooping cough', 63: 'Acute bronchitis', 
    64: 'Chronic kidney disease', 65: 'Acute kidney injury', 66: 'Kidney stones',
    67: 'Polycystic kidney disease', 68: 'Pyelonephritis', 69: 'Glomerulonephritis', 
    70: 'Interstitial nephritis', 71: 'Renal tubular acidosis', 72: 'Nephrotic syndrome',
    73: 'Diabetic nephropathy', 74: 'Lupus nephritis', 75: 'IgA nephropathy', 
    76: 'Nephrocalcinosis', 77: 'Renal artery stenosis', 78: 'Renal vein thrombosis', 
    79: 'Anemia', 80: 'Thalassemia', 81: 'Sickle cell anemia', 82: 'Iron deficiency anemia', 
    83: 'Vitamin B12 deficiency anemia', 84: 'Folate deficiency anemia', 
    85: 'Aplastic anemia', 86: 'Hemolytic anemia', 87: 'Myelodysplastic syndromes', 
    88: 'Polycythemia vera', 89: 'Leukemia', 90: 'Lymphoma', 91: 'Multiple myeloma', 
    92: 'Hodgkin lymphoma', 93: 'Non-Hodgkin lymphoma', 94: 'Chronic lymphocytic leukemia', 
    95: 'Acute lymphoblastic leukemia', 96: 'Chronic myeloid leukemia', 
    97: 'Acute myeloid leukemia', 98: 'Burkitt lymphoma', 99: 'Waldenstrom macroglobulinemia',
    100: 'Ankylosing spondylitis', 101: 'Osteoporosis', 102: 'Paget disease of bone',
    103: 'Osteogenesis imperfecta', 104: 'Rickets', 105: 'Hyperparathyroidism',
    106: 'Hypoparathyroidism', 107: 'Cushing syndrome', 108: 'Addison disease',
    109: 'Graves disease', 110: 'Hashimoto thyroiditis', 111: 'Prostatitis', 
    112: 'Prostate cancer', 113: 'Benign prostatic hyperplasia', 114: 'Testicular cancer',
    115: 'Male breast cancer', 116: 'Penile cancer', 117: 'Ovarian cancer', 
    118: 'Uterine cancer', 119: 'Cervical cancer', 120: 'Endometrial cancer', 
    121: 'Vaginal cancer', 122: 'Vulvar cancer', 123: 'Breast cancer', 124: 'Thyroid cancer',
    125: 'Parathyroid cancer', 126: 'Adrenal cancer', 127: 'Pituitary tumors',
    128: 'Pancreatic cancer', 129: 'Colorectal cancer', 130: 'Esophageal cancer', 
    131: 'Stomach cancer', 132: 'Liver cancer', 133: 'Gallbladder cancer', 
    134: 'Bladder cancer'
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', symptoms=correct_feature_names, selected_symptoms=[])

@app.route('/add_symptom', methods=['POST'])
def add_symptom():
    selected_symptoms = request.form.getlist('selected_symptoms')
    new_symptom = request.form.get('symptom')
    if new_symptom and new_symptom not in selected_symptoms:
        selected_symptoms.append(new_symptom)
    return render_template('index.html', symptoms=correct_feature_names, selected_symptoms=selected_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('selected_symptoms')
    input_features = [1 if symptom in selected_symptoms else 0 for symptom in correct_feature_names]

    # Prepare the input DataFrame for the model
    input_df = pd.DataFrame([input_features], columns=correct_feature_names)

    # Make prediction using the model
    prediction = model.predict(input_df)
    predicted_prognosis = prognosis_mapping.get(prediction[0], "Unknown Prognosis")

    return render_template('index.html', symptoms=correct_feature_names, selected_symptoms=selected_symptoms, prediction_text=f'Predicted Disease: {predicted_prognosis}')

if __name__ == "__main__":
    app.run(debug=True)
