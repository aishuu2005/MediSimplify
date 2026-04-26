"""
medical_engine.py - MediSimplify
==================================
Rule-based Medical Knowledge Engine.

This module contains:
  1. NORMAL_RANGES  – a dictionary mapping common test names to their
                      reference intervals (low, high, unit, plain-English label).
  2. JARGON_GLOSSARY – medical term → simple explanation lookup.
  3. analyse_result() – core triage function that classifies a value as
                        LOW / NORMAL / HIGH and returns actionable advice.
  4. translate_jargon() – looks up a term in the glossary.

Design note: A rule-based dictionary was chosen over a machine-learning model
because it is fully explainable, auditable, and appropriate for an academic MVP
where patient safety is paramount.  All reference ranges are based on standard
clinical guidelines (ICMR / WHO) and are for educational purposes only.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Normal Ranges Dictionary
# ---------------------------------------------------------------------------
# Structure:
#   "Canonical Test Name": {
#       "low":    lower bound of normal range (inclusive),
#       "high":   upper bound of normal range (inclusive),
#       "unit":   expected unit string,
#       "label":  human-readable display name,
#       "low_advice":  plain-English advice when value < low,
#       "high_advice": plain-English advice when value > high,
#   }
#
# Keys are stored in UPPER CASE so matching is case-insensitive.
# ---------------------------------------------------------------------------

NORMAL_RANGES: dict[str, dict] = {
    # ── Complete Blood Count (CBC) ───────────────────────────────────────
    "HEMOGLOBIN": {
        "low": 12.0, "high": 17.5, "unit": "g/dL",
        "label": "Hemoglobin",
        "low_advice": (
            "Low Hemoglobin (Anemia): Your hemoglobin is below the normal range. "
            "This may cause fatigue and breathlessness. "
            "Consider consulting a General Physician or Hematologist. "
            "Increase iron-rich foods (spinach, lentils, red meat) in your diet."
        ),
        "high_advice": (
            "High Hemoglobin (Polycythemia): Your hemoglobin is above normal. "
            "This can thicken the blood. "
            "Please consult a Hematologist for further evaluation."
        ),
    },
    "WBC": {
        "low": 4.0, "high": 11.0, "unit": "×10³/µL",
        "label": "White Blood Cell Count",
        "low_advice": (
            "Low WBC (Leukopenia): A low white cell count may indicate a weakened immune system "
            "or bone marrow issues. Consult a Hematologist."
        ),
        "high_advice": (
            "High WBC (Leukocytosis): Elevated white cells often indicate infection or inflammation. "
            "Consult a General Physician promptly."
        ),
    },
    "PLATELETS": {
        "low": 150.0, "high": 400.0, "unit": "×10³/µL",
        "label": "Platelet Count",
        "low_advice": (
            "Low Platelets (Thrombocytopenia): Risk of prolonged bleeding. "
            "Avoid NSAIDs. Consult a Hematologist immediately."
        ),
        "high_advice": (
            "High Platelets (Thrombocytosis): Risk of clot formation. "
            "Consult a Hematologist or Cardiologist."
        ),
    },
    "RBC": {
        "low": 4.2, "high": 5.9, "unit": "×10⁶/µL",
        "label": "Red Blood Cell Count",
        "low_advice": "Low RBC count may indicate anemia. Consult a General Physician.",
        "high_advice": "High RBC count may indicate dehydration or polycythemia. Consult a Physician.",
    },
    "HEMATOCRIT": {
        "low": 36.0, "high": 52.0, "unit": "%",
        "label": "Hematocrit (PCV)",
        "low_advice": "Low hematocrit suggests anemia or blood loss. Consult a General Physician.",
        "high_advice": "High hematocrit may indicate dehydration or polycythemia. Stay hydrated and consult a Doctor.",
    },

    # ── Blood Sugar (Glucose) ────────────────────────────────────────────
    "FASTING BLOOD SUGAR": {
        "low": 70.0, "high": 100.0, "unit": "mg/dL",
        "label": "Fasting Blood Sugar (FBS)",
        "low_advice": (
            "Low Fasting Blood Sugar (Hypoglycemia): Immediately consume fast-acting carbohydrates "
            "(juice, glucose). Monitor regularly and consult an Endocrinologist."
        ),
        "high_advice": (
            "High Fasting Blood Sugar: Values above 100 mg/dL may indicate Pre-Diabetes (100–125) "
            "or Diabetes (≥126). Consider consulting an Endocrinologist. "
            "Reduce refined sugars and increase physical activity."
        ),
    },
    "POSTPRANDIAL BLOOD SUGAR": {
        "low": 70.0, "high": 140.0, "unit": "mg/dL",
        "label": "Post-Prandial Blood Sugar (PPBS)",
        "low_advice": "Low post-meal sugar. Monitor for hypoglycemia. Consult a Doctor.",
        "high_advice": (
            "High post-meal blood sugar may indicate Diabetes or Insulin Resistance. "
            "Consult an Endocrinologist and review your diet."
        ),
    },
    "HBA1C": {
        "low": 4.0, "high": 5.7, "unit": "%",
        "label": "Glycated Hemoglobin (HbA1c)",
        "low_advice": "Unusually low HbA1c – discuss with your Doctor.",
        "high_advice": (
            "Elevated HbA1c: 5.7–6.4% indicates Pre-Diabetes; ≥6.5% indicates Diabetes. "
            "Consult an Endocrinologist for a personalised management plan."
        ),
    },

    # ── Lipid Profile ────────────────────────────────────────────────────
    "TOTAL CHOLESTEROL": {
        "low": 0.0, "high": 200.0, "unit": "mg/dL",
        "label": "Total Cholesterol",
        "low_advice": "Very low cholesterol is rare; consult a Doctor.",
        "high_advice": (
            "High Total Cholesterol: Risk factor for cardiovascular disease. "
            "Adopt a low-fat diet, exercise regularly, and consult a Cardiologist."
        ),
    },
    "LDL": {
        "low": 0.0, "high": 100.0, "unit": "mg/dL",
        "label": "LDL Cholesterol (Bad)",
        "low_advice": "LDL is within healthy range (lower is better for LDL).",
        "high_advice": (
            "High LDL Cholesterol: Increases risk of heart disease and stroke. "
            "Consult a Cardiologist. Dietary changes and statins may be recommended."
        ),
    },
    "HDL": {
        "low": 40.0, "high": 999.0, "unit": "mg/dL",
        "label": "HDL Cholesterol (Good)",
        "low_advice": (
            "Low HDL Cholesterol: Increases cardiovascular risk. "
            "Regular aerobic exercise and a Mediterranean diet can raise HDL. "
            "Consult a Cardiologist."
        ),
        "high_advice": "High HDL is generally protective. No action required.",
    },
    "TRIGLYCERIDES": {
        "low": 0.0, "high": 150.0, "unit": "mg/dL",
        "label": "Triglycerides",
        "low_advice": "Very low triglycerides – no concern.",
        "high_advice": (
            "High Triglycerides: Risk factor for pancreatitis and heart disease. "
            "Reduce sugar, alcohol, and refined carbs. Consult a Cardiologist."
        ),
    },

    # ── Kidney Function ──────────────────────────────────────────────────
    "CREATININE": {
        "low": 0.6, "high": 1.2, "unit": "mg/dL",
        "label": "Serum Creatinine",
        "low_advice": "Low creatinine may indicate low muscle mass. Consult a Nephrologist if concerned.",
        "high_advice": (
            "High Creatinine: Indicates possible kidney stress or disease. "
            "Stay well-hydrated. Consult a Nephrologist promptly."
        ),
    },
    "UREA": {
        "low": 7.0, "high": 20.0, "unit": "mg/dL",
        "label": "Blood Urea Nitrogen (BUN)",
        "low_advice": "Low BUN may be due to malnutrition. Consult a Doctor.",
        "high_advice": (
            "High BUN: May indicate dehydration or kidney dysfunction. "
            "Increase water intake and consult a Nephrologist."
        ),
    },

    # ── Liver Function ───────────────────────────────────────────────────
    "SGPT": {
        "low": 0.0, "high": 40.0, "unit": "U/L",
        "label": "SGPT / ALT (Liver Enzyme)",
        "low_advice": "Normal low SGPT – no action needed.",
        "high_advice": (
            "High SGPT/ALT: Liver may be inflamed or damaged. "
            "Avoid alcohol and fatty foods. Consult a Gastroenterologist or Hepatologist."
        ),
    },
    "SGOT": {
        "low": 0.0, "high": 40.0, "unit": "U/L",
        "label": "SGOT / AST (Liver Enzyme)",
        "low_advice": "Normal low SGOT – no action needed.",
        "high_advice": (
            "High SGOT/AST: Can indicate liver or heart muscle injury. "
            "Consult a General Physician for further investigation."
        ),
    },
    "BILIRUBIN": {
        "low": 0.0, "high": 1.2, "unit": "mg/dL",
        "label": "Total Bilirubin",
        "low_advice": "Low bilirubin is normal.",
        "high_advice": (
            "High Bilirubin (Jaundice risk): May indicate liver disease, bile duct obstruction, "
            "or haemolysis. Consult a Gastroenterologist promptly."
        ),
    },

    # ── Thyroid ──────────────────────────────────────────────────────────
    "TSH": {
        "low": 0.4, "high": 4.0, "unit": "mIU/L",
        "label": "Thyroid Stimulating Hormone (TSH)",
        "low_advice": (
            "Low TSH (Hyperthyroidism risk): May cause rapid heart rate, weight loss, anxiety. "
            "Consult an Endocrinologist."
        ),
        "high_advice": (
            "High TSH (Hypothyroidism risk): May cause fatigue, weight gain, cold intolerance. "
            "Consult an Endocrinologist for thyroid evaluation."
        ),
    },

    # ── Vitamins & Minerals ──────────────────────────────────────────────
    "VITAMIN D": {
        "low": 20.0, "high": 100.0, "unit": "ng/mL",
        "label": "Vitamin D (25-OH)",
        "low_advice": (
            "Vitamin D Deficiency: Can cause bone weakness (osteoporosis), fatigue, and low immunity. "
            "Increase sun exposure and consider supplements after consulting a Doctor."
        ),
        "high_advice": (
            "Vitamin D Toxicity: Excess Vitamin D can cause hypercalcemia. "
            "Stop supplements and consult a Doctor."
        ),
    },
    "VITAMIN B12": {
        "low": 200.0, "high": 900.0, "unit": "pg/mL",
        "label": "Vitamin B12",
        "low_advice": (
            "Low Vitamin B12: Can cause nerve damage, fatigue, and megaloblastic anemia. "
            "B12 injections or supplements recommended. Consult a General Physician."
        ),
        "high_advice": "High B12 – usually from supplements. Consult a Doctor if not supplementing.",
    },
    "IRON": {
        "low": 60.0, "high": 170.0, "unit": "µg/dL",
        "label": "Serum Iron",
        "low_advice": (
            "Low Serum Iron (Iron Deficiency): May lead to anemia. "
            "Eat iron-rich foods (green leafy vegetables, pulses). Consult a General Physician."
        ),
        "high_advice": (
            "High Serum Iron (Iron Overload/Hemochromatosis risk): "
            "Consult a Hematologist for evaluation."
        ),
    },

    # ── Blood Pressure (stored as systolic for tracking purposes) ────────
    "SYSTOLIC BP": {
        "low": 90.0, "high": 120.0, "unit": "mmHg",
        "label": "Systolic Blood Pressure",
        "low_advice": (
            "Low Systolic BP (Hypotension): May cause dizziness. "
            "Increase fluid and salt intake cautiously. Consult a Cardiologist."
        ),
        "high_advice": (
            "High Systolic BP (Hypertension): Long-term risk for heart disease and stroke. "
            "Reduce salt, exercise regularly, and consult a Cardiologist."
        ),
    },
    "DIASTOLIC BP": {
        "low": 60.0, "high": 80.0, "unit": "mmHg",
        "label": "Diastolic Blood Pressure",
        "low_advice": "Low diastolic BP – monitor and consult a Doctor if symptomatic.",
        "high_advice": (
            "High Diastolic BP: Persistent elevation increases cardiac risk. "
            "Consult a Cardiologist."
        ),
    },
}


# ---------------------------------------------------------------------------
# 2. Multilingual Medical Advice Dictionary
# ---------------------------------------------------------------------------
# Structure:
#   "CATEGORY": {
#       "english": "English advice text",
#       "hindi": "Hindi translation",
#       "tamil": "Tamil translation",
#       "telugu": "Telugu translation",
#       "kannada": "Kannada translation",
#       "malayalam": "Malayalam translation"
#   }
# ---------------------------------------------------------------------------

MULTILINGUAL_ADVICE: dict[str, dict[str, str]] = {
    "DIABETIC": {
        "english": "🔴 Diabetes detected. Consult an Endocrinologist immediately for diagnosis and management. Lifestyle changes and possibly medication may be required.",
        "hindi": "🔴 मधुमेह का पता चला। निदान और प्रबंधन के लिए तुरंत एंडोक्राइनोलॉजिस्ट से सलाह लें। जीवनशैली में बदलाव और संभवतः दवा की आवश्यकता हो सकती है।",
        "tamil": "🔴 நீரிழிவு நோய் கண்டறியப்பட்டது. நோய் கண்டறிதல் மற்றும் நிர்வாகத்திற்கு உடனடியாக ஒரு மருத்துவரிடம் ஆலோசனை பெறுங்கள். வாழ்க்கை முறை மாற்றங்கள் மற்றும் சாத்தியமான மருந்து தேவைப்படலாம்.",
        "telugu": "🔴 డయాబెటిస్ కనుగొనబడింది. నిర్ధారణ మరియు నిర్వహణ కోసం వెంటనే ఎండోక్రైనాలజిస్ట్ ను సంప్రదించండి. జీవనశైలి మార్పులు మరియు సాధ్యమైన మందులు అవసరం కావచ్చు.",
        "kannada": "🔴 ಡಯಬೀಟಿಸ್ ಪತ್ತೆಯಾಗಿದೆ. ರೋಗನಿರ್ಣಯ ಮತ್ತು ನಿರ್ವಹಣೆಗಾಗಿ ತಕ್ಷಣವೇ ಎಂಡೋಕ್ರೈನಾಲಜಿಸ್ಟ್ ಅನ್ನು ಸಂಪರ್ಕಿಸಿ. ಜೀವನಶೈಲಿ ಬದಲಾವಣೆಗಳು ಮತ್ತು ಸಾಧ್ಯವಾದ ಔಷಧಿಗಳ ಅಗತ್ಯವಿರಬಹುದು.",
        "malayalam": "🔴 പ്രമേഹം കണ്ടെത്തി. രോഗനിർണയത്തിനും മാനേജ്മെന്റിനും ഉടനെ ഒരു എൻഡോക്രൈനോളജിസ്റ്റിനെ സമീപിക്കുക. ജീവിതശൈലി മാറ്റങ്ങളും സാധ്യമായ മരുന്നുകളും ആവശ്യമായേക്കാം."
    },
    "PRE_DIABETIC": {
        "english": "⚠️ Pre-Diabetes detected. This is a warning sign. Consult an Endocrinologist. Focus on diet, exercise, and weight management to prevent progression to Diabetes.",
        "hindi": "⚠️ पूर्व-मधुमेह का पता चला। यह एक चेतावनी संकेत है। एंडोक्राइनोलॉजिस्ट से सलाह लें। मधुमेह में प्रगति को रोकने के लिए आहार, व्यायाम और वजन प्रबंधन पर ध्यान दें।",
        "tamil": "⚠️ முன்-நீரிழிவு நோய் கண்டறியப்பட்டது. இது ஒரு எச்சரிக்கை அடையாளம். ஒரு மருத்துவரிடம் ஆலோசனை பெறுங்கள். நீரிழிவு நோய் முன்னேற்றத்தைத் தடுக்க கடைபிடிக்கும் உணவு, உடற்பயிற்சி மற்றும் எடை நிர்வாகம்.",
        "telugu": "⚠️ ప్రీ-డయాబెటిస్ కనుగొనబడింది. ఇది ఒక హెచ్చరిక సంకేతం. ఎండోక్రైనాలజిస్ట్ ను సంప్రదించండి. డయాబెటిస్ ప్రోగ్రెషన్ ను నిరోధించడానికి ఆహారం, వ్యాయామం మరియు బరువు నిర్వహణపై దృష్టి పెట్టండి.",
        "kannada": "⚠️ ಪ್ರೀ-ಡಯಬೀಟಿಸ್ ಪತ್ತೆಯಾಗಿದೆ. ಇದು ಒಂದು ಎಚ್ಚರಿಕೆ ಸಂಕೇತ. ಎಂಡೋಕ್ರೈನಾಲಜಿಸ್ಟ್ ಅನ್ನು ಸಂಪರ್ಕಿಸಿ. ಡಯಬೀಟಿಸ್ ಮುನ್ನಡೆಯನ್ನು ತಡೆಯಲು ಆಹಾರ, ವ್ಯಾಯಾಮ ಮತ್ತು ತೂಕ ನಿರ್ವಹಣೆಯ ಮೇಲೆ ಕೇಂದ್ರೀಕರಿಸಿ.",
        "malayalam": "⚠️ പ്രീ-ഡയബെറ്റിസ് കണ്ടെത്തി. ഇത് ഒരു മുന്നറിയിപ്പ് ചിഹ്നമാണ്. ഒരു എൻഡോക്രൈനോളജിസ്റ്റിനെ സമീപിക്കുക. പ്രമേഹത്തിലേക്കുള്ള പുരോഗതി തടയുന്നതിന് ഭക്ഷണം, വ്യായാമം, ഭാരനിയന്ത്രണം എന്നിവയിൽ ശ്രദ്ധ കേന്ദ്രീകരിക്കുക."
    },
    "HIGH_CHOLESTEROL": {
        "english": "🟡 High Cholesterol detected. Consult a Cardiologist. Reduce saturated fats, increase fiber intake, and consider regular exercise.",
        "hindi": "🟡 उच्च कोलेस्ट्रॉल का पता चला। कार्डियोलॉजिस्ट से सलाह लें। संतृप्त वसा कम करें, फाइबर का सेवन बढ़ाएं, और नियमित व्यायाम पर विचार करें।",
        "tamil": "🟡 அதிக கொலஸ்ட்ரால் கண்டறியப்பட்டது. ஒரு இதய நல மருத்துவரிடம் ஆலோசனை பெறுங்கள். பூரண கொழுப்புகளைக் குறைக்கவும், நார்ப்பொருள் உட்கொள்ளலை அதிகரிக்கவும், தொடர்ந்து உடற்பயிற்சியை கருதுங்கள்.",
        "telugu": "🟡 అధిక కొలెస్ట్రాల్ కనుగొనబడింది. కార్డియాలజిస్ట్ ను సంప్రదించండి. సంతృప్త కొవ్వులను తగ్గించండి, ఫైబర్ ఇంటేక్ ను పెంచండి, మరియు నియమిత వ్యాయామాన్ని పరిగణించండి.",
        "kannada": "🟡 ಹೆಚ್ಚಿನ ಕೊಲೆಸ್ಟ್ರಾಲ್ ಪತ್ತೆಯಾಗಿದೆ. ಕಾರ್ಡಿಯಾಲಜಿಸ್ಟ್ ಅನ್ನು ಸಂಪರ್ಕಿಸಿ. ಸಂತೃಪ್ತ ಕೊಜ್ಜುಗಳನ್ನು ಕಡಿಮೆ ಮಾಡಿ, ಫೈಬರ್ ಸೇವನೆಯನ್ನು ಹೆಚ್ಚಿಸಿ ಮತ್ತು ನಿಯಮಿತ ವ್ಯಾಯಾಮವನ್ನು ಪರಿಗಣಿಸಿ.",
        "malayalam": "🟡 ഉയർന്ന കൊളസ്ട്രോൾ കണ്ടെത്തി. ഒരു ഹൃദയവിദഗ്ദ്ധനെ സമീപിക്കുക. സംതൃപ്ത മേദയം കുറയ്ക്കുക, ഫൈബർ ഇൻടേക്ക് വർദ്ധിപ്പിക്കുക, നിത്യ വ്യായാമം പരിഗണിക്കുക."
    },
    "NORMAL": {
        "english": "✅ All values are within normal ranges. No immediate action required. Continue healthy habits.",
        "hindi": "✅ सभी मान सामान्य सीमा के भीतर हैं। कोई तत्काल कार्रवाई की आवश्यकता नहीं। स्वस्थ आदतों को जारी रखें।",
        "tamil": "✅ எல்லா மதிப்புகளும் இயல்பான வரம்புகளுக்குள் உள்ளன. உடனடி நடவடிக்கை தேவையில்லை. ஆரோக்கியமான பழக்கங்களை தொடருங்கள்.",
        "telugu": "✅ అన్ని విలువలు సాధారణ పరిధులలో ఉన్నాయి. ఎటువంటి తక్షణ చర్య అవసరం లేదు. ఆరోగ్యకరమైన అలవాట్లను కొనసాగించండి.",
        "kannada": "✅ ಎಲ್ಲಾ ಮೌಲ್ಯಗಳು ಸಾಮಾನ್ಯ ಮಿತಿಗಳೊಳಗಿವೆ. ಯಾವುದೇ ತತ್ಕ್ಷಣ ಕ್ರಿಯೆ ಅಗತ್ಯವಿಲ್ಲ. ಆರೋಗ್ಯಕರ ಅಭ್ಯಾಸಗಳನ್ನು ಮುಂದುವರಿಸಿ.",
        "malayalam": "✅ എല്ലാ മൂല്യങ്ങളും സാധാരണ പരിധികളിൽ ഉണ്ട്. ഉടനടിയുള്ള നടപടി ആവശ്യമില്ല. ആരോഗ്യകരമായ പതിവുകൾ തുടരുക."
    }
}

# ---------------------------------------------------------------------------
# 2. Medical Jargon Glossary
# ---------------------------------------------------------------------------

JARGON_GLOSSARY: dict[str, str] = {
    "hemoglobin":        "The protein in red blood cells that carries oxygen throughout your body.",
    "hematocrit":        "The percentage of your blood volume made up of red blood cells.",
    "leukocyte":         "Another name for white blood cells, which fight infections.",
    "erythrocyte":       "Another name for red blood cells, which carry oxygen.",
    "thrombocyte":       "Another name for platelets, which help your blood clot.",
    "creatinine":        "A waste product filtered by the kidneys; high levels suggest kidney stress.",
    "bilirubin":         "A yellow pigment from broken-down red blood cells, processed by the liver.",
    "sgpt":              "A liver enzyme (ALT); elevated levels indicate liver cell damage.",
    "sgot":              "A liver enzyme (AST); can be elevated in liver or heart muscle injury.",
    "tsh":               "Thyroid Stimulating Hormone – controls how active your thyroid gland is.",
    "ldl":               "Low-Density Lipoprotein – the 'bad' cholesterol that can clog arteries.",
    "hdl":               "High-Density Lipoprotein – the 'good' cholesterol that protects the heart.",
    "triglycerides":     "Fats in the blood; high levels increase heart disease risk.",
    "hba1c":             "Average blood sugar over 3 months; used to diagnose and monitor diabetes.",
    "polycythemia":      "A condition where your body produces too many red blood cells.",
    "leukopenia":        "Abnormally low white blood cell count, reducing infection-fighting ability.",
    "thrombocytopenia":  "Low platelet count; increases risk of bruising and bleeding.",
    "hypertension":      "Persistently high blood pressure, a major risk factor for heart disease.",
    "hypotension":       "Persistently low blood pressure, which can cause dizziness and fainting.",
    "anemia":            "A condition where the blood lacks enough healthy red blood cells.",
    "jaundice":          "Yellowing of the skin/eyes caused by excess bilirubin in the blood.",
    "hypothyroidism":    "An underactive thyroid gland, causing fatigue, weight gain, and coldness.",
    "hyperthyroidism":   "An overactive thyroid, causing weight loss, rapid heartbeat, and anxiety.",
    "endocrinologist":   "A specialist doctor who treats hormone-related conditions (diabetes, thyroid).",
    "nephrologist":      "A specialist doctor who treats kidney diseases.",
    "hematologist":      "A specialist doctor who treats blood disorders.",
    "cardiologist":      "A specialist doctor who treats heart and blood vessel diseases.",
    "gastroenterologist":"A specialist doctor who treats digestive system and liver diseases.",
}


# ---------------------------------------------------------------------------
# 3. Core Analysis Function
# ---------------------------------------------------------------------------

def analyse_result(test_name: str, value: float) -> str:
    """
    Classifies a lab test result and returns the status key.

    Parameters
    ----------
    test_name : str   – name of the test (case-insensitive)
    value     : float – numeric result

    Returns
    -------
    str – status key: "NORMAL", "LOW", "HIGH", "PRE_DIABETIC", "DIABETIC", "UNKNOWN"
    """
    key = test_name.strip().upper()

    # Search for a matching key (exact match first, then partial match)
    matched_key = None
    if key in NORMAL_RANGES:
        matched_key = key
    else:
        for k in NORMAL_RANGES:
            if k in key or key in k:
                matched_key = k
                break

    if matched_key is None:
        return "UNKNOWN"

    ref   = NORMAL_RANGES[matched_key]
    low   = ref["low"]
    high  = ref["high"]

    # Special handling for blood sugar tests
    if matched_key == "HBA1C":
        if value >= 6.5:
            return "DIABETIC"
        elif value >= 5.7:
            return "PRE_DIABETIC"
        else:
            return "NORMAL"

    elif matched_key == "FASTING BLOOD SUGAR":
        if value >= 126:
            return "DIABETIC"
        elif value >= 100:
            return "PRE_DIABETIC"
        else:
            return "NORMAL"

    else:
        # Default logic for other tests
        if value < low:
            return "LOW"
        elif value > high:
            # Special handling for cholesterol
            if matched_key == "TOTAL CHOLESTEROL":
                return "HIGH_CHOLESTEROL"
            else:
                return "HIGH"
        else:
            return "NORMAL"


def get_advice(status: str, selected_lang: str = "english") -> str:
    """
    Get translated medical advice for a given status and language.

    Parameters
    ----------
    status : str – status key ("DIABETIC", "PRE_DIABETIC", "HIGH_CHOLESTEROL", "NORMAL")
    selected_lang : str – language code (default: "english")

    Returns
    -------
    str – translated advice text, defaults to English if translation missing
    """
    if status in MULTILINGUAL_ADVICE:
        return MULTILINGUAL_ADVICE[status].get(selected_lang.lower(),
                                               MULTILINGUAL_ADVICE[status]["english"])
    return "Translation not available."


def get_status_colour(status: str) -> str:
    """
    Get the hex colour code for a given status.

    Parameters
    ----------
    status : str – status key

    Returns
    -------
    str – hex colour code
    """
    colour_map = {
        "NORMAL": "#27ae60",      # green
        "LOW": "#e74c3c",         # red
        "HIGH": "#e67e22",        # orange
        "PRE_DIABETIC": "#f39c12", # yellow/orange
        "DIABETIC": "#e74c3c",    # red
        "HIGH_CHOLESTEROL": "#e67e22", # orange
        "UNKNOWN": "#6c757d",     # grey
    }
    return colour_map.get(status, "#6c757d")


def get_test_info(test_name: str) -> dict:
    """
    Get label and reference range information for a test.

    Parameters
    ----------
    test_name : str – name of the test

    Returns
    -------
    dict with keys:
        label : str – display-friendly test name
        range : str – normal range string
        unit  : str – unit of measurement
    """
    key = test_name.strip().upper()

    # Search for a matching key
    matched_key = None
    if key in NORMAL_RANGES:
        matched_key = key
    else:
        for k in NORMAL_RANGES:
            if k in key or key in k:
                matched_key = k
                break

    if matched_key is None:
        return {
            "label": test_name,
            "range": "N/A",
            "unit": "",
        }

    ref = NORMAL_RANGES[matched_key]
    low = ref["low"]
    high = ref["high"]
    unit = ref["unit"]
    label = ref["label"]

    # Special handling for blood sugar tests
    if matched_key == "HBA1C":
        return {
            "label": label,
            "range": "5.7–6.4% (Pre-diabetic), ≥6.5% (Diabetic)",
            "unit": unit,
        }
    elif matched_key == "FASTING BLOOD SUGAR":
        return {
            "label": label,
            "range": "<100 mg/dL (Normal), 100–125 mg/dL (Pre-diabetic), ≥126 mg/dL (Diabetic)",
            "unit": unit,
        }
    else:
        range_str = f"{low}–{high} {unit}" if low > 0 else f"< {high} {unit}"
        return {
            "label": label,
            "range": range_str,
            "unit": unit,
        }


def get_all_known_tests() -> list[str]:
    """Returns a sorted list of all test names in the reference database."""
    return sorted(NORMAL_RANGES.keys())


# ---------------------------------------------------------------------------
# 4. Jargon Translation Function
# ---------------------------------------------------------------------------

def translate_jargon(term: str) -> str:
    """
    Returns a plain-English explanation for a medical term.
    Case-insensitive look-up with partial match fallback.
    """
    key = term.strip().lower()

    if key in JARGON_GLOSSARY:
        return JARGON_GLOSSARY[key]

    # Partial match (e.g. "polycythemia vera" → matches "polycythemia")
    for glossary_term, explanation in JARGON_GLOSSARY.items():
        if glossary_term in key or key in glossary_term:
            return f"({glossary_term.title()}) {explanation}"

    return (
        f"Sorry, '{term}' is not in the MediSimplify glossary. "
        "Try rephrasing or consult a medical professional."
    )
