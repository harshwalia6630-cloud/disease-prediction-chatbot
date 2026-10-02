"""Canonical symptom vocabulary and the lay-language phrases that map onto it.

Canonical names come from the 131 symptoms in the Kaggle disease-symptom
dataset (underscores removed, spelling fixed). Synonyms cover the way people
actually describe symptoms in chat ("throwing up", "tummy ache", "can't sleep").
"""

# raw Kaggle column value -> canonical display name (only where cleaning is needed)
RAW_FIXES = {
    "dischromic__patches": "discolored patches",
    "spotting__urination": "spotting urination",
    "toxic_look_(typhos)": "toxic look",
    "cold_hands_and_feets": "cold hands and feet",
    "swollen_extremeties": "swollen extremities",
    "scurring": "scarring",
    "burning_micturition": "burning urination",
    "diarrhoea": "diarrhea",
    "fluid_overload": "fluid overload",
    "foul_smell_of urine": "foul smell of urine",
}


def canonical(raw):
    raw = raw.strip()
    return RAW_FIXES.get(raw, raw.replace("_", " ").replace("  ", " ").strip())


# lay phrase -> canonical symptom
SYNONYMS = {
    # gastrointestinal
    "throwing up": "vomiting", "throw up": "vomiting", "threw up": "vomiting", "puking": "vomiting",
    "vomit": "vomiting", "vomited": "vomiting",
    "feel sick to my stomach": "nausea", "nauseous": "nausea", "queasy": "nausea", "feel like vomiting": "nausea",
    "tummy ache": "stomach pain", "stomach ache": "stomach pain", "stomachache": "stomach pain",
    "belly ache": "belly pain", "pain in my stomach": "stomach pain", "pain in my belly": "belly pain",
    "pain in my abdomen": "abdominal pain", "abdomen hurts": "abdominal pain", "cramping in my stomach": "cramps",
    "loose motions": "diarrhea", "loose stools": "diarrhea", "watery stools": "diarrhea", "runny stool": "diarrhea",
    "the runs": "diarrhea", "diarrhoea": "diarrhea",
    "can't poop": "constipation", "hard stools": "constipation", "constipated": "constipation",
    "heartburn": "acidity", "acid reflux": "acidity", "burning in my chest": "acidity", "sour taste": "acidity",
    "gas": "passage of gases", "bloated": "distention of abdomen", "bloating": "distention of abdomen",
    "swollen belly": "swelling of stomach", "not hungry": "loss of appetite", "no appetite": "loss of appetite",
    "don't feel like eating": "loss of appetite", "lost my appetite": "loss of appetite",
    "always hungry": "excessive hunger", "blood in my stool": "bloody stool", "blood in stool": "bloody stool",
    "pain when i poop": "pain during bowel movements", "itchy bottom": "irritation in anus",
    "upset stomach": "indigestion",
    # respiratory
    "short of breath": "breathlessness", "shortness of breath": "breathlessness", "can't breathe": "breathlessness",
    "difficulty breathing": "breathlessness", "trouble breathing": "breathlessness", "out of breath": "breathlessness",
    "wheezing": "breathlessness", "coughing": "cough", "coughed": "cough",
    "stuffy nose": "congestion", "blocked nose": "congestion", "nasal congestion": "congestion",
    "runny nose": "runny nose", "nose is running": "runny nose", "sneezing": "continuous sneezing",
    "sneeze": "continuous sneezing", "mucus": "phlegm", "sputum": "phlegm",
    "coughing up blood": "blood in sputum", "sore throat": "throat irritation", "scratchy throat": "throat irritation",
    "can't smell": "loss of smell", "lost my sense of smell": "loss of smell",
    # general / fever
    "fever": "high fever", "high temperature": "high fever", "temperature": "high fever", "feverish": "high fever",
    "low grade fever": "mild fever", "slight fever": "mild fever",
    "tired": "fatigue", "exhausted": "fatigue", "tiredness": "fatigue", "no energy": "fatigue", "worn out": "fatigue",
    "sluggish": "lethargy", "weak": "fatigue", "feel weak": "fatigue",
    "shivers": "shivering", "shaking": "shivering", "chilly": "chills", "sweat": "sweating", "sweaty": "sweating",
    "night sweats": "sweating", "lost weight": "weight loss", "losing weight": "weight loss",
    "gained weight": "weight gain", "putting on weight": "weight gain", "overweight": "obesity",
    "dehydrated": "dehydration", "feel unwell": "malaise", "general discomfort": "malaise",
    "swollen glands": "swelled lymph nodes", "swollen lymph nodes": "swelled lymph nodes",
    # head / neuro
    "head hurts": "headache", "head ache": "headache", "migraine": "headache", "pounding head": "headache",
    "dizzy": "dizziness", "lightheaded": "dizziness", "light headed": "dizziness",
    "room is spinning": "spinning movements", "vertigo": "spinning movements",
    "lose my balance": "loss of balance", "off balance": "loss of balance", "unsteady": "unsteadiness",
    "can't concentrate": "lack of concentration", "trouble concentrating": "lack of concentration",
    "brain fog": "lack of concentration", "blurry vision": "blurred and distorted vision",
    "blurred vision": "blurred and distorted vision", "vision is blurry": "blurred and distorted vision",
    "confused": "altered sensorium", "confusion": "altered sensorium",
    "slurring": "slurred speech", "one side of my body is weak": "weakness of one body side",
    "stiff neck": "stiff neck", "neck hurts": "neck pain", "pain in my neck": "neck pain",
    "anxious": "anxiety", "worried all the time": "anxiety", "sad": "depression", "depressed": "depression",
    "irritable": "irritability", "mood changes": "mood swings", "can't sleep": "restlessness",
    "trouble sleeping": "restlessness", "insomnia": "restlessness", "restless": "restlessness",
    # skin
    "rash": "skin rash", "rashes": "skin rash", "itchy": "itching", "itch": "itching", "itchiness": "itching",
    "red spots": "red spots over body", "spots on my body": "red spots over body",
    "pimples": "pus filled pimples", "zits": "pus filled pimples", "acne": "pus filled pimples",
    "blisters": "blister", "peeling skin": "skin peeling", "flaky skin": "skin peeling", "scaly": "skin peeling",
    "dry patches": "skin peeling", "silvery scales": "silver like dusting", "yellow skin": "yellowish skin",
    "skin is yellow": "yellowish skin", "yellow eyes": "yellowing of eyes", "eyes are yellow": "yellowing of eyes",
    "dark patches": "discolored patches", "discoloured patches": "discolored patches", "bruises": "bruising",
    "bruise easily": "bruising", "crusty sores": "yellow crust ooze", "sores around my nose": "red sore around nose",
    "scars": "scarring", "pitted nails": "small dents in nails", "brittle nails": "brittle nails",
    # eyes
    "red eyes": "redness of eyes", "watery eyes": "watering from eyes", "eyes are watering": "watering from eyes",
    "pain behind my eyes": "pain behind the eyes", "puffy eyes": "puffy face and eyes", "sunken eyes": "sunken eyes",
    # musculoskeletal
    "joints hurt": "joint pain", "joint ache": "joint pain", "achy joints": "joint pain", "painful joints": "joint pain",
    "swollen joints": "swelling joints", "muscle aches": "muscle pain", "body aches": "muscle pain",
    "body ache": "muscle pain", "sore muscles": "muscle pain", "back hurts": "back pain", "backache": "back pain",
    "knees hurt": "knee pain", "hip pain": "hip joint pain", "stiff joints": "movement stiffness",
    "stiffness": "movement stiffness", "hurts to walk": "painful walking", "weak muscles": "muscle weakness",
    "arms and legs feel weak": "weakness in limbs", "limbs feel weak": "weakness in limbs",
    "swollen legs": "swollen legs", "swollen feet": "swollen extremities", "swollen ankles": "swollen extremities",
    "veins are visible": "prominent veins on calf", "bulging veins": "swollen blood vessels",
    "cramp": "cramps", "cramping": "cramps",
    # urinary
    "burning when i pee": "burning urination", "burning while urinating": "burning urination",
    "painful urination": "burning urination", "pain when urinating": "burning urination",
    "pee a lot": "polyuria", "frequent urination": "polyuria", "urinating frequently": "polyuria",
    "always need to pee": "continuous feel of urine", "urge to urinate": "continuous feel of urine",
    "smelly urine": "foul smell of urine", "dark pee": "dark urine", "yellow pee": "yellow urine",
    "bladder pain": "bladder discomfort",
    # cardio / metabolic
    "chest hurts": "chest pain", "pain in my chest": "chest pain", "heart is racing": "fast heart rate",
    "racing heart": "fast heart rate", "heart pounding": "palpitations", "irregular heartbeat": "palpitations",
    "always thirsty": "dehydration", "very thirsty": "dehydration", "excessive thirst": "dehydration",
    "blood sugar": "irregular sugar level", "sugar levels": "irregular sugar level",
    "cold hands": "cold hands and feet", "cold feet": "cold hands and feet", "goiter": "enlarged thyroid",
    "irregular periods": "abnormal menstruation", "dry lips": "drying and tingling lips",
    "tingling lips": "drying and tingling lips",
}

NEGATIONS = {"no", "not", "never", "without", "dont", "don't", "didnt", "didn't", "havent", "haven't",
             "hasnt", "hasn't", "isnt", "isn't", "nor", "denies", "free"}
