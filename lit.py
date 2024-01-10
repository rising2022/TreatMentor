import streamlit as st
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('all')
nltk.download('wordnet')
nltk.download('stopwords')
# Predicts diseases based on the symptoms entered and selected by the user.
# importing all necessary libraries
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean
from nltk.corpus import wordnet 
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from time import time
from collections import Counter
import operator
from xgboost import XGBClassifier
import math
from Treatment import diseaseDetail
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from Treatment import diseaseDetail
from googlesearch import search

warnings.simplefilter("ignore")
# Load the saved Logistic Regression model
file_path = 'C:/Users/deval/Downloads/Disease-Detection-based-on-Symptoms-master (3)/Disease-Detection-based-on-Symptoms-master/classifier.pkl'

import wikipediaapi

def get_disease_details(disease_name):
    user_agent = "Mayur/1.0 (mayur.22010670@viit.ac.in)"  # Replace with your user agent
    wiki_wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': user_agent})
    page_py = wiki_wiki.page(disease_name)
    if page_py.exists():
        st.write("Title:", page_py.title)
        st.write("Summary:", page_py.summary[:500])  # Print the first 500 characters of the summary
    else:
        st.write(f"Page for {disease_name} does not exist on Wikipedia.")

with open(file_path, 'rb') as file:
    logistic_regression_model = pickle.load(file)

df_comb = pd.read_csv("C:/Users/deval/Downloads/Disease-Detection-based-on-Symptoms-master (3)/Disease-Detection-based-on-Symptoms-master/Dataset/dis_sym_dataset_comb.csv")
df_norm = pd.read_csv("C:/Users/deval/Downloads/Disease-Detection-based-on-Symptoms-master (3)/Disease-Detection-based-on-Symptoms-master/Dataset/dis_sym_dataset_norm.csv") # Individual Disease

X = df_comb.iloc[:, 1:]
# X = X[]
Y = df_comb.iloc[:, 0:1]

# Streamlit UI
st.title('Disease Prediction')
st.write('Enter the following symptoms to predict disease:')


sym_input = st.text_input('Symptoms', value="Symptoms, separated, by, commas")

# Process the user input into a list of symptoms
user_symptoms = [sym.strip() for sym in sym_input.split(',')]
processed_user_symptoms = []
lemmatizer = WordNetLemmatizer()
splitter = WordPunctTokenizer()

for sym in user_symptoms:
    sym = sym.replace('-', ' ')
    sym = sym.replace("'", '')
    sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
    processed_user_symptoms.append(sym)
if st.button('Predict'):
    try:
        # Preprocess the input symptoms (similar to your preprocessing steps)
        processed_symptoms_str = ", ".join(processed_user_symptoms)
        st.write(f"Processed Symptoms: {processed_symptoms_str}")

        k=10
        st.write(f"\nTop {k} diseases predicted based on symptoms")
        
        diseases = ['Guillain-Barré syndrome', 'Lupus erythematosus', 'Epilepsy', 'Chronic fatigue syndrome', 'Iron Deficiency Anemia', 'Carbon monoxide poisoning', 'Scarlet fever', 'Dengue', 'Lice', 'Brucellosis', 'Chickenpox', 'Adult Inclusion Conjunctivitis', 'Preterm birth', 'Myocardial Infarction (Heart Attack)', 'SARS', 'Stroke', 'Muscular dystrophy', 'Alcohol Abuse and Alcoholism', 'Pulmonary embolism', 'Polycystic ovary syndrome (PCOS)', 'Breast Cancer / Carcinoma', 'Paratyphoid fever', 'Bubonic plague', "Parkinson's Disease", 'Scrub Typhus', 'Colitis', 'Chagas disease', 'Corneal Abrasion', 'Melanoma', 'Abscess', 'Myasthenia gravis', 'Puerperal sepsis', 'Keratoconjunctivitis Sicca (Dry eye syndrome)', 'Mouth Breathing', 'Nipah virus infection', 'Oral Cancer', 'Chronic obstructive pulmonary disease (COPD)', 'Asbestos-related diseases', 'Hypotonia', 'Strep throat', 'Diabetic Retinopathy', 'Appendicitis', 'Acute encephalitis syndrome', 'Fibroids', 'Congestive heart disease', 'Scabies', 'SIDS', 'Shin splints', 'Pelvic inflammatory disease', 'Keratoconus', 'Trichinosis', 'Cancer', 'Childhood Exotropia', 'Interstitial cystitis', 'Rocky Mountain spotted fever', 'Varicose Veins', 'Carpal Tunnel Syndrome', 'Iritis', 'Sarcoma', 'Mumps', 'Lymphoma', 'Insomnia', 'Lyme disease', 'Haemophilia', 'Aniseikonia', 'Chlamydia', 'Jaundice', 'Bad Breath (Halitosis)', 'Neonatal Respiratory Disease Syndrome(NRDS)', 'Rubella', 'Autism', 'Kala-azar/ Leishmaniasis', 'Quinsy', 'Cholera', 'Hepatitis', 'Nausea and Vomiting of Pregnancy and  Hyperemesis gravidarum', 'Inflammatory Bowel Disease', 'Kaposi’s Sarcoma', 'Alzheimer', 'Post Menopausal Bleeding', 'Amblyopia', 'Aseptic meningitis', 'Hand, Foot and Mouth Disease', 'Lead poisoning', 'Asthma', 'Vitiligo', 'Sciatica', 'Hepatitis B', 'Fibromyalgia', 'Thalassaemia', 'Hepatitis D', 'lactose intolerance', 'Leprosy', 'Leptospirosis', 'Malaria', 'Multiple sclerosis', 'Rift Valley fever', 'Stevens-Johnson syndrome', 'Tinnitus', 'Poliomyelitis', 'Dracunculiasis (guinea-worm disease)', 'Dysentery', 'Botulism', 'Trachoma', 'Cavities', 'Amaurosis Fugax', 'Herpes Simplex', 'Burns', 'Orbital Dermoid', 'Perennial Allergic Conjunctivitis', 'Necrotizing Fasciitis', 'Anaemia', 'Progeria', 'Nasal Polyps', 'Ear infection', 'Osteoarthritis', 'Trichomoniasis', 'Anxiety', 'Common cold', 'Cough', 'Bleeding Gums', 'Campylobacter infection', 'Migraine', 'Taeniasis/cysticercosis', 'Tonsillitis', 'Schizophrenia', 'Zika virus disease', 'Rheumatism', 'Dehydration', 'Candidiasis', 'Eclampsia', 'Rheumatic fever', 'Warkany syndrome', 'Diphtheria', 'Hypothyroid', 'Preeclampsia', "Bell's Palsy", 'Influenza', 'Laryngitis', 'Myelitis', "Raynaud's Phenomenon", 'Frost Bite', 'Antepartum hemorrhage (Bleeding in late pregnancy)', 'Tuberculosis', 'papilloedema', 'Endometriosis', 'Anisometropia', 'Postpartum depression/ Perinatal depression', 'Shaken Baby Syndrome', 'Multiple myeloma', 'Ques fever', 'Human papillomavirus', 'Osteomyelitis', 'Celiacs disease', 'Heat-Related Illnesses and Heat waves', 'GERD', 'Yaws', 'Neoplasm', 'Mastitis', 'Smallpox', 'Ectopic pregnancy', 'Scrapie', 'Pinguecula', 'Sepsis', 'Measles', 'Presbyopia', 'Scurvy', 'Osteoporosis', 'Diabetes Mellitus', 'Tularemia', 'Cerebral palsy', 'Colorectal Cancer', 'Gangrene', 'Acquired Capillary Haemangioma of Eyelid', 'Yellow Fever', 'Glaucoma', 'Bronchitis', 'Food Poisoning', 'Pericarditis', 'Premenstrual syndrome', 'Sickle-cell anemia', 'Rickets', 'Hepatitis A', 'Vitamin B12 Deficiency', 'Calculi', 'Sarcoidosis', 'Toxic shock syndrome', 'High risk pregnancy', 'Pneumonia', 'Arthritis', 'Gonorrhea', 'Ebola', 'Tetanus', 'Hypermetropia', 'Early pregnancy loss', 'Irritable bowel syndrome', 'Crimean Congo haemorrhagic fever (CCHF)', 'Turners Syndrome', 'Anthrax', 'Narcolepsy', 'Taeniasis', 'Legionellosis', 'Gaming disorder', 'Obesity', 'Dementia', 'Tennis elbow', 'Coronavirus disease 2019 (COVID-19)', 'Kuru', 'Black Death', 'Porphyria', 'Shingles', 'Mad cow disease', 'Atrophy', 'Astigmatism', 'Brain Tumour', 'Chorea', 'Sexually transmitted infections (STIs)', 'Blindness', 'Genital herpes', 'Goitre', 'Japanese Encephalitis', 'Mononucleosis', 'Diarrhea', 'Obsessive Compulsive Disorder', "Down's Syndrome", 'Post-herpetic neuralgia', 'Hyperthyroidism', 'Beriberi', 'Congenital anomalies (birth defects)', 'Marburg fever', 'Alopecia (hair loss)', 'Warts', 'Rabies', 'Stomach ulcers', 'Leukemia', 'Gastroenteritis', 'Hepatitis C', 'Vasovagal syncope', 'Myopia', 'Psoriasis', 'Chikungunya Fever', 'Peritonitis', 'Bunion', 'Amoebiasis', 'Chalazion', 'Cleft Lip and Cleft Palate', 'Acquired Immuno Deficiency Syndrome', 'Impetigo', 'Repetitive strain injury', 'Syphilis', 'Shigellosis', 'Sub-conjunctival Haemorrhage', 'Condyloma', 'Coronary Heart Disease', 'Neuralgia', 'Eczema', 'Middle East respiratory syndrome coronavirus (MERS‐CoV)', 'Lung cancer', 'Urticaria', 'Tay-Sachs disease', 'Hepatitis E']
        # diseases.sort()
        
        vectorizer = CountVectorizer()
        X__train_vectorized = vectorizer.fit_transform(X)  # Assuming X_train is your training data

        # Transform the processed user symptoms using the same vectorizer
        X_input_vectorized = vectorizer.transform(processed_user_symptoms)
        num_user_symptoms = len(processed_user_symptoms)
        X_input_vectorized = sp.csr_matrix(X_input_vectorized, shape=(num_user_symptoms, 489))
        # print(X_input_vectorized.shape)
        # print(type(X_input_vectorized))
        

        # Predict using the logistic regression model
        predictionproba = logistic_regression_model.predict_proba(X_input_vectorized)
        
        
        # Display the prediction probabilities
        # st.write("Prediction Probabilities:")
        # for i, prob in enumerate(predictionproba[0]):
        #     st.write(f"{diseases[i]}: {round(prob * 100, 2)}%")
        # for i, (disease, prob) in enumerate(zip(diseases[:10], predictionproba[0][:10])):
        #     st.write(f"{disease}: {round(prob * 100, 2)}%")

        # Combine diseases and probabilities into a list of tuples
        disease_prob_tuples = list(zip(diseases, predictionproba[0]))

        # Sort the list of tuples based on probabilities in descending order
        sorted_disease_prob_tuples = sorted(disease_prob_tuples, key=lambda x: x[1], reverse=True)
        
        # Display the top 10 diseases with the highest probabilities
        for i, (disease, prob) in enumerate(sorted_disease_prob_tuples[:10]):
            st.write(f"{disease}: {round(prob * 10000, 2)}%")
        
        first_disease = sorted_disease_prob_tuples[0][0]

        get_disease_details(first_disease)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


