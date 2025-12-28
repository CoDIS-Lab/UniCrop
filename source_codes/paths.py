import os

filename = "sample_data.csv"
INPUT_PATH = f"./data/{filename}"
MAP_PATH   = "./source_files/unicrop_feature_mapping.csv"
OUTPUT_PATH = "./"+filename.split(".")[0]+"_output"

OUT_CLEANED_INPUT   = "./source_files/cleaned_input_table.csv"
OUT_CLEANED_MAPPING = "./source_files/cleaned_feature_mapping.csv"
OUT_FETCH_PLAN      = "./source_files/fetch_plan.csv"

# Figures folder
FIGURES_PATH = OUTPUT_PATH+"/unicrop_figures"
os.makedirs(FIGURES_PATH, exist_ok=True)