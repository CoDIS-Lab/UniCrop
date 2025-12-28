# unicrop_model_main.py

from pipeline import run_modelling_after_download

if __name__ == "__main__":
    results = run_modelling_after_download(
        data_filepath="combined_final_dataset.csv",   # <-- your merged output
        target_column="yield",
        group_columns=["fp_District", "fp_Season(SA = Summer Autumn, WS = Winter Spring)"]
    )

    print(results)
