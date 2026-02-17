import pandas as pd
import os

def clean_csv_data(input_file, noisy_reps_list):
    """
    Removes specific rep IDs from the dataset.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Load the data
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")
    print(f"Unique reps before cleaning: {df['rep_id'].nunique()}")

    # Filter out the noisy reps
    # We keep only rows where the rep_id is NOT in our noisy list
    cleaned_df = df[~df['rep_id'].isin(noisy_reps_list)]

    # Calculate how many reps were removed
    removed_count = df['rep_id'].nunique() - cleaned_df['rep_id'].nunique()
    
    # Save the cleaned data
    output_file = "D:\Workout-Bie\last123_cleaned_partial_curl_training_data.csv" 
    cleaned_df.to_csv(output_file, index=False)
    
    print("--- Cleaning Complete ---")
    print(f"Removed {removed_count} noisy reps.")
    print(f"New data shape: {cleaned_df.shape}")
    print(f"Cleaned file saved as: {output_file}")

# --- CONFIGURATION ---
FILE_TO_CLEAN = 'D:\Workout-Bie\last12_cleaned_partial_curl_training_data.csv'

# Enter the rep numbers you want to delete here
# Example: [2, 5, 12] will delete all frames associated with those reps
NOISY_REPS = [472,482,499,508,509,510,519,528,531,534,535,536,537,541,542,545,546,547,548,556,557,558,559,560,561,562,564,565,575,585,586,587,588,589,590,591,592,596,621,641,653,654,655,656,657,658,659,660,661,662,663,664,675,676,677,678,685,687,688,689,699,700,701,702,703,704,705,706,723,736,768,801] 

if __name__ == "__main__":
    clean_csv_data(FILE_TO_CLEAN, NOISY_REPS)
    