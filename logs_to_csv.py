import os
import re
import pandas as pd

def parse_log_file(filepath):
    """
    Extracts model names and heart rates from a single log file.
    Returns a dict: {model_name: heart_rate}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to capture Model and Estimated Heart Rate
    pattern = r"Model:\s*(.*?)\nEstimated Heart Rate:\s*([\d\.]+)\s*BPM"
    matches = re.findall(pattern, content)

    data = {}
    for model, hr in matches:
        data[model.strip()] = float(hr)

    return data


def process_logs(log_dir, output_file="output"):
    """
    Processes all .log files in a directory and creates a DataFrame.
    """
    all_data = {}

    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            filepath = os.path.join(log_dir, filename)

            # Extract row name (first part before underscore or .log)
            row_name = filename.split("_")[0].replace(".log", "")

            log_data = parse_log_file(filepath)
            all_data[row_name] = log_data

    # Create DataFrame
    df = pd.DataFrame.from_dict(all_data, orient='index')

    df_formatted = df.copy()

    for col in df_formatted.columns:
        df_formatted[col] = df_formatted[col].apply(
            lambda x: f"{x:.2f}".replace('.', ',') if pd.notnull(x) else ""
        )

    # Save outputs
    df_formatted.to_csv(f"{output_file}.csv", sep=';')

    return df_formatted


if __name__ == "__main__":
    log_directory = "./results"  # <-- change to your folder path
    df = process_logs(log_directory, "open-rppg_results")

    print(df)