import os


def save_dataframe(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, compression="gzip")
    print(f"Dataframe sauvegardé dans {output_path}")
