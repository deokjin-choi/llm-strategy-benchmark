# scripts/combine_results.py
import os, glob
import pandas as pd

def combine_env_results(base_dir="infer_results", out_dir="infer_results"):
    os.makedirs(out_dir, exist_ok=True)
    envs = ["home", "company", "amd_cloud"]

    for env in envs:
        env_dir = os.path.join(base_dir, env)
        if not os.path.exists(env_dir):
            continue

        for scenario_folder in os.listdir(env_dir):
            folder_path = os.path.join(env_dir, scenario_folder)
            if not os.path.isdir(folder_path):
                continue

            all_csvs = glob.glob(os.path.join(folder_path, "*.csv"))
            if not all_csvs:
                continue

            dfs = [pd.read_csv(f) for f in all_csvs]
            df_all = pd.concat(dfs, ignore_index=True)

            out_file = f"{env}_{scenario_folder}.csv"
            out_path = os.path.join(out_dir, out_file)
            df_all.to_csv(out_path, index=False)
            print(f"✅ Saved {out_file} with {len(df_all)} rows.")

if __name__ == "__main__":
    combine_env_results()
