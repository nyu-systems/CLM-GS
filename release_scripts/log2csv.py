import json
import pandas as pd
import os
import sys

def get_suffix_in_folder(folder):

    if not os.path.exists(folder):
        return None
    
    if not folder.endswith("/"):
        folder += "/"
    
    suffix_list_candidates = []
    for ws in [1,2,4,8,16,32]:
        for rk in range(ws):
            suffix_list_candidates.append(f"ws={ws}_rk={rk}")
    
    suffix_list = []
    for suffix in suffix_list_candidates:
        if os.path.exists(folder + "python_" + suffix + ".log"):
            suffix_list.append(suffix)

    return suffix_list












def extract_final_metrics_from_log(log_file):
    """
    Extract final metrics from a python.log file.
    Returns a dictionary with metrics: test_psnr, train_psnr, num_3dgs, 
    max_memory, pinned_memory, total_time
    """
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    metrics = {
        "test_psnr": None,
        "train_psnr": None,
        "num_3dgs": None,
        "max_gpu_memory_gb": None,
        "pinned_cpu_memory_gb": None,
        "total_time_s": None,
        "iterations": None,
        "throughput": None
    }
    
    # Parse from end to beginning to get the last occurrences
    for line in reversed(lines):
        # Extract end2end total_time (last occurrence)
        if metrics["total_time_s"] is None and "end2end total_time:" in line:
            # end2end total_time: 351.060 s, iterations: 30001, throughput 85.46 it/s
            try:
                metrics["total_time_s"] = float(line.split("end2end total_time: ")[1].split(" s")[0])
                metrics["iterations"] = int(line.split("iterations: ")[1].split(",")[0])
                metrics["throughput"] = float(line.split("throughput ")[1].split(" it/s")[0])
            except:
                pass
        
        # Extract test PSNR (last occurrence)
        if metrics["test_psnr"] is None and "Evaluating test:" in line:
            # [ITER 29997] Evaluating test: L1 0.015133645385503769 PSNR 32.101051330566406
            try:
                metrics["test_psnr"] = float(line.split("PSNR ")[1].strip())
            except:
                pass
        
        # Extract train PSNR (last occurrence)
        if metrics["train_psnr"] is None and "Evaluating train:" in line:
            # [ITER 29997] Evaluating train: L1 0.013125141151249409 PSNR 33.42594528198242
            try:
                metrics["train_psnr"] = float(line.split("PSNR ")[1].strip())
            except:
                pass
        
        # Extract memory metrics and 3dgs count (last occurrence with all info)
        if metrics["num_3dgs"] is None and "Now num of 3dgs:" in line and "Max Memory usage:" in line and "Now Pinned Memory:" in line:
            # iteration[29997,30001) densify_and_prune. Now num of 3dgs: 1215377. ... Max Memory usage: 1.7503581047058105 GB. ... Now Pinned Memory: 0.39734649658203125 GB
            try:
                metrics["num_3dgs"] = int(line.split("Now num of 3dgs: ")[1].split(".")[0])
                metrics["max_gpu_memory_gb"] = float(line.split("Max Memory usage: ")[1].split(" GB")[0])
                metrics["pinned_cpu_memory_gb"] = float(line.split("Now Pinned Memory: ")[1].split(" GB")[0])
            except:
                pass
        
        # Stop early if we have all metrics
        if all(v is not None for v in metrics.values()):
            break
    
    return metrics

def extract_all_experiments_to_csv(folder_path, output_csv=None):
    """
    Extract metrics from all subfolders in the given folder and save to CSV.
    Each subfolder should contain a python.log file.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    if output_csv is None:
        output_csv = os.path.join(folder_path, "experiment_results.csv")
    
    results = []
    
    # Get all subfolders
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    subfolders.sort()
    
    print(f"Found {len(subfolders)} subfolders in {folder_path}")
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        log_file = os.path.join(subfolder_path, "python.log")
        
        # Try alternative naming if python.log doesn't exist
        if not os.path.exists(log_file):
            # Try to find python_*.log files
            suffix_list = get_suffix_in_folder(subfolder_path)
            if suffix_list and len(suffix_list) > 0:
                log_file = os.path.join(subfolder_path, f"python_{suffix_list[0]}.log")
        
        if not os.path.exists(log_file):
            print(f"Warning: No log file found in {subfolder}, skipping...")
            continue
        
        print(f"Processing {subfolder}...")
        metrics = extract_final_metrics_from_log(log_file)
        
        if metrics:
            result_row = {
                "experiment": subfolder,
                "test_psnr": round(metrics["test_psnr"], 2) if metrics["test_psnr"] is not None else None,
                "train_psnr": round(metrics["train_psnr"], 2) if metrics["train_psnr"] is not None else None,
                "num_3dgs": metrics["num_3dgs"],
                "max_gpu_memory_gb": round(metrics["max_gpu_memory_gb"], 2) if metrics["max_gpu_memory_gb"] is not None else None,
                "pinned_cpu_memory_gb": round(metrics["pinned_cpu_memory_gb"], 2) if metrics["pinned_cpu_memory_gb"] is not None else None,
                "total_time_s": round(metrics["total_time_s"], 2) if metrics["total_time_s"] is not None else None,
                "iterations": metrics["iterations"],
                "throughput_it_s": round(metrics["throughput"], 2) if metrics["throughput"] is not None else None
            }
            results.append(result_row)
        else:
            print(f"Warning: Could not extract metrics from {subfolder}")
    
    # Create DataFrame and save to CSV and Markdown
    if results:
        df = pd.DataFrame(results)
        
        # Sort by experiment name after removing timestamp prefix
        def get_sort_key(exp_name):
            # Format: TIMESTAMP_TIMESTAMP_SCENENAME_OFFLOADTYPE
            # e.g., 20251111_231509_bicycle_no_offload
            parts = exp_name.split('_')
            if len(parts) >= 3:
                # Skip the first 2 parts (date and time timestamp)
                # Return the rest as the sort key
                return '_'.join(parts[2:])
            return exp_name
        
        df['_sort_key'] = df['experiment'].apply(get_sort_key)
        df = df.sort_values('_sort_key')
        df = df.drop(columns=['_sort_key'])
        df = df.reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully saved results to {output_csv}")
        
        # # Save to Markdown
        # output_md = output_csv.replace('.csv', '.md')
        # df.to_markdown(output_md, index=False)
        # print(f"Successfully saved markdown table to {output_md}")
        
        print(f"Processed {len(results)} experiments")
        print("\nPreview of results:")
        print(df.to_string())
    else:
        print("No results to save!")

if __name__ == "__main__":
    # read args from command line
    if len(sys.argv) < 2:
        print("Usage: python log2csv.py <folder_path> [output_csv]")
        print("  folder_path: Path to folder containing experiment subfolders")
        print("  output_csv: (optional) Path to output CSV file")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if this is for the new extraction function or old one
    if os.path.isdir(folder_path):
        # Check if it contains subfolders with log files (new format)
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        has_log_subfolders = any(
            os.path.exists(os.path.join(folder_path, sf, "python.log")) 
            for sf in subfolders[:5]  # Check first 5 subfolders
        )
        
        if has_log_subfolders:
            print("Detected log extraction mode...")
            extract_all_experiments_to_csv(folder_path, output_csv)
        else:
            raise ValueError("No log file found in the subfolders")
    else:
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)


