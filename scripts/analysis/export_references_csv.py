import json
import csv
import os

def main():
    references_path = "html/data/human_eval_data/references.json"
    
    with open(references_path, 'r') as f:
        references = json.load(f)

    models_to_export = {
        "molmo_vanilla": "molmo_vanilla_references.csv",
        "binary_last_point": "binary_lp_references.csv",
        "binary": "binary_references.csv"
    }

    # Initialize data structures for each model
    model_data = {m: [] for m in models_to_export}

    for entry in references:
        model = entry.get("model")
        if model in models_to_export:
            dataset = entry.get("dataset")
            example_id = entry.get("example_id")
            reference = entry.get("generated_reference", "")
            
            # Use the model-specific filename format (prefix_dataset_id.jpg)
            # although for CSV we might just want to reference the original img_file
            # but wait, user might want it to match the filenames we'd use if we exported images
            # Looking at shaping_speaker_references.csv: shaping_refcoco_testA_54.jpg
            
            # Map model names to exported prefixes
            model_prefix_map = {
                "molmo_vanilla": "molmo",
                "binary_last_point": "binary_lp",
                "binary": "binary"
            }
            prefix = model_prefix_map.get(model, model)
            img_file = f"{prefix}_{dataset}_{example_id}.jpg"
            
            model_data[model].append({
                "Dataset": dataset,
                "Image_File": img_file,
                "Reference": reference,
                "Annotation": "" # Empty for manual annotation
            })

    # Write CSVs
    for model, filename in models_to_export.items():
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ["Dataset", "Image_File", "Reference", "Annotation"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in model_data[model]:
                writer.writerow(row)
        print(f"Exported {len(model_data[model])} references to {filename}")

if __name__ == "__main__":
    main()
