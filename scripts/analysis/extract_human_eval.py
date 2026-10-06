import json
import os
import glob
import collections

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCES_FILE = os.path.join(BASE_DIR, 'html', 'data', 'human_eval_data', 'references_round2.json')
SUMMARY_DIR = os.path.join(BASE_DIR, 'html', 'outputs')
OUTPUT_FILE = os.path.join(BASE_DIR, 'human_eval_data_clean_21600.json')

MIN_DURATION_MS = 1500

def extract():
    if not os.path.exists(REFERENCES_FILE):
        print(f"Error: Missing {REFERENCES_FILE}")
        return

    # 1. Load the target 7,200 pairs from Round 2
    with open(REFERENCES_FILE, 'r') as f:
        references = json.load(f)
    
    target_pairs = set()
    for r in references:
        target_pairs.add((r['image_file'], r['model']))
    
    print(f"Targeting {len(target_pairs)} unique (image, model) pairs.")
    print(f"Goal: {len(target_pairs) * 3} total responses.")

    # 2. Collect and filter responses
    # responses[(img, speaker)] = [trial_data, ...]
    responses = collections.defaultdict(list)
    
    summary_files = glob.glob(os.path.join(SUMMARY_DIR, "summary_*.json"))
    print(f"Scanning {len(summary_files)} summary files...")

    valid_participants = 0
    
    for fpath in sorted(summary_files): # Sorted to be deterministic
        with open(fpath, 'r') as j:
            try:
                data = json.load(j)
                pid = str(data.get('participant_id', ''))
                if 'participant' in pid.lower() or 'participant' in os.path.basename(fpath).lower():
                    continue
            except:
                continue
            
            trials = data.get('all_examples', [])
            gold_trials = [t for t in trials if t.get('speaker') == 'gold']
            if not gold_trials: continue
            
            gold_hits = sum(1 for t in gold_trials if t.get('is_hit'))
            # Quality Criterion: >= 80% gold accuracy (4/5)
            if gold_hits / len(gold_trials) >= 0.8:
                valid_participants += 1
                pid = data.get('participant_id')
                bid = data.get('bucket_id')
                
                for t in trials:
                    img = t.get('img_name')
                    speaker = t.get('speaker')
                    key = (img, speaker)
                    
                    if key in target_pairs:
                        # Validation: Not sketchy
                        if t.get('click_x') is not None and t.get('duration_ms', 0) >= MIN_DURATION_MS:
                            # Deduplication: Stop at 3 responses per pair
                            if len(responses[key]) < 3:
                                # Add metadata from summary top level if needed
                                trial_clean = dict(t)
                                trial_clean['participant_id'] = pid
                                trial_clean['bucket_id'] = bid
                                responses[key].append(trial_clean)

    print(f"Found {valid_participants} valid participants.")

    # 3. Identify missing slots for top-up
    missing_list = []
    for key in sorted(target_pairs):
        count = len(responses[key])
        if count < 3:
            missing_list.append({
                "img": key[0],
                "speaker": key[1],
                "needed": 3 - count
            })
    
    with open(os.path.join(BASE_DIR, 'html', 'missing_trials_round2.json'), 'w') as f:
        json.dump(missing_list, f, indent=2)
    print(f"Saved {len(missing_list)} missing pairs to html/missing_trials_round2.json")

    # 4. Flatten and export
    final_data = []
    for key in sorted(responses.keys()):
        final_data.extend(responses[key])

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_data, f, indent=2)

    total_responses = len(final_data)
    print(f"\nExtraction Complete:")
    print(f"Total trials saved: {total_responses}")
    print(f"Unique pairs covered: {len(responses)}")
    print(f"Missing slots: {(len(target_pairs) * 3) - total_responses}")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract()
