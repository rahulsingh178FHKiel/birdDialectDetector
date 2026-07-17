import os
import requests
import json
import time
import pandas as pd

# --- Configuration ---
API_KEY = "79fda5a5f52bac09898a9d78c054095b77be383f"  # Xeno-Canto API Key
SPECIES = "Emberiza citrinella"
# Biological Compromise: Regional Bounding Boxes to capture tight geography but enough data.
REGIONS = {
    "east_germany": "50.5,11.0,54.0,15.0",
    "south_poland": "49.0,18.0,51.5,23.0",
    "southern_uk": "50.5,-2.5,52.5,1.5",
    "netherlands_belgium": "50.0,3.0,53.5,7.2",
    "southern_sweden": "55.0,11.0,60.0,19.0"
}
BASE_URL = "https://xeno-canto.org/api/3/recordings"
DATA_DIR = os.path.join("..", "data")
METADATA_FILE = os.path.join("..", "data", "metadata.csv")

# Data Hygiene Constraints
ALLOWED_QUALITY = ["A", "B"]
ALLOWED_MONTHS = [3, 4, 5, 6]  # March to June (Breeding Season)
MAX_REC_PER_RECORDIST = 5       # Prevent microphone signature bias
MAX_SAMPLES_PER_REGION = 50        # Regional Dialect target (Total ~169 files)

def fetch_region_metadata(region_name, box_coords):
    """Fetch all song metadata for a specific species and region using API v3 tag syntax."""
    print(f"\n--- Fetching metadata for {region_name.upper()} ---")

    # API v3 tag-based query:
    # box:<lat1,lon1,lat2,lon2> — geographic bounding box
    query = (
        f'sp:"{SPECIES}" '
        f'type:song '
        f'box:{box_coords} '
        f'q:">C" '
        f'month:">2" month:"<7"'
    )
    params = {"query": query, "key": API_KEY, "per_page": 500}

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"Error fetching data for {region_name}: {response.status_code}")
        print(f"Server Response: {response.text}")
        return []

    data = response.json()
    recordings = data.get("recordings", [])
    print(f"Found {len(recordings)} raw results for {region_name}.")
    return recordings

def apply_hygiene_filters(recordings, region_name):
    """Apply temporal, quality, and recordist de-duplication filters."""
    filtered = []
    recordist_counts = {}
    
    for rec in recordings:
        # 1. Quality Filter (A, B)
        if rec.get("q") not in ALLOWED_QUALITY:
            continue
            
        # 2. Temporal Filter (March - June)
        date_str = rec.get("date", "")
        try:
            # Date format is usually YYYY-MM-DD
            month = int(date_str.split("-")[1])
            if month not in ALLOWED_MONTHS:
                continue
        except (IndexError, ValueError):
            continue # Skip if date is malformed
            
        # 3. Recordist De-duplication (Max 5 per recordist)
        recordist = rec.get("rec", "Unknown")
        recordist_counts[recordist] = recordist_counts.get(recordist, 0) + 1
        if recordist_counts[recordist] > MAX_REC_PER_RECORDIST:
            continue
            
        # If all checks pass, add region label and store
        rec["target_region"] = region_name
        filtered.append(rec)
        
    print(f"Filtered down to {len(filtered)} high-quality recordings for {region_name}.")
    return filtered

def download_recordings(filtered_recordings):
    """Download the .mp3 files with a responsible delay."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    all_metadata = []
    
    for i, rec in enumerate(filtered_recordings):
        region = rec["target_region"]
        region_dir = os.path.join(DATA_DIR, region)
        if not os.path.exists(region_dir):
            os.makedirs(region_dir)
            
        file_id = rec["id"]
        file_path = os.path.join(region_dir, f"{file_id}.mp3")
        
        # Skip if already exists
        if os.path.exists(file_path):
            all_metadata.append(rec)
            continue
            
        print(f"[{i+1}/{len(filtered_recordings)}] Downloading {file_id} for {region}...")
        
        try:
            audio_response = requests.get(rec["file"], timeout=30)
            if audio_response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(audio_response.content)
                all_metadata.append(rec)
            else:
                print(f"Failed to download {file_id}: {audio_response.status_code}")
        except Exception as e:
            print(f"Error downloading {file_id}: {e}")
            
        # Responsible rate-limiting
        time.sleep(2)
        
    return all_metadata

def main():
    final_dataset_metadata = []
    seen_ids = set()

    for region_name, box_coords in REGIONS.items():
        raw_rec = fetch_region_metadata(region_name, box_coords)
        clean_rec = apply_hygiene_filters(raw_rec, region_name)

        # Global Deduplication: Remove any recordings already claimed by another region
        clean_rec = [r for r in clean_rec if r["id"] not in seen_ids]

        # Cap the data to maintain class balance
        clean_rec = clean_rec[:MAX_SAMPLES_PER_REGION]

        # Update seen_ids
        for r in clean_rec:
            seen_ids.add(r["id"])

        metadata = download_recordings(clean_rec)
        final_dataset_metadata.extend(metadata)
        
    # Save metadata — extract only the flat fields we actually need.
    # Avoids all nested-object issues (sono, osci, annotation-set, also).
    if final_dataset_metadata:
        KEEP_FIELDS = ["id", "gen", "sp", "en", "rec", "cnt", "loc",
                       "lat", "lon", "date", "q", "length", "type",
                       "file", "file-name", "target_region"]
        rows = []
        for rec in final_dataset_metadata:
            rows.append({k: rec.get(k, "") for k in KEEP_FIELDS})
        df = pd.DataFrame(rows)
        df.to_csv(METADATA_FILE, index=False)
        print(f"\nSuccess! Metadata for {len(df)} recordings saved to {METADATA_FILE}")

if __name__ == "__main__":
    main()
