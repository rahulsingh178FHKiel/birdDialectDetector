import requests
import pandas as pd
import time

API_KEY = "79fda5a5f52bac09898a9d78c054095b77be383f"
SPECIES = "Emberiza citrinella"
BASE_URL = "https://xeno-canto.org/api/3/recordings"

# We will test both Countries and Cities to show the difference in data volume
COUNTRIES = [
    "germany", "united kingdom", "france", "poland", "czech republic", 
    "sweden", "norway", "finland", "denmark", "netherlands", "austria"
]

CITIES = [
    "berlin", "london", "paris", "prague", "vienna", "warsaw", 
    "stockholm", "oslo", "helsinki", "copenhagen", "amsterdam"
]

def check_yield(tag_type, location):
    """
    Queries Xeno-Canto API and returns the total number of recordings 
    matching the strict project hygiene filters.
    """
    query = (
        f'sp:"{SPECIES}" '
        f'type:song '
        f'{tag_type}:"{location}" '
        f'q:">C" '
        f'month:">2" month:"<7"'
    )
    params = {"query": query, "key": API_KEY}
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return int(data.get("numRecordings", 0))
    except Exception as e:
        pass
    
    return 0

def main():
    print("Querying Xeno-Canto API for high-quality Yellowhammer recordings...")
    print("Filters: Quality A/B, March-June, Song type only\n")
    
    # 1. Evaluate Countries
    print("Evaluating by Country (Recommended)...")
    country_results = []
    for country in COUNTRIES:
        count = check_yield("cnt", country)
        country_results.append({"Location": country.title(), "Recordings": count, "Type": "Country"})
        time.sleep(0.5) # Be nice to the API
        
    # 2. Evaluate Cities
    print("Evaluating by City...")
    city_results = []
    for city in CITIES:
        count = check_yield("loc", city)
        city_results.append({"Location": city.title(), "Recordings": count, "Type": "City"})
        time.sleep(0.5)
        
    # Build and sort tables
    df_countries = pd.DataFrame(country_results).sort_values(by="Recordings", ascending=False).reset_index(drop=True)
    df_cities = pd.DataFrame(city_results).sort_values(by="Recordings", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*50)
    print(" YELLOWHAMMER RECORDING YIELDS (EUROPE)")
    print("="*50)
    
    print("\n--- TABLE 1: BY COUNTRY ---")
    print(df_countries[["Location", "Recordings"]].to_string())
    
    print("\n--- TABLE 2: BY CITY ---")
    print(df_cities[["Location", "Recordings"]].to_string())
    
    print("\nNote: City-level queries in Xeno-Canto often return 0 because recordists use")
    print("highly unstructured text for locations (e.g., 'Hampstead Heath, near London').")
    print("If you want large datasets, consider classifying by Country (e.g., UK vs DE vs FR).")

if __name__ == "__main__":
    main()
