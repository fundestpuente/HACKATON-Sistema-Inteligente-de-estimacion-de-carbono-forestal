import requests
import json

BASE_URL = "https://api.inaturalist.org/v1/observations"

def getallobservations(per_page=10000, page=2):
    params = {
        "place_id": 7512,   # Ecuador
        "taxon_id": 47126,
        "quality_grade": "research",
        "per_page": per_page,
        "page": page
    }

    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json()["results"]


def get_nearby_observationsex(lat, lon, radius_km=100, per_page=50):
    params = {
        "place_id": 7515,       # Ecuador
        "lat": lat,
        "lng": lon,
        "radius": radius_km,
        "taxon_id": 47126,      
        "quality_grade": "research",
        "per_page": per_page
    }

    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.json()["results"]


def species_frequency(observations):
    freq = {}

    for obs in observations:
        taxon = obs.get("taxon")
        if taxon:
            name = taxon["name"]
            freq[name] = freq.get(name, 0) + 1

    return freq


def get_image_urls(observations):
    urls = []

    for obs in observations:
        for photo in obs.get("photos", []):
            url = photo["url"].replace("square", "large")
            urls.append(url)

    return urls




dict = getallobservations()

print(json.dumps(species_frequency(dict), indent=2))
print(json.dumps(get_image_urls(dict), indent=2))
