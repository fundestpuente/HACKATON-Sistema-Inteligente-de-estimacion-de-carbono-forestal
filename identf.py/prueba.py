import requests

url = "https://api.inaturalist.org/v1/observations"

headers = {
    "User-Agent": "AcademicProject/1.0 (contact@example.com)"
}

params = {
    "place_id": 7512,
    "per_page": 10
}

r = requests.get(url, headers=headers, params=params)

print("Status:", r.status_code)
print("URL:", r.url)
print("Response:", r.text[:300])
#....