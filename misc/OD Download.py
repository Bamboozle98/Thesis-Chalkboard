import requests
import os

BASE_URL = r"https://studentsecuedu66932-my.sharepoint.com/shared?id=%2Fpersonal%2Fhartda23%5Fecu%5Fedu%2FDocuments%2FResearch%20Files%2FSuperpixel%20Project%2Fpathology&listurl=%2Fpersonal%2Fhartda23%5Fecu%5Fedu%2FDocuments&source=waffle"  # Replace this

SAVE_DIR = r"E:\Geradt\downloaded_tiffs"
os.makedirs(SAVE_DIR, exist_ok=True)

for i in range(2, 504):  # from 002 to 503 inclusive
    filename = f"{i:03d}.tiff"
    url = f"{BASE_URL}{filename}"
    save_path = os.path.join(SAVE_DIR, filename)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Missing or failed: {filename} (status {response.status_code})")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
