import os
import requests
import time
import zipfile

URL = "https://zenodo.org/records/20078259/files/checkpoints.zip"

ZIP_PATH = "checkpoints.zip"
OUT_DIR = "checkpoints"


def download_file(url, filepath, max_retries=10):
    temp_file = filepath + ".part"

    session = requests.Session()
    session.trust_env = False 

    for attempt in range(max_retries):
        try:
            downloaded = 0
            if os.path.exists(temp_file):
                downloaded = os.path.getsize(temp_file)

            headers = {}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                print(f"[INFO] Resuming from {downloaded/1024/1024:.2f} MB")

            response = session.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0)) + downloaded

            mode = "ab" if downloaded > 0 else "wb"

            with open(temp_file, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total > 0:
                            percent = downloaded / total * 100
                            print(f"\r[Downloading] {percent:.2f}%", end="")

            print("\n[INFO] Download complete.")
            os.rename(temp_file, filepath)
            return

        except Exception as e:
            print(f"\n[WARNING] attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(3)

    raise RuntimeError("❌ Download failed after retries")

def check_zip_integrity(zip_path):
    import zipfile

    with zipfile.ZipFile(zip_path, 'r') as z:
        bad_file = z.testzip()  

    return bad_file  

def main():
    if os.path.exists(OUT_DIR) and len(os.listdir(OUT_DIR)) > 0:
        print("[INFO] checkpoints already exist, skip.")
        return

    if not os.path.exists(ZIP_PATH):
        print("[INFO] Downloading checkpoints.zip ...")
        download_file(URL, ZIP_PATH)
    else:
        print("[INFO] checkpoints.zip already exists")

    print("[INFO] Checking zip integrity...")
    bad_file = check_zip_integrity(ZIP_PATH)

    if bad_file is not None:
        print(f"[ERROR] Corrupted file detected: {bad_file}")
        print("[INFO] Removing corrupted zip and retrying...")

        os.remove(ZIP_PATH)
        if os.path.exists(ZIP_PATH + ".part"):
            os.remove(ZIP_PATH + ".part")

        download_file(URL, ZIP_PATH)

        bad_file = check_zip_integrity(ZIP_PATH)
        if bad_file is not None:
            raise RuntimeError("❌ Zip still corrupted after retry")

    print("[INFO] Zip integrity OK")

    print("[INFO] Extracting...")
    os.makedirs(OUT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(OUT_DIR)

    print("[INFO] Extraction done.")

    os.remove(ZIP_PATH)
    print("[INFO] Removed checkpoints.zip")


if __name__ == "__main__":
    main()
