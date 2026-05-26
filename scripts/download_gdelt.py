from src.ingestion.download_gdelt import GDELT_KEYWORDS, download_gdelt

if __name__ == "__main__":
    files = download_gdelt("2018-01-01", "2026-04-30", GDELT_KEYWORDS)
    print(f"saved/kept {len(files)} files")
