from src.ingestion.download_gdelt import download_gdelt


if __name__ == "__main__":
    files = download_gdelt("2018-01-01", "2026-04-30", "bitcoin,btc,cryptocurrency")
    print(f"saved {len(files)} files")

