from src.ingestion.download_macro import download_macro


if __name__ == "__main__":
    paths = download_macro()
    for path in paths:
        print(path)

