from src.ingestion.download_ohlcv import download_ohlcv


if __name__ == "__main__":
    path = download_ohlcv()
    print(path)

