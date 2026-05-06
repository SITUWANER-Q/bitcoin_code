from src.models.m12_ra_bicma import build as build_m12


if __name__ == "__main__":
    model = build_m12(num_dim=52, txt_dim=32)
    print(type(model).__name__)

