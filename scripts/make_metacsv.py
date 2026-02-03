import pandas as pd

df = pd.read_csv("c:/Users/user00/Desktop/MCI_Diffusion/outputs/mlp_diffusion/samples_q3_0202.csv")

# 컬럼명 맞추기
df = df.rename(columns={"lat": "latitude", "lon": "longitude"})

# 폴더명 규칙이 소수 6자리라서 맞춰주는 게 안전함
df["latitude"] = df["latitude"].astype(float).round(6)
df["longitude"] = df["longitude"].astype(float).round(6)

# 중복 제거(같은 좌표 여러 번이면 엔진을 중복 실행하게 됨)
dfu = df[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)

# grid_id 붙이기
grid = dfu.copy()
grid.insert(0, "grid_id", range(1, len(grid) + 1))

out = "c:/Users/user00/Desktop/MCI_Diffusion/grid_metadata_from_samples.csv"
grid.to_csv(out, index=False, encoding="utf-8-sig")
print("saved:", out, "rows:", len(grid))