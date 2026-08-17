from sklearn.datasets import fetch_openml

# DELVE-style variants available on OpenML:
# puma8nh, puma8NH, puma32h, puma32H
Xy = fetch_openml(name="puma8nh", version=1, as_frame=True)
X, y = Xy.data, Xy.target

print(X.shape, y.shape)  # (8192, 8) for puma8nh

Xy = fetch_openml(name="puma32H", version=1, as_frame=True)
print(Xy.data.shape, Xy.target.shape)  # (8192, 32), (8192,)