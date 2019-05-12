import sklearn.datasets as skdata
import identification as spid


X, y = skdata.make_blobs(
    n_samples=50, centers=5, n_features=30, random_state=0)
identifier = spid.create_trained_identifier(X, y)
