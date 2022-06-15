from layer.decorators import dataset as layer_dataset, model as layer_model
from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import pipeline, node
from kedro.runner import SequentialRunner
from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
import pandas as pd
import layer

# Prepare a data catalog
data_catalog = DataCatalog({"iris": MemoryDataSet()})


# Prepare first node
def build_dataset():
    iris = datasets.load_iris()

    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    return df


# Prepare second node
def train_model(iris_dataset=None):
    if iris_dataset is None:
        iris_dataset = layer.get_dataset("iris").to_pandas()

    clf = SVC()

    X = iris_dataset.drop(["target"], axis=1)
    y = iris_dataset["target"]

    clf.fit(X, y)
    return clf


# Assemble nodes into a pipeline
metadata_store_enabled_pipeline = pipeline([
    node(
        func=layer_dataset("iris")(build_dataset),
        inputs=None,
        outputs="iris",
    ),
    node(
        func=layer_model("predictor")(train_model),
        inputs="iris",
        outputs="model",
    )
])

local_pipeline = pipeline([
    node(
        func=build_dataset,
        inputs=None,
        outputs="iris",
    ),
    node(
        func=train_model,
        inputs="iris",
        outputs="model",
    )
])

ENABLE_METADATA_STORE = True

# Create a runner to run the pipeline
runner = SequentialRunner()

# Run the pipeline
if ENABLE_METADATA_STORE:
    layer_api_key = ""  # You can get yours from: https://app.layer.ai/me/settings/developer
    layer.login_with_api_key(layer_api_key)
    layer.init("layer-kedro")
    print(runner.run(metadata_store_enabled_pipeline, data_catalog))
else:
    print(runner.run(local_pipeline, data_catalog))
