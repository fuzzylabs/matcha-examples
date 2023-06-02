import logging
import mlflow
from sklearn import datasets
from sklearn.svm import SVC

logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
)

def setup_tracking():
    mlflow.set_tracking_uri("http://51.142.152.216:5000")
    mlflow.set_experiment("iris-classifier")
    mlflow.sklearn.autolog()

def train(dataset):
    clf = SVC()
    clf.fit(dataset.data, dataset.target_names[dataset.target])
    return clf

def main():
    setup_tracking()

    logging.info("Training model")
    iris_dataset = datasets.load_iris()
    model = train(iris_dataset)

    example = iris_dataset.data[100]
    prediction = model.predict([example])
    logging.info(f"Prediction for {example} -> {prediction}")

if __name__ == "__main__":
    main()
