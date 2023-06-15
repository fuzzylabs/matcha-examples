import logging
import mlflow
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matcha_ml.core as matcha


logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
)


def setup_tracking():
    mlflow_uri = matcha.get(resource_name='experiment-tracker', property_name=None).components[0].find_property('tracking-url').value
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("iris-classifier")
    mlflow.sklearn.autolog()

def create_train_test_data():
    data, target = datasets.load_iris(return_X_y=True, as_frame=True)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)

    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train):
    clf = SVC()
    clf.fit(x_train, y_train)
    return clf 

def evaluate_model(clf, x_test, y_test):
    return accuracy_score(y_test, clf.predict(x_test))


def main():
    setup_tracking()
    
    with mlflow.start_run():
        logging.info("Training model")

        x_train, x_test, y_train, y_test = create_train_test_data()
        model = train_model(x_train, y_train)

        accuracy = evaluate_model(model, x_test, y_test)

        logging.info(f"Accuracy: {accuracy}")

        example, label = x_train.loc[100], y_train.loc[100]
        prediction = model.predict([example])
        logging.info(f"Predict for {label} ({example.values}) -> {prediction}")


if __name__ == "__main__":
    main()
