"""A pipeline to load, train and evaluate a simple recommendation model."""
from zenml.pipelines import pipeline


@pipeline
def recommendation_pipeline(
    load_data, 
    train, 
    evaluate,
):
    trainset, testset = load_data()
    model = train(trainset)
    score = evaluate(model, testset)