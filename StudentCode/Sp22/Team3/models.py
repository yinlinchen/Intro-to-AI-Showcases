import csv
import random
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import HTML


def compute_accuracy(model, training, testing):
    X_training = [row["evidence"] for row in training]
    y_training = [row["label"] for row in training]

    model.fit(X_training, y_training)

    X_testing = [row["evidence"] for row in testing]
    y_testing = [row["label"] for row in testing]

    predictions = model.predict(X_testing)

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_testing, predictions), annot=True, fmt="g")
    plt.title(f"Confusion Matrix ({type(model).__name__})")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")

    correct = 0
    incorrect = 0
    total = 0

    for actual, predicted in zip(y_testing, predictions):
        total += 1
        if actual == predicted:
            correct += 1
        else:
            incorrect += 1

    name = type(model).__name__
    accuracy = f"{100 * correct / total:.2f}%"
    return dict(name=name, correct=correct, incorrect=incorrect, accuracy=accuracy)


def main():
    models = [
        LogisticRegression(),
        Perceptron(),
        GaussianNB(),
        KNeighborsClassifier(n_neighbors=1),
        DecisionTreeClassifier(),
        SVC(),
    ]

    results = list()

    with open("BankNote_Authentication.csv") as f:
        reader = csv.reader(f)
        next(reader)
        data = list()

        for row in reader:
            data.append(
                {
                    "evidence": [float(cell) for cell in row[:4]],
                    "label": "Authentic" if row[4] == "0" else "Counterfeit",
                }
            )

    holdout = int(0.50 * len(data))
    random.shuffle(data)
    testing = data[:holdout]
    training = data[holdout:]

    for model in models:
        accuracy = compute_accuracy(model, training, testing)
        results.append(accuracy)

    results_df = dict(
        Model=[result["name"] for result in results],
        Correct=[result["correct"] for result in results],
        Incorrect=[result["incorrect"] for result in results],
        Accuracy=[result["accuracy"] for result in results],
    )

    df = pd.DataFrame(results_df)
    print(df.to_markdown())
    # print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
