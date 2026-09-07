# Credit Card Fraud Detection

Hyperparameter-tuned classifiers for card fraud, trained in a notebook and served behind a
Dockerised Flask prediction API.

## The problem

The Kaggle credit-card dataset: 284,807 transactions, of which 492 are fraudulent — **0.172%**.
That ratio is the whole difficulty. A model that predicts "never fraud" for every row scores
99.83% accuracy and is worth nothing, so accuracy is the one metric that cannot be used to
judge this problem. Recall on the fraud class is what matters, traded against how many
legitimate transactions you are willing to flag.

Features are PCA components (`V1`–`V28`) plus `Time` and `Amount`, so no feature engineering
is possible and no feature is interpretable.

## Approach

Grid-searched hyperparameters per model, then evaluated on a held-out set of 56,962
transactions containing 98 frauds. Trained models are serialised with `joblib` and loaded by a
Flask app that exposes a single `/predict` endpoint.

## Results

On the 98 fraud cases in the test set:

| Model | Fraud precision | Fraud recall | Fraud F1 | Best params |
|---|---|---|---|---|
| Decision Tree | 0.91 | **0.81** | 0.85 | `criterion=gini, max_depth=5` |
| Logistic Regression | 0.86 | 0.58 | 0.70 | `C=10, penalty=l2, solver=lbfgs` |

Read the recall column, not the accuracy column. Both models sit at 99.9% accuracy and they
are not close to equivalent: the decision tree catches 79 of 98 frauds, logistic regression
catches 57. Nineteen missed frauds versus forty-one is the entire difference between them, and
the accuracy figure hides it completely.

The depth-5 tree beating logistic regression is worth noting — with PCA features and this much
imbalance, a shallow tree carving out a few dense fraud regions does better than a linear
boundary, and it stays fast and inspectable.

A Random Forest was also trained and is served by the API, but I did not record its evaluation
in the notebook output, so I am not quoting a number for it.

## Running it

Train (the grid searches take several hours):

```bash
jupyter notebook ET_401_Credit_Card_Fraud.ipynb
```

Serve the pre-trained models:

```bash
git clone https://github.com/AsifShaikDS/credit-card-fraud-detection.git
cd credit-card-fraud-detection/et_deployment_ml_model
docker build -t fraud-api .
docker run -p 4000:80 fraud-api
```

```bash
curl -X POST http://localhost:4000/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "Random Forest", "data": [0, -1.359, -0.072, 2.536, "... 30 values total"]}'
```

`model` accepts `"Random Forest"` or `"Logistic Regression"`.

## What I would do differently

**The API returns a class, not a probability, and that is the wrong output for this problem.**
Fraud detection is a thresholding decision — a bank wants to set how many legitimate
transactions it will tolerate flagging, and then catch as much fraud as that budget allows.
Returning a hard 0/1 at sklearn's default 0.5 cutoff takes that decision away from the caller
and hard-codes a trade-off nobody chose. It should return `predict_proba`, and the threshold
should be picked from a precision-recall curve against an explicit cost assumption.

**I would evaluate with average precision rather than accuracy**, and I would report a
precision-recall curve instead of a single operating point, because at 0.172% positives the
ROC curve looks encouraging no matter what the model does.

**The endpoint trusts its input completely.** It reads `request.json['model']` and
`request.json['data']` with no schema, no length check, and no error handling — a missing key
is a 500, and a feature vector of the wrong length reaches sklearn and fails there. A
Pydantic request model, or plain length and type validation, is a few lines and turns every
one of those into a 400 with a usable message.

**The models are committed as `.joblib` binaries with no version pin.** They were pickled
against a specific scikit-learn version and will break or silently misbehave on a different
one. The training environment should be pinned in the requirements file and the model version
recorded alongside the artifact.
