# Statcast Expected Batting Average (xBA) Model

### A Machine Learning Approach to Contact Quality and Hitting Performance

## Overview

This project builds a fully reproducible machine learning pipeline to model **Expected Batting Average (xBA)** using MLB Statcast data.

The goal is to:

* Reproduce Statcast's published xBA using logistic regression
* Improve upon Statcast using enhanced features and gradient boosting
* Generate player-level leaderboards
* Use this enhanced xBA calculation to find undervalued players

---

## What is Expected Batting Average?

Expected Batting Average (xBA) estimates the probability that a batted ball becomes a hit based on its quality of contact.

It is fundamentally a probabilistic model:

[
xBA = P(\text{Hit} \mid \text{Contact Quality})
]

Statcast’s published xBA is based primarily on:

* Exit Velocity
* Launch Angle

This project replicates that baseline and extends it using additional predictive features and modern machine learning methods.

---

## Project Architecture

This repository is structured as a complete end-to-end machine learning pipeline:

```
statcast-xba-model/

data/
    raw/
    processed/
    models/
    predictions/

src/

    statcast_pull.py
        Pulls Statcast data via pybaseball

    build_dataset.py
        Cleans data and engineers features

    build_model.py
        Trains Logistic Regression and XGBoost models

    predict.py
        Generates xBA predictions on new data

    calculate_player_xba.py
        Aggregates predictions into player leaderboards
```

---

## Methodology

### Target Variable

Binary classification:

```
is_hit = 1 if single/double/triple/home_run
is_hit = 0 otherwise
```

Other qualified at bats are included as:

```
xBA contribution = 0
```

to ensure consistency with official batting average and Statcast methodology.

---

### Features

Baseline model:

* launch_speed
* launch_angle
* launch_speed_sq
* launch_angle_sq
* exit_velocity × launch_angle interaction

Enhanced model adds:

* spray_angle
* sprint_speed

---

### Models Implemented

#### Logistic Regression

Baseline probabilistic model:

[
P(hit) = \frac{1}{1 + e^{-X\beta}}
]

Used to replicate Statcast methodology.

---

#### XGBoost Classifier

Gradient boosted decision trees capable of capturing:

* nonlinear relationships
* feature interactions
* higher-order effects

---

## Evaluation Metrics

Models are evaluated using:

* ROC-AUC
* Log Loss
* Brier Score
* Correlation with Statcast xBA

---

## Player Leaderboards

Predictions are aggregated to produce player-level metrics:

* Actual Batting Average
* Statcast xBA
* Model xBA
* Model vs Statcast differences

This allows identification of:

* Underperforming hitters
* Overperforming hitters
* Players undervalued by Statcast

---

## Example Output

| Player   | Actual BA | Statcast xBA | Model xBA |
| -------- | --------- | ------------ | --------- |
| Player A | .265      | .282         | .301      |
| Player B | .291      | .305         | .317      |

---

## Reproducibility

Full pipeline execution:

```
python statcast_pull.py

python build_dataset.py

python build_model.py

python predict.py

python calculate_player_xba.py
```

---

## Results Summary

The enhanced and XGBoost models improve predictive performance over baseline logistic regression and closely track Statcast expected batting average while incorporating additional predictive information.

---

## Future Improvements

Potential extensions:

* Hyperparameter tuning
* Cross-validation
* Bayesian modeling approaches
* Incorporation of defensive positioning
* Park factors
* Temporal modeling of player skill

---

## Author

Matt Waring

---
