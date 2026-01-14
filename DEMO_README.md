# Model Demonstration Script

This script (`demo_models.py`) demonstrates all models from `final.ipynb` by taking user input and running predictions through all classification and clustering models.

## Features

### Classification Models
- **Logistic Regression** - Elastic-net regularized logistic regression
- **Decision Tree** - Decision tree classifier with hyperparameter tuning
- **LightGBM** - Gradient boosting framework (if available)
- **CatBoost** - Gradient boosting framework (if available)
- **Ensemble** - Weighted average of LightGBM and CatBoost (if both available)

### Clustering Models
- **K-Means** - K-means clustering with optimal k selection
- **DBSCAN** - Density-based clustering with noise detection

## Requirements

Install required packages:
```bash
pip install pandas numpy scikit-learn imbalanced-learn scipy matplotlib seaborn
```

Optional (for LightGBM and CatBoost):
```bash
pip install lightgbm catboost
```

## Usage

### Basic Usage
```bash
python demo_models.py
```

The script will:
1. Load and preprocess the data
2. Train all models
3. Prompt you for input features
4. Run predictions through all models
5. Display results

### Using Sample Data
To skip input and use sample data:
```bash
python demo_models.py --sample
# or
python demo_models.py -s
```

## Input Features

When prompted, you'll need to provide:

### Numeric Features
- `Administrative` - Number of administrative pages visited
- `Administrative_Duration` - Time spent on administrative pages (seconds)
- `Informational` - Number of informational pages visited
- `Informational_Duration` - Time spent on informational pages (seconds)
- `ProductRelated` - Number of product-related pages visited
- `ProductRelated_Duration` - Time spent on product pages (seconds)
- `BounceRates` - Bounce rate (0-1)
- `ExitRates` - Exit rate (0-1)
- `PageValues` - Page value metric
- `SpecialDay` - Special day indicator (0-1)

### Categorical Features
- `Month` - Month (1-12, where Jan=1, Feb=2, ..., Dec=12)
- `OperatingSystems` - Operating system (1-8)
- `Browser` - Browser type (1-13)
- `Region` - Geographic region (1-9)
- `TrafficType` - Traffic type (1-20)
- `VisitorType` - Visitor type (0=Other, 1=New Visitor, 2=Returning Visitor)
- `Weekend` - Weekend indicator (0=False, 1=True)

## Example Output

```
======================================================================
MODEL DEMONSTRATION SCRIPT
======================================================================

This script demonstrates all models from final.ipynb
Models: Logistic Regression, Decision Tree, LightGBM, CatBoost, Ensemble
Clustering: K-Means, DBSCAN

======================================================================
STEP 1: Loading and Preprocessing Data
======================================================================
...

======================================================================
STEP 5: Classification Predictions
======================================================================

Logistic Regression:
  Prediction: Purchase
  Purchase Probability: 0.7234
  Confidence: 0.7234

Decision Tree:
  Prediction: Purchase
  Purchase Probability: 0.8123
  Confidence: 0.8123

...

======================================================================
SUMMARY
======================================================================

Classification Results:
  Logistic Regression: Purchase (Probability: 0.7234)
  Decision Tree: Purchase (Probability: 0.8123)
  ...

Clustering Results:
  K-Means: Cluster 1
  DBSCAN: N/A (requires full dataset evaluation)
```

## Notes

- **Training Time**: The script trains all models from scratch, which may take a few minutes
- **DBSCAN Limitation**: DBSCAN doesn't have a direct `predict` method for new points. The script shows the parameters but notes that full dataset evaluation is needed
- **Model Availability**: LightGBM and CatBoost are optional. The script will work without them but won't include those models or the ensemble

## Troubleshooting

1. **File not found**: Make sure `online_shoppers_intention.csv` is in the same directory
2. **Import errors**: Install missing packages using `pip install <package-name>`
3. **Memory issues**: Reduce the number of iterations in hyperparameter search (edit `n_iter` parameters)

## Customization

You can modify the script to:
- Save trained models for faster subsequent runs
- Add more models
- Change hyperparameter search ranges
- Modify evaluation metrics

