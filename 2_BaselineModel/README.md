# Baseline Model

## Model Selection

* **Baseline Model Type:** Random Chance Classifier (Uniform Dummy Classifier)
* **Rationale:** A random chance baseline provides the absolute lower bound of performance. In a multi-class setting with balanced classes, it represents a model that makes predictions uniformly at random. 

## Model Performance

* **Evaluation Metric:** Classification Accuracy
* **Performance Score:** 
  * **Binary Classification (UK vs. Continent):** 50.00%
  * **4-Class Classification:** 25.00%
  * **5-Class Classification:** 20.00%
* **Cross-Validation Score:** N/A (Theoretical analytical baseline; score is mathematically deterministic and holds constant across all folds).

## Evaluation Methodology

* **Data Split:** Stratified Train/Validation/Test split (e.g., 80/20 train/validation) to ensure class balance remains identical to the population distribution.
* **Evaluation Metrics:** Accuracy is used as the primary metric. Because class distributions are kept balanced across regions, accuracy provides a clear, un-biased measure of classification performance.

## Metric Practical Relevance

* **Accuracy:** In the context of dialect classification, accuracy directly translates to the percentage of bird recordings whose regional origin is correctly identified. Comparing our CNN results (~87.5% for binary, ~65% for 4-class) against this baseline demonstrates that the model successfully learns distinct acoustic patterns corresponding to geographic regions.
