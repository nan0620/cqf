##Question 1: What are voting classifiers in ensemble learning?

A voting classifier is an ensemble learning method designed to bring together the predictions of several different machine learning models to arrive at a final classification decision. This method is commonly used to solve classification problems in which each model votes on which category the sample belongs to. Below is a detailed explanation of voting classifiers:

1. __Hard Voting__:
Hard Voting is a form of voting classifier in which multiple models select the final prediction according to Majority Vote. In Hard Voting, each model makes a prediction for a given input sample, and then the final prediction is the category that receives the most votes.
__EXAMPLE__: Suppose there are three different models A, B, and C that each classify the same sample. Model A predicts category 1, model B predicts category 2, and model C predicts category 1. Since category 1 received two votes and category 2 only one vote, the final decision in hard voting is to choose category 1.
__Hard voting is applicable to both binary classification problems and multi-category classification problems.__ It is a simple but effective method that is often used to combine several different types of models to obtain more stable and accurate classification results.

2. __Soft Voting__:
Unlike hard voting, soft voting takes into account the predicted probability or score of each model, not just the category labels. In soft voting, each model estimates a probability or score for each category for a given input sample and the final decision is based on the average of these probabilities.
__EXAMPLE__: Suppose there are three different models A, B, and C that classify the same sample. Model A estimates the probability of the sample belonging to category 1 to be 0.7 and category 2 to be 0.3; model B estimates the probability of category 1 to be 0.4 and category 2 to be 0.6; and model C estimates the probability of category 1 to be 0.6 and category 2 to be 0.4. In soft voting the probabilities of each category are averaged, and the final decision is then likely to be on category 1 because it has the highest average probability.
__Soft voting is commonly used to deal with probability estimation and is particularly suitable for multi-category classification problems__ because it is able to utilise probabilistic information from multiple models.

__Advantages of voting classifiers include:__
- Improved model robustness, as it combines the opinions of multiple models.
- Reduces the variance of the model, especially when a single model may overfit the data.
- Allows combining different types of models for better performance.

---

##Question 2: Explain the role of the regularization parameter C in a Support Vector Machine (SVM) model. How does varying C affect the model’s bias and variance trade-off?

In Support Vector Machines (SVMs), the regularisation parameter C is a key hyperparameter that is used to control the complexity and tolerance of the model. The role of C is to balance the trade-off between maximising intervals and minimising classification errors. This balance affects the performance of the SVM model on both training and unseen data.

The following is the role and impact of the regularisation parameter C:

__Role of C:__
C controls the tolerance of the SVM model during training. Smaller values of C encourage greater spacing of the model, i.e., some training samples are allowed to be misclassified in order to maintain the simplicity of the model.
Larger values of C cause the model to adapt more tightly to the training data to minimise classification errors, i.e. reduce the spacing, which can lead to more complex decision boundaries.

__Affects the bias and variance trade-off of the model:__

- __Small C (larger intervals, high bias, low variance):__
    - A smaller C value will result in a larger interval for the model, allowing some training samples to be misclassified.
    - This will lead to high bias because the model is more concerned with classifying the training data correctly rather than striving for a smaller training error.
    - Models with high bias may be oversimplified and insensitive to noise in the data, and therefore may perform better on unseen data.

- __Large C (smaller interval, low bias, high variance):__
    - A larger C value will result in the model adapting more tightly to the training data to minimise classification errors.
    - This will result in low bias as the model is more concerned with classifying the training data correctly, i.e., it seeks a smaller training error.
    - Models with low bias may be more complex and sensitive to noise in the training data and therefore may overfit and perform poorly on unseen data.

Thus, the choice of C affects the bias and variance trade-off of the SVM model. __Smaller C values produce high bias, low variance models for noisier data, while larger C values produce low bias, high variance models for cleaner data.__ Choosing the appropriate C-value is key, and methods such as cross-validation are often required to determine the optimal hyper-parameter settings that will allow the model to perform well on both training and test data.

---

## Question 3: Follow the 7-steps to model building for your selected ticker.



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 示例数据：假设我们有一个包含特征的数据框，其中包括O-C、H-L、Sign、Past Returns等列。
# 请确保您的数据包含标记（0或1），用于二元分类。
data = pd.read_csv('stock_data.csv')  # 请替换为实际数据文件的路径

# 特征工程
features = data[['O-C', 'H-L', 'Sign', 'Past Returns', 'Momentum', 'SMA', 'EMA']]
labels = data['Label']  # Label是标记列，0表示负向趋势，1表示正向趋势

# 数据预处理
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

# 模型构建：随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 预测
y_pred = rf_classifier.predict(X_test)

# 模型评估
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
