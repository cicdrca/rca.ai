C:\Users\krish\PycharmProjects\LightGPM\venv\Scripts\python.exe C:\Users\krish\PycharmProjects\LightGPM\npmErrorLGBMV2.py 
Label Distribution:
 label
0    2
1    2
2    2
3    2
4    2
dtype: int64
Embedding Shape: (10, 384)
Training labels: (array([0, 1, 2, 3, 4], dtype=int64), array([1, 1, 1, 1, 1], dtype=int64))
Test labels: (array([0, 1, 2, 3, 4], dtype=int64), array([1, 1, 1, 1, 1], dtype=int64))
C:\Users\krish\PycharmProjects\LightGPM\venv\Lib\site-packages\sklearn\metrics\_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
C:\Users\krish\PycharmProjects\LightGPM\venv\Lib\site-packages\sklearn\metrics\_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
C:\Users\krish\PycharmProjects\LightGPM\venv\Lib\site-packages\sklearn\metrics\_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))

Classification Report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00         1
           1       0.50      1.00      0.67         1
           2       0.50      1.00      0.67         1
           3       1.00      1.00      1.00         1
           4       0.00      0.00      0.00         1

    accuracy                           0.60         5
   macro avg       0.40      0.60      0.47         5
weighted avg       0.40      0.60      0.47         5


Prediction Results:
Predicted Label: 2
Confidence: 0.23
Suggested Fix: Verify the package name and check if it's available in the npm registry. Consider updating your npm version.

Process finished with exit code 0
