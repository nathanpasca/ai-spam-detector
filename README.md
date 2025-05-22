# AI Spam Detection System

A Python-based spam detection system utilizing a Naive Bayes classifier and TF-IDF vectorization to classify text messages as spam or ham. This project was developed as part of a machine learning course assignment to demonstrate text classification techniques.

## Features

- **Machine Learning Classification**: Uses Naive Bayes classifier with TF-IDF vectorization
- **Model Evaluation**: Generates ROC and Precision-Recall curves for performance analysis
- **Text Preprocessing**: Automated text cleaning and feature extraction
- **Customizable Threshold**: Adjustable spam detection sensitivity
- **Performance Metrics**: Comprehensive evaluation with accuracy, precision, recall, and F1-score

## Prerequisites

Before running this system, make sure you have:

- Python 3.9+
- A labeled dataset (`dataset.csv`) with columns `text` and `text_type` (containing 'ham' or 'spam')
- Required Python libraries (listed below)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nathanpasca/ai-spam-detector
   cd ai-spam-detector
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas scikit-learn numpy matplotlib seaborn
   ```

3. **Prepare the dataset:**
   - Ensure you have a `dataset.csv` file in the project directory with two columns:
     - `text`: message content
     - `text_type`: label ('ham' or 'spam')

## Usage

1. **Run the script:**
   ```bash
   python3 spam_detection.py
   ```

2. **The system will:**
   - Load and preprocess the dataset (`dataset.csv`)
   - Train a Naive Bayes classifier using TF-IDF vectorized text
   - Generate evaluation curves (ROC and Precision-Recall) and save them as `model_evaluation_curves.png`
   - Display performance metrics and classification results

## Example Usage

### Testing Individual Messages

```python
# Example of testing custom messages
test_messages = [
    "Win a free iPhone now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account will be closed",
    "Thanks for the meeting today"
]

for message in test_messages:
    prediction = model.predict([message])
    print(f"Message: '{message}' -> Classification: {prediction[0]}")
```

### Console Output

```
Dataset loaded: 5000 messages
Training completed successfully!
Model Accuracy: 95.2%
Precision: 94.8%
Recall: 95.6%
F1-Score: 95.2%

Evaluation curves saved to 'model_evaluation_curves.png'

Sample predictions:
Message: 'Win a free iPhone now!' -> Classification: Spam
Message: 'Hey, are we still meeting for lunch tomorrow?' -> Classification: Ham
```

## Configuration

### Adjusting Spam Threshold

The default spam threshold is set to 0.5 but can be adjusted in the code for different sensitivity levels:

```python
spam_threshold = 0.5  # Adjust this value (0.0 to 1.0)
```

- **Lower values** (e.g., 0.3): More sensitive, catches more spam but may have false positives
- **Higher values** (e.g., 0.7): Less sensitive, fewer false positives but may miss some spam

## Dataset Format

Ensure your `dataset.csv` file follows this format:

```csv
text,text_type
"Free money! Click here now!",spam
"Hello, how are you doing?",ham
"URGENT: Your account will be closed",spam
"Thanks for the meeting today",ham
"Click here to claim your prize",spam
"See you at the office tomorrow",ham
```

## File Structure

```
ai-spam-detector/
├── spam_detection.py          # Main script
├── dataset.csv               # Training dataset
├── model_evaluation_curves.png  # Generated evaluation plots
└── README.md                 # This file
```

## Technical Details

- **Algorithm**: Multinomial Naive Bayes
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Evaluation Metrics**: ROC curve, Precision-Recall curve, Accuracy, Precision, Recall, F1-Score
- **Text Processing**: Tokenization and vectorization using scikit-learn

## Model Performance

The system provides comprehensive evaluation metrics:

- **ROC Curve**: Shows the trade-off between true positive rate and false positive rate
- **Precision-Recall Curve**: Displays precision vs. recall for different thresholds
- **Confusion Matrix**: Visual representation of classification results
- **Classification Report**: Detailed metrics for each class

## Troubleshooting

### Common Issues

1. **Dataset Not Found**: Ensure `dataset.csv` is in the same directory as the script.

2. **Import Errors**: Install all required dependencies using pip:
   ```bash
   pip install pandas scikit-learn numpy matplotlib seaborn
   ```

3. **Memory Issues**: For large datasets, consider using data sampling or batch processing.

4. **Permission Errors**: Make sure Python has write permissions to save the evaluation curves.

5. **Encoding Issues**: Ensure your dataset is saved in UTF-8 encoding.

## Performance Notes

- For optimal performance, use a diverse and well-labeled dataset
- Larger datasets generally improve classification accuracy
- Consider data preprocessing steps like removing stop words for better results
- Balance your dataset - having roughly equal numbers of spam and ham messages improves performance

## Extending the System

### Adding New Features

1. **Different Algorithms**: Try other classifiers like SVM, Random Forest, or Logistic Regression
2. **Feature Engineering**: Add features like message length, special character count, or word frequency
3. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
4. **Hyperparameter Tuning**: Use GridSearchCV to optimize model parameters

### Example Extension

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Try different classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

# Compare performance
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"{name} Accuracy: {accuracy:.3f}")
```

## Academic Disclaimer

This project was developed as part of an academic assignment in Machine Learning. While it demonstrates text classification concepts, it may require additional features, robustness, and security measures for production use.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [matplotlib](https://matplotlib.org/) - Plotting library
- [seaborn](https://seaborn.pydata.org/) - Statistical data visualization
- [NumPy](https://numpy.org/) - Numerical computing library

## Support

If you encounter any problems or have questions, please open an issue on GitHub or contact the project maintainer.
