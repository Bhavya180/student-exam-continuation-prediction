# Student Exam Continuation Prediction

A machine learning application that predicts whether students will continue their preparation for competitive examinations. This project uses Streamlit to provide an interactive web interface for exam continuation prediction, feature analysis, and comparison across different exams.

## Project Overview

This application helps analyze and predict student continuation patterns in competitive exam preparation. It includes:
- **Prediction Module**: Predict if a student will continue exam preparation
- **Comparison Analysis**: Compare prediction results across different exams
- **Exam Comparison**: Analyze trends and patterns across examination types
- **Feature Analysis**: Visualize feature importance and student metrics

## Project Structure

- `app.py` - Main Streamlit application and UI logic
- `model.py` - Machine learning model for predictions
- `feature_engineering.py` - Feature calculation and engineering
- `data_generator.py` - Generate synthetic training data
- `utils.py` - Utility functions and visualization charts
- `requirements.txt` - Project dependencies
- `LICENSE` - Project license

## Requirements

- Python 3.7 or higher
- pip (Python package manager)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Bhavya180/student-exam-continuation-prediction
cd student-exam-continuation-prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dependencies

The project uses the following libraries:
- **streamlit** - Interactive web application framework
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **plotly** - Interactive data visualization
- **joblib** - Model serialization and caching

## Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

### Main Features

1. **Prediction Page**
   - Select an examination type (e.g., GATE, UPSC, JEE)
   - Input student parameters
   - Get predictions on exam continuation likelihood
   - View feature importance and analysis

2. **Comparison Page**
   - Compare predictions across different student profiles
   - Analyze continuation patterns
   - View detailed metrics and statistics

3. **Exam Comparison Page**
   - Compare trends across different competitive exams
   - Analyze exam-specific patterns
   - View distribution and performance metrics

## Project Workflow

1. **Data Generation**: `data_generator.py` creates synthetic student data
2. **Feature Engineering**: `feature_engineering.py` calculates multi-factor scores
3. **Model Training**: `model.py` trains and manages the prediction model
4. **Visualization**: `utils.py` provides charts and visualizations
5. **Web Interface**: `app.py` serves the interactive UI

## Configuration

Modify the examination types and parameters in `app.py` sidebar to customize:
- Available examinations
- Feature thresholds
- Prediction criteria

## Model Details

The prediction model uses:
- Multi-factor scoring based on student preparation metrics
- Feature importance analysis for interpretability
- Classification algorithms from scikit-learn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions, please open an issue on GitHub.

## Author

[Your Name/Organization]

## Acknowledgments

- Built with Streamlit
- Machine learning with scikit-learn
- Visualizations with Plotly
