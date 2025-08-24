# Honeywell Food & Beverage Process Anomaly Detection


https://github.com/user-attachments/assets/22e0187b-c45e-4d40-b884-3e7157c5b48f



## 🎯 Project Overview

This project focuses on predicting process anomalies in the food and beverage industry using machine learning techniques. The system identifies manufacturing mistakes that could affect product quality by analyzing various process parameters including raw material quantities, machine settings, temperature, mixing time, and other critical factors.

**Author:** Kowsigan M V  
**Student ID:** 226011  
**Email:** kowsiganmv@gmail.com  
**Institution:** KLN College of Engineering  

## 📋 Problem Statement

The challenge is to design a predictive maintenance system that:
- Collects process data (ingredient weights, mixer speed, oven temperature, timing)
- Predicts quality issues before they occur
- Prevents product defects caused by incorrect measurements, temperature variations, or improper mixing
- Provides real-time monitoring and visualization through a dashboard

## 🗂️ Datasets

### 1. Food & Beverage Process Data (`FnB_Process_Data_Batch_Wise.xlsx`)
**Process Parameters:**
- `Batch_ID`: Production batch identifier (1-25)
- `Time`: Process stage/time step
- `Flour (kg)`: Flour quantity used
- `Sugar (kg)`: Sugar quantity used
- `Yeast (kg)`: Yeast quantity used
- `Water Temp (°C)`: Water temperature for mixing
- `Salt (kg)`: Salt quantity used
- `Mixer Speed (RPM)`: Mixer rotation speed
- `Mixing Temp (°C)`: Temperature during mixing
- `Fermentation Temp (°C)`: Fermentation temperature
- `Oven Temp (°C)`: Baking temperature
- `Oven Humidity (%)`: Oven moisture level

### 2. Milk Quality Data ([Kaggle Dataset](https://www.kaggle.com/datasets/prudhvignv/milk-grading))
**Quality Parameters:**
- `pH`: Acidity/alkalinity level
- `Temperature`: Storage/measurement temperature
- `Taste`: Binary feature (good/spoiled)
- `Odor`: Binary feature (fresh/bad smell)
- `Fat`: Binary feature (fat presence)
- `Turbidity`: Binary feature (clarity/cloudiness)
- `Colour`: Numerical color intensity value
- `Grade`: Quality classification (high/medium/low)

## 🔄 Methodology

### Data Processing Pipeline
1. **Data Cleaning & Preprocessing**
   - Feature extraction with individual quality scoring
   - Deviation measurement from ideal recipe values
   - Badness score calculation and normalization
   - MinMax scaling for feature standardization

2. **Pattern Recognition**
   - Parallel coordinates analysis
   - Correlation analysis between parameters and quality
   - Optimal parameter range identification

3. **Feature Engineering**
   - Individual quality calculation: `Q_individual = Q_batch - badness_score + normalization`
   - MinMax scaling formula: `X_scaled = (X - X_min) / (X_max - X_min)`

## 🤖 Machine Learning Models

### Regression Models (Food & Beverage Data)
1. **Linear Regression**
   - Simple linear relationship modeling
   - Easy interpretation but limited for complex patterns

2. **Random Forest Regression** ⭐ **Best Performer**
   - Ensemble method with multiple decision trees
   - Captures complex non-linear interactions
   - Highest accuracy among tested models

3. **Gradient Boosting Regression**
   - Sequential tree building with error correction
   - High accuracy but sensitive to parameter tuning

### Classification Models (Milk Quality Data)
1. **Neural Network (Keras Sequential)** 
   - Architecture: Input → 16 neurons → 8 neurons → 3 output neurons
   - ReLU activation for hidden layers, Softmax for output
   - Adam optimizer with categorical crossentropy loss

2. **Support Vector Machine (SVM)**
   - RBF kernel with C=1.0 and gamma='scale'
   - Effective for clear class boundaries

3. **Random Forest Classifier** ⭐ **Best Performer**
   - 7 estimators with max_depth=5
   - Highest accuracy with minimal misclassifications

## 📊 Results

### Model Performance Comparison
- **Random Forest**: Highest accuracy for both regression and classification tasks
- **Neural Network**: Good performance, slightly lower than Random Forest
- **SVM**: Reliable but lower accuracy compared to ensemble methods

### Key Findings
- Quality strongly correlates with parameter adherence to optimal ranges
- Optimal ranges identified:
  - Water Temperature: 26-27°C
  - Mixer Speed: 145-160 RPM
  - Mixing/Fermentation Temp: ~38°C and 37.5°C
  - Oven conditions: ~180°C, 44-46% humidity

## 🚀 Implementation

### Technologies Used
- **Python**: Core programming language
- **Scikit-learn**: Machine learning algorithms
- **Keras/TensorFlow**: Deep learning models
- **Pandas & NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Streamlit**: Interactive dashboard
- **Joblib**: Model serialization and deployment

### Model Deployment
- Models saved using Joblib for efficient loading
- Streamlit dashboard for real-time predictions
- Interactive visualization of process parameters

## 📁 Project Structure

```
├── data/
│   ├── FnB_Process_Data_Batch_Wise.xlsx
│   └── milk_quality_data.csv
├── notebooks/
│   ├── food_beverage_analysis.ipynb
│   └── milk_quality_analysis.ipynb
├── models/
│   ├── random_forest_fnb.joblib
│   └── random_forest_milk.joblib
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## 🔗 Links & Resources

### Code Repositories
- **GitHub Repository**: [HonyWell_Multi_Variable_Data_Prediction](https://github.com/kowsiganMV/HonyWell_Multi_Variable_Data_Prediction)

### Google Colab Notebooks
- **Food & Beverage Analysis**: [Colab Link](https://colab.research.google.com/drive/1FmaDYMCOzkRSGGeeUDuLImzgCRB2iMH5?usp=sharing)
- **Milk Quality Prediction**: [Colab Link](https://colab.research.google.com/drive/1gfplK-vBFLkMTjlmLbiOAD4Cnfse0_GR#scrollTo=cWd7GJK4xqZQ)

### Demo
- **Streamlit App Demo**: [Video Link](https://drive.google.com/file/d/1CfZU23_tVAhV_3BQtIJ6qi-_y_junnhE/view?usp=drive_link)

## 📚 References

1. [Review on food quality assessment using machine learning and electronic nose system](https://www.sciencedirect.com/science/article/pii/S2590137023000626)

2. [Using machine learning models to predict the quality of plant-based foods](https://www.sciencedirect.com/science/article/pii/S2665927123001120)

3. [Milk Quality Prediction Using Machine Learning - ResearchGate](https://www.researchgate.net/publication/376064637_Milk_Quality_Prediction_Using_Machine_Learning)

4. [Milk Quality Prediction Using Machine Learning - EBSCO](https://search.ebscohost.com/login.aspx?direct=true&profile=ehost&scope=site&authtype=crawler&jrnl=24141399&AN=183567171&h=QE17fuODH8j6sX5k1lmPYQmOaS2oMqj8Q6jYmAZQHvD%2BwilbBAbRE6j%2BGQA8egIYA53zNd5Aw%2BI46Ss3iOiDDQ%3D%3D&crl=c)

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Libraries
```python
pandas
numpy
scikit-learn
tensorflow
matplotlib
seaborn
streamlit
joblib
```

### Running the Application
```bash
# Run Streamlit dashboard
streamlit run src/streamlit_app.py

# Run Jupyter notebooks
jupyter notebook notebooks/
```

## 🎯 Future Enhancements

- Real-time data integration from IoT sensors
- Advanced deep learning architectures
- Multi-product quality prediction
- Automated alert systems for anomaly detection
- Integration with manufacturing execution systems (MES)

## 📄 License

This project is developed for educational purposes as part of the Honeywell challenge.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

---
**Contact**: kowsiganmv@gmail.com | KLN College of Engineering
