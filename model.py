import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from data_generator import generate_jee_data, generate_neet_data, generate_commerce_data
import streamlit as st


class ExamModel:
    def __init__(self, exam_type):
        self.exam_type = exam_type
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False

    def _get_data(self):
        if self.exam_type == 'JEE':
            df = generate_jee_data(50000)
        elif self.exam_type == 'NEET':
            df = generate_neet_data(50000)
        elif self.exam_type == 'CA' or self.exam_type == 'Commerce':
            df = generate_commerce_data(50000)
        else:
            return pd.DataFrame()
        
        # Randomly sample 10,000 records from the 50,000 generated
        return df.sample(n=10000, random_state=42)

    def train(self):
        df = self._get_data()
        if df.empty:
            raise ValueError(f"No data generated for {self.exam_type}")

        # Preprocessing
        X = df.drop('Target_Continuation', axis=1)
        y = df['Target_Continuation']
        
        # Identification of categorical vs numerical
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['number']).columns
        
        self.feature_columns = X.columns.tolist()

        # Encoding
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le

        # Scaling
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])

        # Train/Test Split (Internal valid)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.ann_model.fit(X_train, y_train)
        self.is_trained = True
        
        acc_rf = accuracy_score(y_test, self.model.predict(X_test))
        acc_ann = accuracy_score(y_test, self.ann_model.predict(X_test))
        return acc_rf, acc_ann

    def predict(self, input_data):
        """
        input_data: dict of feature values matching the schema
        """
        if not self.is_trained:
            self.train()
            
        df = pd.DataFrame([input_data])
        
        # Ensure minimal missing cols handling (though UI should provide all)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0 # Default fallback
                
        # Encode
        for col, le in self.encoders.items():
            if col in df.columns:
                # Handle unseen labels carefully or strictly
                # For this demo, we assume inputs match drop-downs
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    # Fallback for unseen label
                    df[col] = 0 
        
        # Scale
        # We need to construct the numerical part exactly like training
        # This implementation re-selects numerical columns based on training phase logic
        # Ideally we'd persist the column lists. Re-deriving for simplicity here:
        numerical_cols = [c for c in self.feature_columns if c not in self.encoders]
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])

        # Ensure correct column order and droppping any extra user inputs not in model
        df = df[self.feature_columns]
        
        # Predict RF
        prob_rf = self.model.predict_proba(df)[0][1] # Probability of class 1 (Continue)
        pred_rf = self.model.predict(df)[0]
        
        # Predict ANN
        prob_ann = self.ann_model.predict_proba(df)[0][1]
        pred_ann = self.ann_model.predict(df)[0]
        
        return pred_rf, prob_rf, pred_ann, prob_ann

    def get_feature_importance(self):
        if not self.is_trained:
            return {}
        return dict(zip(self.feature_columns, self.model.feature_importances_))

    def compare_models(self):
        """
        Trains RF, LR, and DT and returns metrics and confusion matrices.
        """
        df = self._get_data()
        if df.empty:
            raise ValueError(f"No data generated for {self.exam_type}")

        # Preprocessing
        X = df.drop('Target_Continuation', axis=1)
        y = df['Target_Continuation']
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['number']).columns
        
        # Encoding (local to this method to not mess up main model state if needed, 
        # but for comparison we can re-do it)
        X_encoded = X.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            
        scaler = StandardScaler()
        X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "ANN (Deep Learning)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                "Accuracy": acc,
                "Precision": prec,
                "F1 Score": f1,
                "Confusion Matrix": cm
            }
            
        return results

@st.cache_resource
def get_trained_model(exam_type):
    model = ExamModel(exam_type)
    acc_rf, acc_ann = model.train()
    return model, acc_rf, acc_ann

@st.cache_data
def get_model_comparison(exam_type):
    model = ExamModel(exam_type)
    return model.compare_models()

@st.cache_data
def get_dataset_distribution(exam_type):
    model = ExamModel(exam_type)
    return model._get_data()
