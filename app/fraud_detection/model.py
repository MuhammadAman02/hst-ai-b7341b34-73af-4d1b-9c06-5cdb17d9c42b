import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class FraudDetectionModel:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()

    def train(self, data):
        # Assuming data is a pandas DataFrame with relevant features
        scaled_data = self.scaler.fit_transform(data)
        self.model.fit(scaled_data)

    def predict(self, transaction):
        # Assuming transaction is a dictionary with relevant features
        df = pd.DataFrame([transaction])
        scaled_transaction = self.scaler.transform(df)
        prediction = self.model.predict(scaled_transaction)
        return "Fraudulent" if prediction[0] == -1 else "Normal"

    def get_feature_importance(self):
        # This is a placeholder. Isolation Forest doesn't have built-in feature importance.
        # In a real-world scenario, you might use a different model or implement a custom method.
        return {"Feature 1": 0.3, "Feature 2": 0.2, "Feature 3": 0.5}