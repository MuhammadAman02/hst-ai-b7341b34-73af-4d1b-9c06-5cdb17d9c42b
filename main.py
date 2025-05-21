import os
from dotenv import load_dotenv
from nicegui import ui, app
from app.fraud_detection.ui import FraudDetectionUI
from app.fraud_detection.model import FraudDetectionModel

# Load environment variables
load_dotenv()

# Initialize the fraud detection model
model = FraudDetectionModel()

# Initialize the UI
fraud_detection_ui = FraudDetectionUI(model)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Irish Credit Union Fraud Detection", 
           port=int(os.getenv("PORT", 8000)),
           reload=False)