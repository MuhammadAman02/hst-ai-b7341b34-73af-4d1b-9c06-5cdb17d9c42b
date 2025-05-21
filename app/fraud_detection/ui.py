from nicegui import ui
import pandas as pd

class FraudDetectionUI:
    def __init__(self, model):
        self.model = model
        self.setup_ui()

    def setup_ui(self):
        @ui.page('/')
        def home():
            ui.label('Irish Credit Union Fraud Detection').classes('text-h3 text-primary')
            
            with ui.tabs().classes('w-full') as tabs:
                ui.tab('Fraud Check', icon='search')
                ui.tab('Model Training', icon='school')
                ui.tab('Analytics', icon='analytics')

            with ui.tab_panels(tabs, value='Fraud Check'):
                with ui.tab_panel('Fraud Check'):
                    self.fraud_check_panel()
                
                with ui.tab_panel('Model Training'):
                    self.model_training_panel()
                
                with ui.tab_panel('Analytics'):
                    self.analytics_panel()

    def fraud_check_panel(self):
        ui.label('Check a Transaction').classes('text-h5')
        amount = ui.number('Transaction Amount', value=0)
        account_age = ui.number('Account Age (days)', value=0)
        transaction_frequency = ui.number('Transaction Frequency (per month)', value=0)

        def check_transaction():
            transaction = {
                'amount': amount.value,
                'account_age': account_age.value,
                'transaction_frequency': transaction_frequency.value
            }
            result = self.model.predict(transaction)
            ui.notify(f'Transaction classified as: {result}', color='green' if result == 'Normal' else 'red')

        ui.button('Check Transaction', on_click=check_transaction)

    def model_training_panel(self):
        ui.label('Train the Model').classes('text-h5')
        file_upload = ui.upload(label='Upload Training Data (CSV)', auto_upload=True)

        def train_model(file):
            if file:
                df = pd.read_csv(file.path)
                self.model.train(df)
                ui.notify('Model trained successfully!', color='green')

        ui.button('Train Model', on_click=lambda: train_model(file_upload.value))

    def analytics_panel(self):
        ui.label('Model Analytics').classes('text-h5')
        feature_importance = self.model.get_feature_importance()
        
        with ui.column():
            ui.label('Feature Importance:').classes('text-h6')
            for feature, importance in feature_importance.items():
                ui.linear_progress(value=importance, label=feature).classes('w-full')