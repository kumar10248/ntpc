import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import joblib
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IrradianceModel:
    """
    Solar irradiance prediction model using ensemble of machine learning techniques
    and deep learning for time-series forecasting.
    """
    
    def __init__(self, config=None):
        """
        Initialize the irradiance model with configuration
        
        Args:
            config (dict): Configuration parameters for the model
        """
        self.config = config or {
            'time_steps': 24,  # Look back window (hours)
            'forecast_horizon': 48,  # Forecast hours ahead
            'models': ['xgboost', 'lstm', 'rf'],  # Model ensemble
            'feature_engineering': True,
            'uncertainty_quantification': True,
            'validation_split': 0.2,
            'test_split': 0.1,
            'random_state': 42,
            'model_path': 'saved_models/irradiance/'
        }
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.models = {}
        self.feature_columns = None
        self.seasonal_features = True
        self.spatial_features = True
        
    def _engineer_features(self, data):
        """
        Generate features for the irradiance model
        
        Args:
            data (pd.DataFrame): Raw weather and solar data
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        logger.info("Engineering features for irradiance prediction")
        
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Extract datetime components
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Add cyclical encoding for time features to handle periodicity
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Calculate clear sky irradiance based on location and time
        if all(col in df.columns for col in ['latitude', 'longitude', 'day_of_year', 'hour']):
            df['clear_sky_irradiance'] = self._calculate_clear_sky_irradiance(
                df['latitude'], df['longitude'], df['day_of_year'], df['hour']
            )
        
        # Weather feature interactions
        if 'cloud_cover' in df.columns and 'temperature' in df.columns:
            df['cloud_temp_interaction'] = df['cloud_cover'] * df['temperature']
        
        # Handle cloud cover and atmospheric clarity
        if 'cloud_cover' in df.columns:
            df['clearness_index'] = 1 - df['cloud_cover']
        
        # Add lag features for time series aspects
        for col in ['temperature', 'cloud_cover', 'irradiance', 'humidity']:
            if col in df.columns:
                # Create lag features (previous hours)
                for lag in [1, 3, 6, 12, 24]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
                # Create rolling statistics
                df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=6).mean()
                df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24).mean()
                df[f'{col}_rolling_std_24h'] = df[col].rolling(window=24).std()
                
        # Drop rows with NaN values created by lagging
        df = df.dropna()
        
        # Store feature columns for future reference
        self.feature_columns = [col for col in df.columns 
                              if col not in ['timestamp', 'irradiance', 'date', 'solar_output']]
        
        return df
    
    def _calculate_clear_sky_irradiance(self, latitude, longitude, day_of_year, hour):
        """
        Calculate theoretical clear-sky irradiance based on solar position algorithms
        
        This is a simplified version. Production code would use pvlib or similar packages
        for more accurate calculations including atmospheric effects
        """
        # Convert to numpy arrays for vectorized operations
        latitude = np.array(latitude)
        day_of_year = np.array(day_of_year)
        hour = np.array(hour)
        
        # Solar declination angle
        declination = 23.45 * np.sin(np.radians(360.0 * (284 + day_of_year) / 365.0))
        
        # Hour angle
        hour_angle = 15.0 * (hour - 12.0)
        
        # Calculate solar elevation angle
        elevation = np.arcsin(
            np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
        )
        elevation_deg = np.degrees(elevation)
        
        # Set irradiance to 0 for nighttime
        elevation_deg = np.maximum(elevation_deg, 0)
        
        # Simplified clear-sky model (Bird Clear Sky Model simplified)
        solar_constant = 1361  # W/m²
        transmittance = 0.7  # Atmospheric transmittance
        clear_sky_irradiance = solar_constant * transmittance**(1.0 / np.sin(elevation)) * np.sin(elevation)
        
        # Set small values and negatives to 0
        clear_sky_irradiance = np.maximum(clear_sky_irradiance, 0)
        
        return clear_sky_irradiance
        
    def prepare_time_series_data(self, df, target_col='irradiance'):
        """
        Prepare data for time series prediction (for LSTM model)
        
        Args:
            df (pd.DataFrame): Processed feature dataframe
            target_col (str): Target column name
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Define features and target
        features = df[self.feature_columns].values
        target = df[target_col].values
        
        # Scale data
        scaled_features = self.scaler_X.fit_transform(features)
        scaled_target = self.scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        time_steps = self.config['time_steps']
        for i in range(len(scaled_features) - time_steps - self.config['forecast_horizon'] + 1):
            X.append(scaled_features[i:(i + time_steps)])
            # Target is the value at time_steps + forecast_horizon ahead
            y.append(scaled_target[i + time_steps + self.config['forecast_horizon'] - 1])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        train_size = int(len(X) * (1 - self.config['validation_split'] - self.config['test_split']))
        val_size = int(len(X) * self.config['validation_split'])
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def build_lstm_model(self):
        """
        Build and compile LSTM model for time series forecasting
        
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            LSTM(128, activation='relu', return_sequences=True, 
                 input_shape=(self.config['time_steps'], len(self.feature_columns))),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data):
        """
        Train the irradiance prediction models
        
        Args:
            data (pd.DataFrame): Historical weather and irradiance data
            
        Returns:
            dict: Trained model results and metrics
        """
        logger.info("Training irradiance prediction model")
        
        # Feature engineering
        if self.config['feature_engineering']:
            processed_data = self._engineer_features(data)
        else:
            processed_data = data.copy()
            self.feature_columns = [col for col in data.columns 
                                 if col not in ['timestamp', 'irradiance', 'date', 'solar_output']]
        
        # Prepare data for tabular models
        X = processed_data[self.feature_columns]
        y = processed_data['irradiance']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_split'] + self.config['validation_split'],
            random_state=self.config['random_state']
        )
        
        # Further split test into validation and test
        val_ratio = self.config['validation_split'] / (self.config['validation_split'] + self.config['test_split'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, 
            test_size=1-val_ratio,
            random_state=self.config['random_state']
        )
        
        metrics = {}
        
        # Train XGBoost model
        if 'xgboost' in self.config['models']:
            logger.info("Training XGBoost model")
            xgb_model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=self.config['random_state'],
                n_jobs=-1
            )
            
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=True
            )
            
            self.models['xgboost'] = xgb_model
            y_pred_xgb = xgb_model.predict(X_test)
            mse_xgb = np.mean((y_test - y_pred_xgb) ** 2)
            mae_xgb = np.mean(np.abs(y_test - y_pred_xgb))
            metrics['xgboost'] = {'mse': mse_xgb, 'mae': mae_xgb}
            
            # Feature importance
            importance = xgb_model.feature_importances_
            feature_importance = {feature: importance[i] for i, feature in enumerate(self.feature_columns)}
            metrics['feature_importance'] = feature_importance
            
            # Save model
            joblib.dump(xgb_model, f"{self.config['model_path']}xgboost_irradiance.pkl")
        
        # Train Random Forest model
        if 'rf' in self.config['models']:
            logger.info("Training Random Forest model")
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config['random_state'],
                n_jobs=-1
            )
            
            rf_model.fit(X_train, y_train)
            
            self.models['rf'] = rf_model
            y_pred_rf = rf_model.predict(X_test)
            mse_rf = np.mean((y_test - y_pred_rf) ** 2)
            mae_rf = np.mean(np.abs(y_test - y_pred_rf))
            metrics['rf'] = {'mse': mse_rf, 'mae': mae_rf}
            
            # Save model
            joblib.dump(rf_model, f"{self.config['model_path']}rf_irradiance.pkl")
        
        # Train LSTM model for time series forecasting
        if 'lstm' in self.config['models']:
            logger.info("Training LSTM model")
            # Prepare time series data
            X_train_ts, y_train_ts, X_val_ts, y_val_ts, X_test_ts, y_test_ts = self.prepare_time_series_data(
                processed_data, 'irradiance'
            )
            
            # Build and train LSTM model
            lstm_model = self.build_lstm_model()
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = lstm_model.fit(
                X_train_ts, y_train_ts,
                validation_data=(X_val_ts, y_val_ts),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )
            
            self.models['lstm'] = lstm_model
            
            # Evaluate LSTM model
            y_pred_lstm = lstm_model.predict(X_test_ts)
            y_pred_lstm = self.scaler_y.inverse_transform(y_pred_lstm).flatten()
            y_test_lstm = self.scaler_y.inverse_transform(y_test_ts.reshape(-1, 1)).flatten()
            
            mse_lstm = np.mean((y_test_lstm - y_pred_lstm) ** 2)
            mae_lstm = np.mean(np.abs(y_test_lstm - y_pred_lstm))
            metrics['lstm'] = {'mse': mse_lstm, 'mae': mae_lstm, 'history': history.history}
            
            # Save model
            lstm_model.save(f"{self.config['model_path']}lstm_irradiance")
            
            # Save scalers
            joblib.dump(self.scaler_X, f"{self.config['model_path']}scaler_X.pkl")
            joblib.dump(self.scaler_y, f"{self.config['model_path']}scaler_y.pkl")
        
        logger.info(f"Irradiance model training complete. Metrics: {metrics}")
        return metrics
    
    def predict(self, weather_data, model_type='ensemble'):
        """
        Predict solar irradiance using trained models
        
        Args:
            weather_data (pd.DataFrame): Weather data with required features
            model_type (str): Type of model to use ('xgboost', 'lstm', 'rf', or 'ensemble')
            
        Returns:
            np.array: Predicted irradiance values
        """
        if self.config['feature_engineering']:
            processed_data = self._engineer_features(weather_data)
        else:
            processed_data = weather_data.copy()
        
        if model_type == 'ensemble':
            predictions = {}
            weights = {'xgboost': 0.5, 'lstm': 0.3, 'rf': 0.2}
            
            # XGBoost prediction
            if 'xgboost' in self.models:
                X = processed_data[self.feature_columns]
                predictions['xgboost'] = self.models['xgboost'].predict(X)
            
            # Random Forest prediction
            if 'rf' in self.models:
                X = processed_data[self.feature_columns]
                predictions['rf'] = self.models['rf'].predict(X)
            
            # LSTM prediction
            if 'lstm' in self.models:
                # Prepare time series data for prediction
                features = processed_data[self.feature_columns].values
                scaled_features = self.scaler_X.transform(features)
                
                X_lstm = []
                time_steps = self.config['time_steps']
                
                for i in range(len(scaled_features) - time_steps + 1):
                    X_lstm.append(scaled_features[i:(i + time_steps)])
                
                X_lstm = np.array(X_lstm)
                
                lstm_preds = self.models['lstm'].predict(X_lstm)
                lstm_preds = self.scaler_y.inverse_transform(lstm_preds).flatten()
                
                # Align predictions with original data
                lstm_full = np.full(len(processed_data), np.nan)
                lstm_full[time_steps-1:time_steps-1+len(lstm_preds)] = lstm_preds
                predictions['lstm'] = lstm_full
            
            # Combine predictions with weights
            final_predictions = np.zeros(len(processed_data))
            weight_sum = 0
            
            for model_name, preds in predictions.items():
                if not np.all(np.isnan(preds)):
                    mask = ~np.isnan(preds)
                    final_predictions[mask] += weights[model_name] * preds[mask]
                    weight_sum += weights[model_name]
            
            # Normalize by actual weights used
            final_predictions = final_predictions / weight_sum if weight_sum > 0 else final_predictions
            
            return final_predictions
        else:
            # Single model prediction
            if model_type not in self.models:
                raise ValueError(f"Model {model_type} not found in trained models")
            
            if model_type in ['xgboost', 'rf']:
                X = processed_data[self.feature_columns]
                return self.models[model_type].predict(X)
            elif model_type == 'lstm':
                # Prepare time series data for LSTM
                features = processed_data[self.feature_columns].values
                scaled_features = self.scaler_X.transform(features)
                
                X_lstm = []
                time_steps = self.config['time_steps']
                
                for i in range(len(scaled_features) - time_steps + 1):
                    X_lstm.append(scaled_features[i:(i + time_steps)])
                
                X_lstm = np.array(X_lstm)
                
                lstm_preds = self.models['lstm'].predict(X_lstm)
                lstm_preds = self.scaler_y.inverse_transform(lstm_preds).flatten()
                
                # Align predictions with original data
                lstm_full = np.full(len(processed_data), np.nan)
                lstm_full[time_steps-1:time_steps-1+len(lstm_preds)] = lstm_preds
                
                return lstm_full
    
    def load_models(self, model_path=None):
        """
        Load saved models from disk
        
        Args:
            model_path (str): Path to saved models directory
            
        Returns:
            None
        """
        model_path = model_path or self.config['model_path']
        
        # Load XGBoost model
        try:
            self.models['xgboost'] = joblib.load(f"{model_path}xgboost_irradiance.pkl")
        except:
            logger.warning("Could not load XGBoost model")
        
        # Load Random Forest model
        try:
            self.models['rf'] = joblib.load(f"{model_path}rf_irradiance.pkl")
        except:
            logger.warning("Could not load Random Forest model")
        
        # Load LSTM model
        try:
            self.models['lstm'] = tf.keras.models.load_model(f"{model_path}lstm_irradiance")
            self.scaler_X = joblib.load(f"{model_path}scaler_X.pkl")
            self.scaler_y = joblib.load(f"{model_path}scaler_y.pkl")
        except:
            logger.warning("Could not load LSTM model")
        
        # Load feature columns
        try:
            with open(f"{model_path}feature_columns.txt", 'r') as f:
                self.feature_columns = f.read().splitlines()
        except:
            logger.warning("Could not load feature columns")

    def evaluate(self, test_data):
        """
        Evaluate model performance on test data
        
        Args:
            test_data (pd.DataFrame): Test data with ground truth
            
        Returns:
            dict: Evaluation metrics
        """
        if self.config['feature_engineering']:
            processed_data = self._engineer_features(test_data)
        else:
            processed_data = test_data.copy()
        
        X_test = processed_data[self.feature_columns]
        y_test = processed_data['irradiance']
        
        results = {}
        
        for model_name, model in self.models.items():
            if model_name in ['xgboost', 'rf']:
                y_pred = model.predict(X_test)
                mse = np.mean((y_test - y_pred) ** 2)
                mae = np.mean(np.abs(y_test - y_pred))
                rmse = np.sqrt(mse)
                
                results[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
        
        # LSTM evaluation requires special handling for time series
        if 'lstm' in self.models:
            X_train_ts, y_train_ts, X_val_ts, y_val_ts, X_test_ts, y_test_ts = self.prepare_time_series_data(
                processed_data, 'irradiance'
            )
            
            y_pred_lstm = self.models['lstm'].predict(X_test_ts)
            y_pred_lstm = self.scaler_y.inverse_transform(y_pred_lstm).flatten()
            y_test_lstm = self.scaler_y.inverse_transform(y_test_ts.reshape(-1, 1)).flatten()
            
            mse_lstm = np.mean((y_test_lstm - y_pred_lstm) ** 2)
            mae_lstm = np.mean(np.abs(y_test_lstm - y_pred_lstm))
            rmse_lstm = np.sqrt(mse_lstm)
            
            results['lstm'] = {
                'mse': mse_lstm,
                'mae': mae_lstm,
                'rmse': rmse_lstm
            }
        
        # Ensemble prediction
        y_pred_ensemble = self.predict(test_data, model_type='ensemble')
        # Filter out NaN values
        mask = ~np.isnan(y_pred_ensemble)
        mse_ensemble = np.mean((y_test[mask] - y_pred_ensemble[mask]) ** 2)
        mae_ensemble = np.mean(np.abs(y_test[mask] - y_pred_ensemble[mask]))
        rmse_ensemble = np.sqrt(mse_ensemble)
        
        results['ensemble'] = {
            'mse': mse_ensemble,
            'mae': mae_ensemble,
            'rmse': rmse_ensemble
        }
        
        return results

if __name__ == "__main__":
    # Example usage:
    # Create synthetic data for testing
    dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='H')
    np.random.seed(42)
    
    # Sample weather data
    data = pd.DataFrame({
        'timestamp': dates,
        'temperature': 25 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 365)) + np.random.normal(0, 3, len(dates)),
        'cloud_cover': np.random.uniform(0, 1, len(dates)),
        'humidity': np.random.uniform(0.3, 0.9, len(dates)),
        'wind_speed': np.random.weibull(2, len(dates)) * 5,
        'latitude': np.full(len(dates), 40.7),
        'longitude': np.full(len(dates), -74.0),
    })
    
    # Generate synthetic irradiance data
    hour_of_day = data['timestamp'].dt.hour
    day_of_year = data['timestamp'].dt.dayofyear
    
    # Simplified solar irradiance model for synthetic data
    max_irradiance = 1000  # W/m²
    day_factor = np.sin(np.pi * (day_of_year - 81) / 365) * 0.5 + 0.5  # Seasonal variation
    hour_factor = np.sin(np.pi * hour_of_day / 12)  # Daily variation
    hour_factor = np.maximum(hour_factor, 0)  # No negative values (night)
    
    # Base irradiance with daily and seasonal patterns
    data['irradiance'] = max_irradiance * day_factor * hour_factor
    
    # Reduce irradiance based on cloud cover
    data['irradiance'] = data['irradiance'] * (1 - 0.7 * data['cloud_cover'])
    
    # Add some random variation
    data['irradiance'] = data['irradiance'] + np.random.normal(0, 50, len(dates))
    data['irradiance'] = np.maximum(data['irradiance'], 0)  # No negative irradiance
    
    # Initialize and train the model
    irradiance_model = IrradianceModel()
    irradiance_model.train(data)
    
    # Make predictions
    predictions = irradiance_model.predict(data.iloc[-100:])
    
    # Evaluate model
    evaluation = irradiance_model.evaluate(data.iloc[-1000:])
    print(f"Model evaluation: {evaluation}")