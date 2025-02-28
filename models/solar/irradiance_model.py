import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime
import pickle
# Load the data with the special format
def load_custom_data(file_path):
    """
    Load CSV data file with the specified custom format:
    - Row 2, columns 1-3: latitude, longitude, elevation
    - Row 5 onwards: time series data (time, temperature, wind speed, direct radiation, shortwave radiation)
    """
    # Read metadata (row 2)
    metadata = pd.read_csv(file_path, header=None, nrows=1, skiprows=1)
    latitude = metadata.iloc[0, 0]
    longitude = metadata.iloc[0, 1]
    elevation = metadata.iloc[0, 2]
    
    print(f"Location metadata: Lat={latitude}, Long={longitude}, Elevation={elevation}m")
    
    # Read actual time series data (starting from row 5)
    df = pd.read_csv(file_path, skiprows=10, low_memory=False)  # Added low_memory=False to fix mixed types warning
    
    # Rename columns if needed
    if len(df.columns) >= 5:
        column_mapping = {
            df.columns[0]: 'timestamp',
            df.columns[1]: 'temperature',
            df.columns[2]: 'wind_speed',
            df.columns[3]: 'direct_radiation',
            df.columns[4]: 'shortwave_radiation_instant'
        }
        df = df.rename(columns=column_mapping)
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    
    # Convert timestamp to datetime
    try:
        # Check if timestamp is in Unix timestamp format (negative numbers suggest Unix time)
        if df['timestamp'].dtype == np.int64 or (df['timestamp'].astype(str).str.match(r'-?\d+').all()):
            # Convert Unix timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            print("Converted Unix timestamp to datetime format")
        else:
            # Try standard datetime parsing
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print("Converted timestamp to datetime format")
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        # Add season (meteorological seasons)
        df['season'] = df['month'] % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
        
    except Exception as e:
        print(f"Error converting timestamp: {str(e)}")
        print("Creating dummy time features based on row index...")
        
        # Create dummy time features if timestamp conversion fails
        total_rows = len(df)
        # Assume data is hourly and starts at midnight Jan 1
        hours_per_day = 24
        days_per_year = 365
        
        df['hour'] = df.index % hours_per_day
        df['day_of_year'] = (df.index // hours_per_day) % days_per_year + 1
        df['month'] = ((df.index // hours_per_day) % days_per_year // 30) + 1  # Approximate
        df['month'] = df['month'].clip(1, 12)  # Ensure valid month range
        df['year'] = 2020  # Assign a default year
        df['season'] = df['month'] % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
    
    # Add location metadata as constants
    df['latitude'] = latitude
    df['longitude'] = longitude
    df['elevation'] = elevation
    
    return df

# Calculate theoretical energy generation based on physical models
def calculate_energy_potential(df):
    """
    Calculate theoretical energy potential based on physical models
    Wind power ~ wind_speed^3
    Solar power ~ direct_radiation + factor * shortwave_radiation
    
    Takes latitude into account for solar calculations
    """
    # Wind power calculation (theoretical)
    # P = 0.5 * air_density * swept_area * Cp * wind_speed^3
    # Using simplified constants for demonstration
    air_density = 1.225  # kg/m³
    wind_turbine_efficiency = 0.35  # Typical efficiency (Betz limit is 0.593)
    swept_area_per_sqkm = 2000  # m² of turbine swept area per km² (conservative estimate)
    
    # Air density adjustment for elevation
    if 'elevation' in df.columns:
        # Simple approximation: air density decreases by ~1.2% per 100m of elevation
        elevation_factor = 1 - 0.012 * (df['elevation'] / 100)
        air_density = air_density * elevation_factor.iloc[0]  # Use the first value since elevation is constant
    
    # Wind energy in kWh per sq. km
    df['wind_energy_potential'] = 0.5 * air_density * swept_area_per_sqkm * wind_turbine_efficiency * df['wind_speed']**3 / 1000  # Convert to kWh
    
    # Solar power calculation (theoretical)
    # P = solar_irradiance * panel_area * panel_efficiency
    panel_efficiency = 0.2  # 20% efficiency for modern solar panels
    panel_area_per_sqkm = 300000  # m² of panel area per km² (30% coverage)
    
    # Latitude-based adjustment for solar irradiance
    if 'latitude' in df.columns and 'month' in df.columns and 'hour' in df.columns:
        # Simple approximation of the effect of latitude on solar panel effectiveness
        # Higher latitudes get less direct sunlight, especially in winter
        latitude_rad = np.radians(abs(df['latitude'].iloc[0]))
        
        # Calculate declination angle for each day of year (simplified)
        df['declination'] = 23.45 * np.sin(np.radians(360/365 * (df['day_of_year'] - 81)))
        
        # Calculate solar angle factor (simplified)
        # 1.0 at equator at noon, decreases with latitude and as you move away from noon
        hour_angle = (df['hour'] - 12) * 15  # 15 degrees per hour
        hour_angle_rad = np.radians(hour_angle)
        declination_rad = np.radians(df['declination'])
        
        # Solar elevation angle = asin(sin(latitude) * sin(declination) + cos(latitude) * cos(declination) * cos(hour_angle))
        df['solar_elevation'] = np.arcsin(
            np.sin(latitude_rad) * np.sin(declination_rad) + 
            np.cos(latitude_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
        )
        
        # Convert to degrees and clip negative values (below horizon) to 0
        df['solar_elevation_deg'] = np.degrees(df['solar_elevation']).clip(0)
        
        # Calculate solar factor (effectiveness multiplier)
        # Simplified: sin of elevation angle gives a reasonable approximation
        df['solar_factor'] = np.sin(df['solar_elevation']).clip(0)
    else:
        # Default factor if we don't have needed columns
        df['solar_factor'] = 1.0
    
    # For direct radiation
    if 'direct_radiation' in df.columns:
        # Converting W/m² to kWh/m² for a 1-hour period, adjusted by solar factor
        df['solar_energy_direct'] = df['direct_radiation'] * df['solar_factor'] * panel_area_per_sqkm * panel_efficiency / 1000
    
    # For shortwave radiation (if available)
    if 'shortwave_radiation_instant' in df.columns:
        df['solar_energy_shortwave'] = df['shortwave_radiation_instant'] * df['solar_factor'] * panel_area_per_sqkm * panel_efficiency * 0.9 / 1000  # 0.9 factor as shortwave isn't fully convertible
    
    # Total solar energy
    solar_cols = [col for col in df.columns if col.startswith('solar_energy_')]
    if solar_cols:
        df['solar_energy_potential'] = df[solar_cols].sum(axis=1)
    
    # Combined energy potential
    df['combined_energy_potential'] = df['wind_energy_potential'] + df['solar_energy_potential']
    
    return df

# Feature engineering for better predictions
def engineer_features(df):
    """Create additional features that may improve model accuracy"""
    # Add wind-related features
    if 'wind_speed' in df.columns:
        df['wind_speed_squared'] = df['wind_speed']**2
        df['wind_speed_cubed'] = df['wind_speed']**3
    
    # Add temperature-related features
    if 'temperature' in df.columns:
        # Solar panel efficiency decreases as temperature increases above 25°C
        df['temp_above_25'] = np.maximum(0, df['temperature'] - 25)
        # Very low temperatures can also affect performance
        df['temp_below_0'] = np.maximum(0, -df['temperature'])
    
    # Interaction terms
    if 'wind_speed' in df.columns and 'temperature' in df.columns:
        df['wind_temp_interaction'] = df['wind_speed'] * df['temperature']
    
    # Solar radiation features
    if 'direct_radiation' in df.columns and 'hour' in df.columns:
        # Radiation effectiveness varies by time of day (due to angle of incidence)
        df['radiation_hour_effect'] = df['direct_radiation'] * np.sin(np.pi * df['hour'] / 24)
    
    # Day/night indicator - Check if hour column exists before using it
    if 'hour' in df.columns:
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
    
    # Seasonal wind patterns - Check if necessary columns exist
    if 'wind_speed' in df.columns and 'day_of_year' in df.columns:
        df['seasonal_wind'] = df['wind_speed'] * np.sin(2 * np.pi * df['day_of_year'] / 365)
    
    # Handle missing values if any
    df = df.fillna(method='ffill')  # Forward fill
    
    return df

# Build and train a combined energy prediction model
def build_combined_model(df, features, target='combined_energy_potential'):
    """Build and train a model for combined wind and solar energy prediction"""
    # Make sure target exists in dataframe
    if target not in df.columns:
        print(f"Error: Target '{target}' not found in dataframe columns: {df.columns.tolist()}")
        # Create a simple dummy target variable if not present
        df[target] = df['wind_speed'] * df['direct_radiation'] / 1000
        print(f"Created dummy '{target}' column for demonstration")
    
    # Prepare the data
    X = df[features]
    y = df[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    
    best_model = None
    best_score = -np.inf
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name} {target.replace('_', ' ').title()} Model:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test,
            'test_indices': X_test.index
        }
        
        if r2 > best_score:
            best_score = r2
            best_model = name
        
    pickle.dump(model, open(f"model.pkl", 'wb'))
    
    print(f"\nBest {target.replace('_', ' ').title()} Model: {best_model} with R² of {best_score:.4f}")
    
    # Feature importance for the best model
    if hasattr(results[best_model]['model'], 'feature_importances_'):
        importances = results[best_model]['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 10 Feature Importance:")
        feature_list = list(X.columns)
        for i in range(min(10, len(feature_list))):
            idx = indices[i]
            if idx < len(feature_list):  # Ensure index is valid
                print(f"{feature_list[idx]}: {importances[idx]:.4f}")
    
    return results, X.columns.tolist()

# Build separate models for wind and solar
def build_separate_models(df, features):
    """Build separate models for wind and solar energy prediction"""
    print("\nBuilding Wind Energy Model...")
    wind_results, _ = build_combined_model(df, features, target='wind_energy_potential')
    
    print("\nBuilding Solar Energy Model...")
    solar_results, _ = build_combined_model(df, features, target='solar_energy_potential')
    
    return wind_results, solar_results

# Function to generate hourly predictions
def predict_hourly_energy(df, combined_results, wind_results, solar_results, features):
    """Generate hourly predictions for combined and separate wind/solar energy"""
    # Get the best models
    best_combined_model_name = max(combined_results.items(), key=lambda x: x[1]['r2'])[0]
    best_wind_model_name = max(wind_results.items(), key=lambda x: x[1]['r2'])[0]
    best_solar_model_name = max(solar_results.items(), key=lambda x: x[1]['r2'])[0]
    
    # Get scalers and models
    combined_scaler = combined_results[best_combined_model_name]['scaler']
    wind_scaler = wind_results[best_wind_model_name]['scaler']
    solar_scaler = solar_results[best_solar_model_name]['scaler']
    
    combined_model = combined_results[best_combined_model_name]['model']
    wind_model = wind_results[best_wind_model_name]['model']
    solar_model = solar_results[best_solar_model_name]['model']
    
    # Create a copy of the DataFrame for predictions
    predictions_df = df.copy()
    
    # Prepare features for prediction - ensure all features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for feat in missing_features:
            features.remove(feat)
    
    X = predictions_df[features]
    
    # Scale features
    X_scaled_combined = combined_scaler.transform(X)
    X_scaled_wind = wind_scaler.transform(X)
    X_scaled_solar = solar_scaler.transform(X)
    
    # Make predictions
    predictions_df['combined_energy_prediction'] = combined_model.predict(X_scaled_combined)
    predictions_df['wind_energy_prediction'] = wind_model.predict(X_scaled_wind)
    predictions_df['solar_energy_prediction'] = solar_model.predict(X_scaled_solar)
    
    # Ensure no negative predictions (physically impossible)
    predictions_df['combined_energy_prediction'] = np.maximum(0, predictions_df['combined_energy_prediction'])
    predictions_df['wind_energy_prediction'] = np.maximum(0, predictions_df['wind_energy_prediction'])
    predictions_df['solar_energy_prediction'] = np.maximum(0, predictions_df['solar_energy_prediction'])
    
    # Keep only the time-related and prediction columns for output
    # Get time-related columns that exist in the dataframe
    time_cols = [col for col in ['timestamp', 'hour', 'day_of_year', 'month', 'year'] if col in predictions_df.columns]
    
    # Add prediction columns
    output_cols = time_cols + ['combined_energy_prediction', 'wind_energy_prediction', 'solar_energy_prediction']
    
    hourly_predictions = predictions_df[output_cols]
    
    return hourly_predictions

# Analyze hourly patterns across different months
def analyze_monthly_hourly_patterns(hourly_predictions):
    """Analyze how hourly energy patterns vary by month"""
    # Check if necessary columns exist
    if 'month' not in hourly_predictions.columns or 'hour' not in hourly_predictions.columns:
        print("Error: Cannot analyze monthly hourly patterns - missing 'month' or 'hour' columns")
        # Return empty dataframe if we can't do the analysis
        return pd.DataFrame()
    
    # Group by month and hour
    monthly_hourly = hourly_predictions.groupby(['month', 'hour']).agg({
        'combined_energy_prediction': 'mean',
        'wind_energy_prediction': 'mean',
        'solar_energy_prediction': 'mean'
    }).reset_index()
    
    # Plot a grid of months
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month in range(1, 13):
        ax = axes[month-1]
        month_data = monthly_hourly[monthly_hourly['month'] == month]
        
        if len(month_data) > 0:
            ax.plot(month_data['hour'], month_data['wind_energy_prediction'], 
                    label='Wind Energy', marker='o', markersize=4)
            ax.plot(month_data['hour'], month_data['solar_energy_prediction'], 
                    label='Solar Energy', marker='s', markersize=4)
            ax.plot(month_data['hour'], month_data['combined_energy_prediction'], 
                    label='Combined Energy', marker='^', markersize=4)
            
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Energy (kWh/km²)')
            ax.set_title(f'{month_names[month-1]} - Hourly Energy Production')
            ax.set_xticks(range(0, 24, 3))
            ax.grid(True)
            
            # Only show legend on the first plot
            if month == 1:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig('monthly_hourly_patterns.png')
    plt.show()
    
    return monthly_hourly

# Export hourly predictions to CSV
def export_hourly_predictions(hourly_predictions, filename='hourly_energy_predictions.csv'):
    """Export hourly predictions to a CSV file"""
    hourly_predictions.to_csv(filename, index=False)
    print(f"Hourly predictions exported to {filename}")
    return filename

# Main function to run the entire workflow
def main(file_path):
    """Main function to run the entire workflow"""
    print("Loading and processing custom format data...")
    df = load_custom_data(file_path)
    
    print("\nData overview:")
    print(df.head())
    print("\nData columns:", df.columns.tolist())
    print("\nData shape:", df.shape)
    
    print("\nCalculating theoretical energy potential...")
    df = calculate_energy_potential(df)

    print("*******************************")
    print(df.columns)
    
    print("\nEngineering features...")
    df = engineer_features(df)
    
    # Select features for models
    # Exclude non-feature columns and target variables
    exclude_cols = [
        'timestamp', 'wind_energy_potential', 'solar_energy_potential', 
        'combined_energy_potential', 'solar_energy_direct', 'solar_energy_shortwave',
        'declination', 'solar_elevation', 'solar_elevation_deg', 'solar_factor'  # Exclude intermediate calculation columns
    ]
    
    features = [col for col in df.columns if col not in exclude_cols]
    
    # Check if we have constant columns (like latitude) that shouldn't be used as features
    constant_columns = []
    for col in features:
        if df[col].nunique() == 1:
            constant_columns.append(col)
    
    if constant_columns:
        print(f"\nRemoved constant columns from features: {constant_columns}")
        features = [col for col in features if col not in constant_columns]
    
    print("\nSelected features for modeling:", features)
    
    print("\nBuilding combined energy prediction model...")
    combined_results, features = build_combined_model(df, features)
    
    print("\nBuilding separate wind and solar models...")
    wind_results, solar_results = build_separate_models(df, features)
    
    print("\nGenerating hourly predictions...")
    hourly_predictions = predict_hourly_energy(df, combined_results, wind_results, solar_results, features)
    
    print("\nHourly predictions sample:")
    print(hourly_predictions.head())
    
    # Export the predictions
    output_file = export_hourly_predictions(hourly_predictions)
    
    # Analyze patterns by month
    print("\nAnalyzing hourly patterns across months...")
    monthly_patterns = analyze_monthly_hourly_patterns(hourly_predictions)
    
    # Summary statistics
    print("\nSummary of Energy Production:")
    summary = pd.DataFrame({
        'Statistic': ['Mean (kWh/km²)', 'Median (kWh/km²)', 'Max (kWh/km²)', 'Min (kWh/km²)'],
        'Wind Energy': [
            hourly_predictions['wind_energy_prediction'].mean(),
            hourly_predictions['wind_energy_prediction'].median(),
            hourly_predictions['wind_energy_prediction'].max(),
            hourly_predictions['wind_energy_prediction'].min()
        ],
        'Solar Energy': [
            hourly_predictions['solar_energy_prediction'].mean(),
            hourly_predictions['solar_energy_prediction'].median(),
            hourly_predictions['solar_energy_prediction'].max(),
            hourly_predictions['solar_energy_prediction'].min()
        ],
        'Combined Energy': [
            hourly_predictions['combined_energy_prediction'].mean(),
            hourly_predictions['combined_energy_prediction'].median(),
            hourly_predictions['combined_energy_prediction'].max(),
            hourly_predictions['combined_energy_prediction'].min()
        ]
    })
    
    print(summary)
    
    print(f"\nPredictions have been saved to {output_file}")
    print("\nDone!")
    
    return hourly_predictions, combined_results, wind_results, solar_results

# Example usage - just update the file path
if __name__ == "__main__":
    file_path = "newDataSet.csv"  # Replace with your actual file path
    hourly_predictions, combined_results, wind_results, solar_results = main(file_path)