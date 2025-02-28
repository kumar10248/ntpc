# AI-Powered Renewable Energy Forecasting System

## Overview
This project provides a web-based AI-powered forecasting system that predicts solar and wind energy generation using historical weather data, developed for NTPC.

## Features
- Web Dashboard with interactive charts and analytics
- AI-Powered Predictions using machine learning models
- Real-Time Data Integration with weather APIs
- Geographical Adaptability for different locations
- API for Integration with energy management systems

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables: `cp .env.example .env`
4. Run setup script: `./scripts/setup.sh`
5. Start the application: `uvicorn api.main:app --reload`

## Documentation
See the `docs/` directory for detailed documentation.
