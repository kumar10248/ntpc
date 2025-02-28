from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Renewable Energy Forecasting API",
    description="AI-powered forecasting system for solar and wind energy generation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Renewable Energy Forecasting API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
