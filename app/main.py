# app/main.py
from fastapi import FastAPI
from app.routes import router  # Import the router from routes.py

app = FastAPI()

# Include routes
app.include_router(router)

# Welcome route
@app.get("/")
def welcome():
    return {"message": "Welcome to the SVM Text Classification API"}
