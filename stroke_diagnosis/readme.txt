python -m venv venv
source venv/bin/activate  
run server: uvicorn main:app --reload hoac python3 -m uvicorn backend.api:app --reload
