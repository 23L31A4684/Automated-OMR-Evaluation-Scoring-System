from fastapi import FastAPI, Query
from omr_processor import evaluate_omr
import os
import logging
import pandas as pd

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/evaluate/")
async def evaluate(student_id: str = Query(..., description="Student ID"), version: str = Query(..., description="Version A or B"), image_path: str = Query(..., description="Path to OMR image")):
    logger.info(f"Received request: student_id={student_id}, version={version}, image_path={image_path}")
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return {"error": f"Image not found: {image_path}"}
    try:
        result = evaluate_omr(image_path, student_id, version)
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": "Internal server error"}

@app.post("/upload_image/")
async def upload_image(file: bytes = Query(..., description="Image bytes"), student_id: str = Query(...), version: str = Query(...)):
    image_path = f"samples/uploaded_{student_id}.jpeg"
    with open(image_path, "wb") as f:
        f.write(file)
    return {"message": f"Image uploaded for {student_id}", "image_path": image_path}

@app.get("/get_all_results/")
async def get_all_results():
    if not os.path.exists('results.csv'):
        return {"error": "No results available"}
    try:
        df = pd.read_csv('results.csv', names=['Student ID', 'Version', 'Score'])
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error reading results.csv: {e}")
        return {"error": "Failed to load results"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)