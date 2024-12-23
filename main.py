# app.py
from fastapi import FastAPI, HTTPException
import pandas as pd
import os
import openai
import modules_AI_06_19 as md
from transformers import AutoTokenizer, AutoModelForSequenceClassification


app = FastAPI()

BASE_PATH = "./"
AUDIO_PATH = os.path.join(BASE_PATH, "audio/")
LOCATION_CSV = os.path.join(BASE_PATH, "audio_location.csv")
EMERGENCY_DATA_CSV = os.path.join(BASE_PATH, "emergen_df.csv")
MODEL_PATH = os.path.join(BASE_PATH, "fine_tuned_bert/")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERVICE_KEY = os.getenv('SERVICE_KEY')
CLIENT_ID, CLIENT_SECRET = os.getenv('CLIENT_ID'), os.getenv('CLIENT_SECRET')

A2T = md.AudioTextProcessor(api_key=OPENAI_API_KEY)
Bert = md.ModelInstance(loadpath=MODEL_PATH)
public_data_service = md.PublicData(service=SERVICE_KEY)
Recog = md.GetDistance(
    csv=EMERGENCY_DATA_CSV,
    c_id=CLIENT_ID,
    c_key=CLIENT_SECRET,
    public_data_service=public_data_service
)

point = pd.read_csv(BASE_PATH + 'audio_location.csv')

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

@app.get("/test")
def test_endpoint():
    return {"message": "This is a test endpoint. Working fine!!!!!!!!!!!!!!!!!!!!!!"}

@app.get("/fastapi_hospital/{filename}")
async def process_pipeline(filename: str):

    try:
        audio_filepath = os.path.join(AUDIO_PATH, filename)
        if not os.path.exists(audio_filepath):
            raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없습니다.")
        
        matching_row = point[point["filename"] == filename]
        if matching_row.empty:
            raise HTTPException(status_code=404, detail="위도 및 경도 정보를 찾을 수 없습니다.")
        
        latitude = matching_row.iloc[0]["위도"]
        longitude = matching_row.iloc[0]["경도"]

        text_result = A2T.audio_to_text(AUDIO_PATH, filename)
        
        if not text_result:
            raise HTTPException(status_code=500, detail="오디오 처리 실패")

        summary = A2T.text_summary(text_result)
        model_result = Bert.test(summary['content'])
    
        if not model_result["emergency"]:
            return{"message": "응급 상황이 아닌 것으로 판단됨"}
    
        hospitals = Recog.recommend_hospital(latitude, longitude)\

        if not hospitals:
            return {"message": "응급 상황으로 판단되었지만, 추천할 병원이 없습니다.",
                    "model_result": model_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")

    return {
        "summary": summary["content"],
        "keyword": summary["keyword"],
        "danger_level": model_result["class_name"],
        "audio_latitude": latitude,
        "audio_longitude": longitude,
        "recommended_hospitals": hospitals
    }
    
@app.get("/Hospital2String/{context}/{latitude}/{longitude}")
async def process_pipeline(context: str, latitude: float, longitude: float):

    try:
        summary = A2T.text_summary(context)
        model_result = Bert.test(summary['content'])
    
        if not model_result["emergency"]:
            return{"message": "응급 상황이 아닌 것으로 판단됨"}
    
        hospitals = Recog.recommend_hospital(latitude, longitude)
        if not hospitals:
            return {"message": "응급 상황으로 판단되었지만, 추천할 병원이 없습니다.",
                    "model_result": model_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    
    return {
        "summary": summary["content"],
        "keyword": summary["keyword"],
        "danger_level": model_result["class_name"],
        "audio_latitude": latitude,
        "audio_longitude": longitude,
        "recommended_hospitals": hospitals
    }