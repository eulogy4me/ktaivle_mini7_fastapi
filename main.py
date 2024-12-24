# app.py
from fastapi import FastAPI, HTTPException
import pandas as pd
import os
import openai
import modules_AI_06_19 as md
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json


app = FastAPI()

BASE_PATH = "./"
AUDIO_PATH = os.path.join(BASE_PATH, "audio/")
LOCATION_CSV = os.path.join(BASE_PATH, "audio_location.csv")
EMERGENCY_DATA_CSV = os.path.join(BASE_PATH, "emergen_df.csv")
MODEL_PATH = os.path.join(BASE_PATH, "fine_tuned_bert/")

def load_api_keys(filename):
    with open(filename, 'r') as file:
        keys = json.load(file)
    return keys
openapi=load_api_keys("keys.json")

openai.api_key = openapi['openapi']
service = openapi['service']
c_id, c_key = openapi['c_id'], openapi['c_key']

os.environ['OPENAI_API_KEY'] = openai.api_key
os.environ['SERVICE_KEY'] = service
os.environ['CLIENT_ID'] = c_id
os.environ['CLIENT_SECRET'] = c_key

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
async def process_pipeline(filename: str,recog_num: int):

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
    
        hospitals = Recog.recommend_hospital(latitude, longitude, recog_num)

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
        "recog_number":recog_num,
        "recommended_hospitals": hospitals
    }
    
@app.get("/Hospital2String/{context}/{latitude}/{longitude}")
async def process_pipeline(context: str, latitude: float, longitude: float, number: int):

    try:
        summary = A2T.text_summary(context)
        model_result = Bert.test(summary['content'])
    
        if not model_result["emergency"]:
            return{"message": "응급 상황이 아닌 것으로 판단됨"}
    
        hospitals = Recog.recommend_hospital(latitude, longitude, number)
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
        "recog_number":number,
        "recommended_hospitals": hospitals
    }
    
    
    # uvicorn main:app --reload 
    # merged_audio숫자.wav 
    #