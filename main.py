# app.py
from fastapi import FastAPI, HTTPException
import pandas as pd
import os
import openai
import modules_AI_06_19 as md
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# FastAPI 앱 초기화
app = FastAPI()

# 경로 설정
path = "./"  # 프로젝트의 루트 경로
audio_path = os.path.join(path, "audio/")
location_csv = os.path.join(path, "audio_location.csv")
emergency_data_csv = os.path.join(path, "emergen_df.csv")
model_path = os.path.join(path, "fine_tuned_bert/")

# OpenAI 및 기타 API 키 로드
openapi = md.load_api_keys("keys.json")
openai.api_key = openapi["openapi"]
os.environ["OPENAI_API_KEY"] = openai.api_key
service = openapi["service"]
c_id, c_key = openapi["c_id"], openapi["c_key"]

# 모델 및 필요한 객체 초기화
A2T = md.AudioTextProcessor(api_key=openai.api_key)
Bert = md.ModelInstance(loadpath=model_path)
public_data_service = md.PublicData(service=service)
Recog = md.GetDistance(
    csv=emergency_data_csv,
    c_id=c_id,
    c_key=c_key,
    public_data_service=public_data_service
)

# 위치 매핑 데이터 로드
point = pd.read_csv(location_csv)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

@app.get("/test")
def test_endpoint():
    return {"message": "This is a test endpoint. Working fine!!!!!!!!!!!!!!!!!!!!!!"}


@app.get("/fastapi_hospital/{filename}")
async def process_pipeline(filename: str):
    """
    파이프라인 실행:
    1. 오디오 → 텍스트 변환
    2. 텍스트 요약 및 위험도 판단
    3. 병원 추천
    """
    # Step 1: 오디오 파일 경로 확인
    audio_filepath = os.path.join(audio_path, filename)
    if not os.path.exists(audio_filepath):
        raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없습니다.")

    # Step 2: audio_location.csv에서 위도, 경도 찾기
    try:
        matching_row = point[point["filename"] == filename]
        if matching_row.empty:
            raise HTTPException(status_code=404, detail="위도 및 경도 정보를 찾을 수 없습니다.")
        latitude = matching_row.iloc[0]["위도"]
        longitude = matching_row.iloc[0]["경도"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"위치 매핑 처리 실패: {str(e)}")

    # Step 3: 오디오 → 텍스트 변환
    try:
        text_result = A2T.audio_to_text(audio_path, filename)
        if not text_result:
            raise HTTPException(status_code=500, detail="오디오 처리 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오디오 처리 중 오류 발생: {str(e)}")

    # Step 4: 텍스트 요약 및 응급 여부 판단
    try:
        # 텍스트 요약
        summary = A2T.text_summary(text_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 요약 중 오류 발생: {str(e)}")
    
    try:
        # 모델 예측
        model_result = Bert.test(summary['content'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 처리 중 오류 발생: {str(e)}")
    
    # 응급 상황 여부 확인
    if not model_result["emergency"]:
        return{"message": "응급 상황이 아닌 것으로 판단됨"}
    
    # Step 5: 병원 추천 (응급 상황인 경우에만 진행)
    try:
        hospitals = Recog.recommend_hospital(latitude, longitude)
        if not hospitals:
            return {"message": "응급 상황으로 판단되었지만, 추천할 병원이 없습니다.",
                    "model_result": model_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"병원 추천 중 오류 발생: {str(e)}")

    # Step 6: 결과 반환
    return {
        "summary": summary["content"],
        "keyword": summary["keyword"],
        "danger_level": model_result["class_name"],
        "audio_latitude": latitude,
        "audio_longitude": longitude,
        "recommended_hospitals": hospitals
    }