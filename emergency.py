
import os
import requests
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import openai
from openai import OpenAI
import json
import torch


# 0. load key file------------------
def load_file(filepath):
    with open(filepath, 'r') as file:
        return file.readline().strip()


# 1-1 audio2text--------------------
def audio_to_text(audio_path, filename):
    client = OpenAI()
    audio_file = open(audio_path + filename, "rb")
    transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    language="ko",
                    response_format="text",
                    temperature=0.0)
    return transcript


# 1-2 text2summary------------------
def text_summary(input_text):
    # OpenAI 클라이언트 생성
    client = OpenAI()

    system_role = '''당신은 응급상황에 대한 텍스트에서 핵심 내용을 훌륭하게 요약해주는 어시스턴트입니다.
    응답은 다음의 형식을 지켜주세요.
    {"summary": \"텍스트 요약\",
    "keyword" : \"핵심 키워드(3가지)\"}
    '''
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_role
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
    )
    answer = response.choices[0].message.content
    parsed_answer = json.loads(answer)

    summary = parsed_answer["summary"]
    keyword = parsed_answer["keyword"]

    return summary + ', ' + keyword

# 2. model prediction------------------
def predict(text, model, tokenizer):
    # 입력 문장 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value for key, value in inputs.items()}  # 각 텐서를 GPU로 이동

    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)

    # 로짓을 소프트맥스로 변환하여 확률 계산
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)

    # 가장 높은 확률을 가진 클래스 선택
    pred = torch.argmax(probabilities, dim=-1).item()

    return pred, probabilities



# 3-1. get_distance------------------
def get_dist(start_lat, start_lng, dest_lat, dest_lng, c_id, c_key):
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": c_id,
        "X-NCP-APIGW-API-KEY": c_key,
    }
    params = {
        "start": f"{start_lng},{start_lat}",  # 출발지 (경도, 위도)
        "goal": f"{dest_lng},{dest_lat}",    # 목적지 (경도, 위도)
        "option": "trafast"  # 실시간 빠른 길 옵션
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    # 'route'와 'trafast' 키가 존재하는지 확인하고 예외 처리
    try:
        dist = data['route']['trafast'][0]['summary']['distance']  # m(미터)
        dist = dist / 1000  # km로 변환
    except KeyError as e:
        print(f"응답 데이터에서 예상되는 키를 찾을 수 없음: {e}")
        return None

    return dist

# 3-2. recommendation------------------
def recommend_hospital3(path, emergency, start_lat, start_lng, c_id, c_key):

    # 위도 경도에 ± 0.5 범위에서 먼저 조회
    temp = emergency.loc[emergency['위도'].between(start_lat-0.05, start_lat+0.05) & emergency['경도'].between(start_lng-0.05, start_lng+0.05)].copy()
    # display(temp)

    # 거리 계산
    temp['거리'] = temp.apply(lambda x: get_dist(start_lat, start_lng, x['위도'], x['경도'], c_id, c_key), axis=1)
    return temp.sort_values(by='거리').head(3)
