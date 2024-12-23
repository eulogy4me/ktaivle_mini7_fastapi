import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import openai
import json
import torch
from haversine import haversine
import math
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

from warnings import filterwarnings
filterwarnings('ignore')

import json

# 공공데이터포털 클래스
class PublicData():
    def __init__(self, service) :
        self.service = service
        self.base_url = 'http://apis.data.go.kr/B552657/ErmctInfoInqireService/'
        self.urls = [
            'getEmrrmRltmUsefulSckbdInfoInqire',
            'getSrsillDissAceptncPosblInfoInqire',
            'getEgytListInfoInqire',
            'getEgytLcinfoInqire',
            'getEgytBassInfoInqire',
            'getStrmListInfoInqire',
            'getStrmLcinfoInqire',
            'getStrmBassInfoInqire',
            'getEmrrmSrsillDissMsgInqire'
        ]

    def list_info_inqire(self,path):
        data = []
        url = self.base_url + self.urls[3]
        params = {
            'serviceKey': self.service,
            'pageNo': '1',
            'numOfRows': '1000',
            'format': 'xml'
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            if response.status_code == 200:
                root = ET.fromstring(response.text)

                for item in root.findall('.//item'):
                    data.append({
                        'duty_name': item.findtext('dutyName'),
                        'duty_code': item.findtext('hpid'),
                        'duty_addr': item.findtext('dutyAddr'),
                        'wgs84Lon': item.findtext('wgs84Lon'),
                        'wgs84Lat': item.findtext('wgs84Lat')
                    })
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(e)
        df = pd.DataFrame(data)
        df.to_csv(path)

    def fetch_emrg_bed_info(self, duty_code):
        params = {
            'serviceKey': self.service,
            'hpid': duty_code,
            'pageNo': '1',
            'numOfRows': '10',
            'format': 'xml'
        }
        url = self.base_url + self.urls[0]
        response = requests.get(url, params=params)

        response.raise_for_status()

        if response.status_code == 200:
            root = ET.fromstring(response.text)
            item = root.find('.//body/items/item')
            hvec = int(item.findtext('hvec', default='0'))
            hvidate = item.findtext('hvidate')
            now = datetime.now()
            update_time = datetime.strptime(hvidate, '%Y%m%d%H%M%S')
            valid_time = now - timedelta(minutes=30)
            if update_time < valid_time:
                return 0
            if hvec > 0:
                return int(hvec)
        time.sleep(1)

# TTS, Summary 클래스
class AudioTextProcessor:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.role_description = '''
        당신은 119 긴급전화를 처리하는 AI입니다.
        현재 입력된 텍스트는 급박하고 긴장되는 상황의 발언자가 하는 얘기이므로,
        응급 상황임을 감안하여 발언자의 텍스트를 요약해주세요
        
        아래 입력된 텍스트를 바탕으로 긴급상황 요약을 작성하고, 핵심 키워드 3가지를 JSON 형식으로 반환하세요.

        응답 형식의 예시는 다음과 같습니다:
        {
        "content": "응급상황 요약 내용",
        "keyword": ["키워드1", "키워드2", "키워드3"]        
        }

        위 형식을 반드시 지켜야 합니다.    
        '''
    def audio_to_text(self, audio_path, filename):
        try:
            with open(audio_path + filename, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    language="ko",
                    response_format="text"
                )
                print("Whisper API Response : ", response)
                return response
        except Exception as e:
            return str(e)

    def text_summary(self, input_text):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.role_description},
                    {"role": "user", "content": input_text}
                ]
            )
            response = response.choices[0].message.content
            print("summary response : ", response)
            
            #JSON 파싱
            return json.loads(response)
        except json.JSONDecodeError as e:
            print("JSON parsing error:", e)
            return {"content": "파싱 실패", "keyword": []}
        except Exception as e:
            return str(e)

class BertModel:
    def __init__(self, model_name='klue/bert-base', num_labels=5, batch_size=32, learning_rate=2e-5, epochs=5):
        from sklearn.preprocessing import LabelEncoder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        self.le = LabelEncoder()
        self.trainer = self.initTrainer(batch_size, learning_rate, epochs)

    def preprocess_function(self, data):
        return self.tokenizer(data['text'], truncation=True, padding=True, max_length=128)

    def initData(self, csv):
        from sklearn.model_selection import train_test_split
        from datasets import load_dataset, Dataset
        data = pd.read_csv(csv)
        data['text'] = data['text'].str.replace('"', '').str.replace(',', '')
        data['label'] = self.le.fit_transform(data['label'])
        train, val = train_test_split(data, test_size=0.2, random_state=42)
        train_ds = Dataset.from_pandas(train)
        val_ds = Dataset.from_pandas(val)
        train_ds = train_ds.map(self.preprocess_function, batched=True)
        val_ds = val_ds.map(self.preprocess_function, batched=True)
        return train_ds, val_ds

    def initTrainer(self, batch_size, learning_rate, epochs):
        return Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir='./results',
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.02,
                load_best_model_at_end=True,
                logging_dir='./logs',
                logging_steps=10,
                report_to="tensorboard"
            ),
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    def evaluate(self):
        inputs = self.tokenizer(self.val_ds['text'], return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        self.model = self.model.to(self.device)
        with torch.no_grad():
            probabilities = self.model(**inputs).logits.softmax(dim=1)
        return np.argmax(probabilities.cpu().detach().numpy(), axis=1)

    def predict(self, text):
        inputs = {key: value.to(self.device) for key, value in self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).items()}
        self.model = self.model.to(self.device)
        with torch.no_grad():
            probabilities = self.model(**inputs).logits.softmax(dim=1)
        return torch.argmax(probabilities, dim=-1).item(), probabilities

    def TrainAndEvaluate(self):
        from sklearn.metrics import confusion_matrix, classification_report

        self.trainer.train()
        eval_results = self.trainer.evaluate()
        pred = self.evaluate()
        print(eval_results)
        print(confusion_matrix(self.val_ds['label'], pred))
        print(classification_report(self.val_ds['label'], pred))

    def SaveModel(self, savepath):
        self.model.save_pretrained(savepath)
        self.tokenizer.save_pretrained(savepath)

# Tuned Bert Model 활용 클래스 
class ModelInstance :
    def __init__(self, loadpath):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(loadpath)
        self.tokenizer = AutoTokenizer.from_pretrained(loadpath)

    def predict(self, text):
        inputs = {key: value.to(self.device) for key, value in self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).items()}
        self.model = self.model.to(self.device)
        with torch.no_grad():
            probabilities = self.model(**inputs).logits.softmax(dim=1)
        return torch.argmax(probabilities, dim=-1).item(), probabilities

    def test(self, text):
        predicted_class, probabilities = self.predict(text)
        if (predicted_class > 2):
            print("응급상황이 아닙니다.")
            return {
            "emergency": False,
            "message": "응급상황이 아닙니다.",
            "predicted_class": predicted_class,
            "probabilities": probabilities.tolist()
            }

        print(f"예측된 클래스: {predicted_class}")
        print(f"예측된 위험등급: {predicted_class+1}등급")
        print(f"클래스별 확률: {probabilities}")
        
        return{
            "emergency": True,
            "predicted_class": predicted_class,
            "class_name": f"{predicted_class + 1}등급",
            "probabilities": probabilities.tolist()
        }

# 응급실 관련 클래스
class GetDistance:
    def __init__(self, csv, c_id, c_key, public_data_service):
        self.df = pd.read_csv(csv)
        self.c_id = c_id
        self.c_key = c_key
        self.public_data = public_data_service

    def _filter_nearby_hospitals(self, slat, slon, alpha):
        min_lat, max_lat = slat - alpha, slat + alpha
        min_lon, max_lon = slon - alpha, slon + alpha
        return self.df[
            (self.df['Latitude'] >= min_lat) & (self.df['Latitude'] <= max_lat) &
            (self.df['Longitude'] >= min_lon) & (self.df['Longitude'] <= max_lon)
        ]

    def _calculate_distance(self, slat, slon, dlat, dlon):
        return haversine((slat, slon), (dlat, dlon), unit='km')

    def _get_driving_distance(self, slat, slon, dlat, dlon):
        url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.c_id,
            "X-NCP-APIGW-API-KEY": self.c_key,
        }
        params = {
            "start": f"{slon},{slat}",
            "goal": f"{dlon},{dlat}",
            "option": "trafast"
        }
        response = requests.get(url, headers=headers, params=params)
        return response.json()['route']['trafast'][0]['summary']['distance'] if response.status_code == 200 else math.inf

    def recommend_hospital(self, slat, slon, alpha=0.1):
        # Step1 : Haversine 거리 기반 후보 병원 10개 선정
        filtered_df = self._filter_nearby_hospitals(slat, slon, alpha)

        if filtered_df.empty:
            print("No nearby hospitals found.")
            return []

        try:
            candidates = sorted(
                [(self._calculate_distance(slat, slon, row.Latitude, row.Longitude), row)
                for row in filtered_df.itertuples()],
                key=lambda x: x[0]
            )[:10]
        except AttributeError as e:
            print(f"Error processing hospital data: {e}")
            return []

        # Step 2 : 국립중앙의료원 API로 실시간 데이터 확인
        valid_hospitals = []
        for candidate in candidates:
            duty_code = candidate[1].DutyCode
            bed_count = self.public_data.fetch_emrg_bed_info(duty_code)
            if bed_count > 0:
                valid_hospitals.append((candidate[0], candidate[1]))
                
        if not valid_hospitals:
            print("No hospitals with available beds found.")
            return []
        
        # Step 3 : Naver 지도 API로 주행 거리 계산
        results = []
        for distance, hospital in valid_hospitals:
            driving_distance = self._get_driving_distance(slat, slon, hospital.Latitude, hospital.Longitude)
            if driving_distance < math.inf:
                results.append({
                    "haversine_km": distance,
                    "distance_km": driving_distance / 1000,
                    "hospital_name": hospital.HospitalName,
                    "address": hospital.Address,
                    "emergency_type": hospital.EmergencyType,
                    "phone1": hospital.Phone1,
                    "phone3": hospital.Phone3,
                    "hospital_latitude": hospital.Latitude,
                    "hospital_longitude": hospital.Longitude
                })
        # 주행 거리 기준으로 정렬
        results = sorted(results, key=lambda x: x['distance_km'])[:3]
        return results
 
    
