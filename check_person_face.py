# command ex) python check_person_face.py --pretrained pretrained/ram_plus_swin_large_14m.pth --interval 10 
import argparse
import torch
import os
import cv2
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from gtts import gTTS
import pygame
import time
import csv
from transformers import AutoModel, AutoFeatureExtractor
from PIL import Image
from scipy.spatial.distance import cosine
import pandas as pd
from datetime import datetime


# CSV 파일에 헤더 추가 (처음 실행 시, CSV 파일이 없으면 헤더를 추가)
def write_csv_header(csv_file):
    if not os.path.exists(csv_file):  # 파일이 없으면
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Date', 'Time', 'Person Count'])  # CSV 헤더 작성


# CSV에 데이터 추가 (날짜, 시간, 사람 수 기록)
def write_to_csv(csv_file, date, current_time, person_cnt):
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([date, current_time, person_cnt])  # 날짜, 시간, 사람 수 기록


# 음성 파일을 생성하는 함수
def generate_audio_files():
    audio_files = [
        ("hello_a1.mp3", "안녕하세요"),
        ("hello_a2.mp3", "조금만 더 가까이 와 주세요"),
        ("hello_b1.mp3", "어서오세요 여러분!"),
        ("hello_b2.mp3", "차례대로 카메라 앞으로 와 주세요"),
        ("hello_empty.mp3", "지금은 아무도 없습니다."),
        ("ismember.mp3", "저희 회원입니다. 운동하세요"),
        ("nomember.mp3", "미가입 가입 부탁합니다.")
    ]
    
    for filename, text in audio_files:
        if os.path.exists(filename):
            print(f"이미 음성 파일 '{filename}'이(가) 존재합니다.")
        else:
            tts = gTTS(text=text, lang='ko')
            tts.save(filename)
            print(f"음성 파일 '{filename}'이 생성되었습니다.")


# 음성 파일을 재생하는 함수
def play_audio(audio_file):
    pygame.mixer.init()  # pygame 초기화
    pygame.mixer.music.load(audio_file)  # 음성 파일 로드
    pygame.mixer.music.play()  # 음성 재생
    while pygame.mixer.music.get_busy():  # 음성 재생이 끝날 때까지 대기
        pygame.time.Clock().tick(10)
    print(f"음성 '{audio_file}' 재생이 끝났습니다.")


# 웹캠 입력을 처리하고 사람 수 계산 후, 얼굴 인증을 진행하는 함수
def process_webcam_input(model, device, transform, interval, csv_file, face_model, feature_extractor, reference_embeddings, reference_names):

    cap = cv2.VideoCapture(0) # 웹캠 열기
    if not cap.isOpened():  # 웹캠이 열리지 않으면 종료
        print("웹캠을 열 수 없습니다.")
        return
    
    last_processed_time = 0  # 마지막 처리 시간
    person_cnt = 0  # 초기 사람 수
    
    while True:
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:  # 프레임을 읽지 못하면 종료
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break

        current_time = cv2.getTickCount() / cv2.getTickFrequency()  # 현재 시간 계산

        if current_time - last_processed_time > interval:  # interval 초마다 처리
            last_processed_time = current_time
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # BGR을 RGB로 변환
            input_tensor = transform(image).unsqueeze(0).to(device)  # 이미지 변환 및 텐서화
            res = inference(input_tensor, model)  # 모델 추론
            tags = res[0].split(' | ')  # 태그들 분리

            person_cnt = 0  # 사람 수 초기화
            for word in tags:
                if word.lower() in ['man', 'woman', 'girl', 'boy']:  # 사람 관련 태그에 따라 사람 수 증가
                    person_cnt += 1

            # 사람 수에 따라 음성 파일을 재생
            if person_cnt == 0:
                play_audio("hello_empty.mp3")
            elif person_cnt == 1:
                play_audio("hello_a1.mp3")
                play_audio("hello_a2.mp3")
            elif person_cnt > 1 and person_cnt <= 3:
                play_audio("hello_b1.mp3")
                play_audio("hello_b2.mp3")

            timestamp = time.strftime("%Y%m%d_%H%M%S")  # 타임스탬프 생성
            img_filename = f"captured_image_{timestamp}.jpg"  # 이미지 파일 이름 생성
            cv2.imwrite(img_filename, frame)  # 이미지 저장
            print(f"이미지가 저장되었습니다: {img_filename}")

            # 사람 수를 CSV에 기록
            date = time.strftime("%Y-%m-%d")
            current_time_str = time.strftime("%H:%M:%S")
            write_to_csv(csv_file, date, current_time_str, person_cnt)

            # 사람 수가 0보다 크면 얼굴 인증 시작
            if person_cnt > 0:
                print(f"{person_cnt}명의 얼굴 인증을 시작합니다.")
                for _ in range(person_cnt):  # 사람 수 만큼 얼굴 인증 수행
                    face_recognition_result = recognize_faces(face_model, feature_extractor, reference_embeddings, reference_names)

                    if face_recognition_result is None:
                        print("비회원입니다")
                        play_audio("nomember.mp3")
                    else:
                        print(f"회원입니다: {face_recognition_result}")
                        play_audio("ismember.mp3")

            # 20초 간격 후 다시 사람 수를 인식하도록 반복
            time.sleep(interval)

        cv2.imshow("Webcam", frame)  # 웹캠 화면 출력

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 눌러 종료
            break

    cap.release()  # 웹캠 종료
    cv2.destroyAllWindows()


# 얼굴 모델 로드 함수
def load_face_model():
    model_name = "google/vit-base-patch16-224-in21k"
    try:
        print("모델을 불러오는 중...")
        model = AutoModel.from_pretrained(model_name)  # 모델 로드
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)  # 특징 추출기 로드
        print("모델이 성공적으로 로드되었습니다.")
        return model, feature_extractor
    except Exception as e:
        print(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
        exit()


# 얼굴 이미지를 임베딩으로 변환하는 함수
def get_face_embedding(model, feature_extractor, image):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 이미지를 RGB로 변환
        inputs = feature_extractor(images=pil_image, return_tensors="pt")  # 입력 처리
        outputs = model(**inputs)  # 모델 추론
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()  # 임베딩 추출
        return embeddings
    except Exception as e:
        print(f"임베딩을 가져오는 중 오류가 발생했습니다: {e}")
        return None


# 얼굴 인식 및 유사도 계산
def recognize_faces(face_model, feature_extractor, reference_embeddings, reference_names):
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    if not cap.isOpened():  # 웹캠이 열리지 않으면 종료
        print("카메라를 열 수 없습니다.")
        return None

    print("얼굴을 인식 중입니다. ESC 키를 눌러 종료합니다.")
    start_time = time.time()  # 인증 시작 시간 기록

    while True:
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            print("카메라에서 이미지를 가져올 수 없습니다.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # 얼굴 인식기
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100))  # 얼굴 인식

        for (x, y, w, h) in faces:  # 인식된 얼굴에 대해 처리
            face = frame[y:y + h, x:x + w]  # 얼굴 영역 추출
            embedding = get_face_embedding(face_model, feature_extractor, face)  # 얼굴 임베딩 추출
            if embedding is not None:
                similarities = [calculate_similarity(embedding, ref_emb) for ref_emb in reference_embeddings]  # 유사도 계산
                max_similarity = max(similarities)  # 최대 유사도 추출
                best_match_index = similarities.index(max_similarity)  # 가장 유사한 사람 인덱스
                name = reference_names[best_match_index] if max_similarity > 0.70 else "Unknown"  # 유사도가 70% 이상이면 인증

                # 30초 시간 동안 인증 시도
                if time.time() - start_time > 30:
                    print("인증 시간 초과")
                    cap.release()
                    return None
                
                if name != "Unknown":
                    cap.release()
                    return name

        cv2.imshow("Real-Time Face Recognition", frame)  # 실시간 얼굴 인식 화면 출력

        if cv2.waitKey(1) & 0xFF == 27:  # ESC로 종료
            break

    cap.release()  # 웹캠 종료
    cv2.destroyAllWindows()

    return None


# 이미지 로드 (경로가 잘못되었을 때 경고 출력)
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:  # 이미지 로드 실패
        print(f"이미지 파일을 로드할 수 없습니다: {image_path}")
        return None
    return image


# 두 임베딩의 유사도를 계산하는 함수
def calculate_similarity(embedding1, embedding2):
    try:
        return 1 - cosine(embedding1, embedding2)  # 코사인 유사도 계산
    except Exception as e:
        print(f"유사도 계산 중 오류가 발생했습니다: {e}")
        return 0


# 메인 함수
def main():
    # CSV 파일에 헤더 추가
    write_csv_header("captured_data.csv")

    # 음성 파일 생성
    generate_audio_files()

    # 웹캠 처리 및 사람 수 계산
    args = argparse.Namespace(interval=20, pretrained='pretrained/ram_plus_swin_large_14m.pth', image_size=384)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(image_size=args.image_size)

    model = ram_plus(pretrained=args.pretrained, image_size=args.image_size, vit='swin_l')
    model.eval()  # 모델을 평가 모드로 전환
    model = model.to(device)  # 모델을 적절한 장치로 이동

    # 얼굴 인식 모델 로드
    face_model, feature_extractor = load_face_model()

    # 얼굴 이미지에 대한 임베딩 계산
    reference_images = [
        r"C:\Users\15\ram310\recognize-anything\images\member_faces\Kim Songmin.jpg",
        r"C:\Users\15\ram310\recognize-anything\images\member_faces\Kim Youngho.jpg",
        r"C:\Users\15\ram310\recognize-anything\images\member_faces\Lee Jeak.jpg",
        r"C:\Users\15\ram310\recognize-anything\images\member_faces\Umida.jpg"
    ]
    reference_names = ["Kim Songmin", "Kim Youngho", "Lee Jeak", "Umida"]

    reference_embeddings = []
    for image_path, name in zip(reference_images, reference_names):
        image = load_image(image_path)
        if image is not None:
            embedding = get_face_embedding(face_model, feature_extractor, image)
            if embedding is not None:
                reference_embeddings.append(embedding)

    # 사람 수 계산 및 음성 출력 및 얼굴 인증 처리
    process_webcam_input(model, device, transform, args.interval, "captured_data.csv", face_model, feature_extractor, reference_embeddings, reference_names)

if __name__ == "__main__":
    main()
