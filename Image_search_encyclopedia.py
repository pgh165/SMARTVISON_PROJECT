#모듈가져오기
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import sys
import wikipediaapi
import time


#백과사전 검색 함수
def wiki(class_name):
    wiki = wikipediaapi.Wikipedia( 
        language='ko', #한글사이트 설정
        extract_format=wikipediaapi.ExtractFormat.WIKI)#백과사전 데이터 추출형식 설정

    p_wiki = wiki.page(class_name)#검색된 위키페이지 저장

    if(p_wiki.exists()):#위키페이지가 존재할 경우
        print(p_wiki.text)#페이지 내용 출력
    else:
        print("not exist pages")#오류문구 출력
        sys.exit()#프로그램 종료
        
filename = "image.jpg"#이미지 파일 경로 설정
model = "frozen_model.pb"#모델파일 경로 설정

classNames = None#문자열을 저장할 변수 설정
with open('labels.txt', 'rt', encoding = "UTF-8") as f:#클래스명이 저장된 파일 텍스트모드로 열기
    classNames = f.read().rstrip('\n').split('\n')#문자열의 오른쪽 공백제거 및 공백을 기준으로 문자열 분리한 리스트 반환

net = cv2.dnn.readNet(model)#모델 불러오기
if net is None:#모델파일이 없을 경우
    print("Network load failed!")#에러문구 출력
    sys.exit()#프로그램 종료
    

img = cv2.imread(filename)#이미지 불러오기
if img is None:#이미지파일이 없을 경우
    print("Image load failed!")#에러문구 출력
    sys.exit()#프로그램 종료
    
img_rs = cv2.resize(img, (500,500), interpolation=cv2.INTER_LINEAR)#원본 이미지 크기변환

blob = cv2.dnn.blobFromImage(img_rs, 1.0/127.5, (224,224), (-1,-1,-1))#크기를 변경한 이미지를 blob객체로 생성
net.setInput(blob)#blob객체를 입력으로 설정

start = time.time()#시작시간
prob = net.forward()#추론수행
end = time.time()#종료시간

fps = 1./(end - start)#fps 계산

result =  prob.flatten()#prob객체를 1차원배열로 평탄화
classId = np.argmax(result)#결과배열의 최대값 위치저장
confidence = result[classId]#최대값저장

wiki(classNames[classId])#클래스를 백과사전에 검색하고 결과 출력
print(f'FPS={fps:.1f}')#fps 소수점 한자리까지 출력

pil_image = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)#BGR->RGB 변환
pil_image = Image.fromarray(pil_image)#imread로 읽은 이미지를 pillow 이미지로 변환

fontpath = "C:\\Windows\\Fonts\\H2GTRM.TTF"#폰트파일 위치저장
font = ImageFont.truetype(fontpath, 30)#폰트크기와 폰트설정
draw = ImageDraw.Draw(pil_image)#

#이미지에 클래스와 정확도 표시
draw.text((10, 30), (f'{classNames[classId]} ({confidence * 100:4.2F}%)'), (255,0,0,255), font=font)

img = np.array(pil_image)#pillow 이미지를 cv2 이미지로 변환
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)#RGB->BGR 변환
cv2.imshow(filename, img)#결과 이미지 출력
cv2.waitKey()#키입력까지 대기
