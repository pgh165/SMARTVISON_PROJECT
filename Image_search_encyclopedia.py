import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import sys
import wikipediaapi
import time



def wiki(class_name):
    wiki = wikipediaapi.Wikipedia( 
        language='ko', 
        extract_format=wikipediaapi.ExtractFormat.WIKI)

    p_wiki = wiki.page(class_name)

    if(p_wiki.exists()):
        print(p_wiki.text)
    else:
        print("not exist pages")
        
filename = "melon.jpg"
model = "frozen_model.pb"

classNames = None
with open('labels.txt', 'rt', encoding = "UTF-8") as f:
    classNames = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNet(model)
if net is None:
    print("Network load failed!") 
    sys.exit()
    

img = cv2.imread(filename)
if img is None:
    print("Image load failed!")
    sys.exit()
    
img_rs = cv2.resize(img, (500,500), interpolation=cv2.INTER_LINEAR)

blob = cv2.dnn.blobFromImage(img_rs, 1.0/127.5, (224,224), (-1,-1,-1))
net.setInput(blob)

start = time.time()
prob = net.forward()
end = time.time()

fps = 1./(end - start)#fps 계산

result =  prob.flatten()#prob객체를 1차원배열로 평탄화
classId = np.argmax(result)
confidence = result[classId]

wiki(classNames[classId])#클래스명 검색
print(f'FPS={fps:.1f}')#fps 소수점 한자리까지 출력

pil_image = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(pil_image)

fontpath = "C:\\Windows\\Fonts\\H2GTRM.TTF"
font = ImageFont.truetype(fontpath, 30)
draw = ImageDraw.Draw(pil_image)

#이미지에 클래스와 정확도 표시
draw.text((10, 30), (f'{classNames[classId]} ({confidence * 100:4.2F}%)'), (255,0,0,255), font=font)

img = np.array(pil_image)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow(filename, img)
cv2.waitKey()