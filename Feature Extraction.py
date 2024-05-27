import cv2 as cv
import matplotlib.pyplot as plt

pic = cv.imread('Feature Extraction2/images/PData4.png') #silahkan diganti-ganti gambarnya
grayscale = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(grayscale, 200, 255, cv.THRESH_BINARY_INV)

#Mencari kontur objek di dalam citra digital dan menggambar kontur yang ada
countour, hr = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #hasilnya mungkin ada lebih dari satu kontur di satu variabel ini
cv.drawContours(pic, countour, -1, (0,255,0), 2)

#Mencari fitur Shape - Bounding Box dan menggambar bounding box
x,y,w,h = cv.boundingRect(countour[0]) #hanya kontur ke 0 yang diambil
cv.rectangle(pic, (x,y), (x + w, y +h),(255,0,0), 2)

#Mencari fitur 7 Hu invarian Moment dari citra biner (blob image/hasil thresholding)
moment = cv.HuMoments(cv.moments(thresh)) #hasilnya 7 angka. Momen 1-6 invarian terhadap scale dan rotasi, Momen 7 invarian terhadap mirorring

#menampilkan hasil
print(moment)
plt.imshow(thresh, cmap='gray')
plt.show()
plt.imshow(cv.cvtColor(pic, cv.COLOR_BGR2RGB))
plt.show()
