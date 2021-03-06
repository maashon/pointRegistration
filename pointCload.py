# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b2fjU0z4grUxyKMmk_rJ6YZkmqh_iF0G
"""

from google.colab import drive
drive.mount('/content/drive')

base='/content/drive/MyDrive/colab/code/3DFS/complete/'
brokenDir='/content/drive/MyDrive/colab/code/3DFS/broken/'
name='Baby.xyz'
completePointCloudName=base+name
p=open(completePointCloudName,'r')
p1=open(brokenDir+'p1_'+name,'w')
p2=open(brokenDir+'p2_'+name,'w')
points=[]
for line in p.readlines():
  point=line.split()  
  try:
    x=float(point[0])
    y=float(point[1])
    z=float(point[2])
    l={'x':x,'y':y,'z':z}
  except:
    print(point,'has some problem\n')
  points.append(l)
 


  #find the threshold

sumX,sumY,sumZ=0,0,0

for point in points:
  sumX+=point['x']
  sumY+=point['y']
  sumZ+=point['z']

sumX/=len(points)
sumY/=len(points)
sumZ/=len(points)

tr=60
inter=0
for point in points:
  if point['y']<=sumY+tr:
    p1.write(str(point['x'])+' '+str(point['y'])+' '+str(point['z'])+'\n')
    if point['y']>=sumY-tr:
      inter+=1
  if point['y']>=sumY-tr:
    p2.write(str(point['x'])+' '+str(point['y'])+' '+str(point['z'])+'\n')


p.close()
p1.close()
p2.close()
print(f'sumx is {sumX} and sumy is {sumY} and sumz is {sumZ} ')

#find the correspondence
p1=open(brokenDir+'p1_'+name,'r')
p2=open(brokenDir+'p2_'+name,'r')
point1=[]
point2=[]
for point in p1.readlines():
  p=point.split()  
  point1.append({'x':float(p[0]),'y':float(p[1]),'z':float(p[2])})


for point in p2.readlines():
  p=point.split()  
  point2.append({'x':float(p[0]),'y':float(p[1]),'z':float(p[2])})

print(f'intersectoin {inter*100/len(point1)}\n')

'''
overlaps=0
for i in point1:
  for j in point2:
    if i['x']==j['x']:
      if i['y']==j['y']:
        if i['z']==j['z']:
          overlaps+=1



print(f'in {len(point1)} points, {overlaps} overlaps were found:{overlaps*100/len(point1)} precent')
'''

## txt XYZRGB to XYZ
baseD='/content/drive/MyDrive/colab/code/3DFS/complete/'
dataName='Baby.txt'
rgb=open(baseD+dataName,'r')
xyz=open(baseD+(dataName.split('.')[0])+".xyz","w")


for line in rgb.readlines():
  txt=line.split();
  xyz.write(txt[0]+' '+txt[1]+' '+txt[2]+'\n')


rgb.close()
xyz.close()