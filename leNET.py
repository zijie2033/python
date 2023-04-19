import numpy as np #矩陣運算
import time as t
from math import floor #floor:往下取
from math import ceil
np.set_printoptions(threshold=np.inf)#inf為無窮大的浮點數
np.set_printoptions(suppress=True)#抑制顯示小數位數
'''Layer1
Layer: CNN
input:32x32x1
filter:5x5x6

stride:1
pad:0
output:28x28x6
activation: tanh
'''
### 32x32x1 image input feature map ###
ifmap = np.random.randint(1,256,size=(1,32,32),dtype=np.int_)#隨機整數
print("\n===========input feature map===========\n")
#print(ifmap)
### 5x5x6 convolution kernel ###
filter=np.random.rand(6,1,5,5)#隨機[0,1)間小數
stride=1
pad=0
### 28x28x6 output feature maps ###
ofmap=np.zeros((filter.shape[0],floor((ifmap.shape[1]+2*pad-filter.shape[2])/stride)+1,floor((ifmap.shape[2]+2*pad-filter.shape[3])/stride)+1))

start=t.time()
print("\n===========Start Layer 1 CNN computation===========\n")
for depth in range(ofmap.shape[0]):#depth=6,height=28,width=28,channel=1,i=5,j=5
    for height in range(ofmap.shape[1]):
        for width in range(ofmap.shape[2]):
            for channel in range(filter.shape[1]):
                for i in range(filter.shape[2]):
                    for j in range(filter.shape[3]):
                        ofmap[depth][height][width]+=ifmap[channel][i+height*stride][j+width*stride]*filter[depth][channel][i][j]

#6x28x28=6x28x28+1x32x32*6x1x5x5

### activation function ###
for depth in range(ofmap.shape[0]):
    for height in range(ofmap.shape[1]):
        for width in range(ofmap.shape[2]):
            ofmap[depth][height][width]=2/(1+np.exp(-2*ofmap[depth][height][width]))-1
ifmap=ofmap
#print(ifmap)
''' Layer2
Layer: pooling
input:28x28x6
filter:2x2x6
stride:2
pad:0
output:14x14x6
'''
### 2x2x6 ###
filter=np.ones((6,1,2,2))
stride=2
pad=0
###14x14x6###
ofmap=np.zeros((filter.shape[0],ceil((ifmap.shape[1]+2*pad-filter.shape[2])/stride)+1,ceil((ifmap.shape[2]+2*pad-filter.shape[3])/stride)+1))
print("\n===========Start Layer 2 pooling computation===========\n")
for depth in range(ofmap.shape[0]):#depth=6,height=14,width=14,channel=1,i=2,j=2
    for height in range(ofmap.shape[1]):
        for width in range(ofmap.shape[2]):
            for channel in range(filter.shape[1]):
                for i in range(filter.shape[2]):
                    for j in range(filter.shape[3]):
                        ofmap[depth][height][width]+=ifmap[channel][i+height*stride][j+width*stride]*filter[depth][channel][i][j]
# average pooling
ifmap=ofmap/(filter.shape[2]*filter.shape[3])
#print(ofmap)

'''Layer3
Layer: CNN
input:14x14x6
filter:5x5x16
stride:1
pad:0
output:10x10x16
activation: tanh
'''
### 5x5x16 convolution kernel ###
filter=np.random.rand(16,1,5,5)#隨機[0,1)間小數
stride=1
pad=0
### 10x10x16 output feature maps ###
ofmap=np.zeros((filter.shape[0],floor((ifmap.shape[1]+2*pad-filter.shape[2])/stride)+1,floor((ifmap.shape[2]+2*pad-filter.shape[3])/stride)+1))

print("\n===========Start Layer 3 CNN computation===========\n")
for depth in range(ofmap.shape[0]):#depth=16,height=10,width=10,channel=1,i=5,j=5
    for height in range(ofmap.shape[1]):
        for width in range(ofmap.shape[2]):
            for channel in range(filter.shape[1]):
                for i in range(filter.shape[2]):
                    for j in range(filter.shape[3]):
                        ofmap[depth][height][width]+=ifmap[channel][i+height*stride][j+width*stride]*filter[depth][channel][i][j]


### activation function ###
for depth in range(ofmap.shape[0]):
    for height in range(ofmap.shape[1]):
        for width in range(ofmap.shape[2]):
            ofmap[depth][height][width]=2/(1+np.exp(-2*ofmap[depth][height][width]))-1
ifmap=ofmap
#print(ifmap)

''' Layer4
Layer: pooling
input:10x10x16
filter:2x2x16
stride:2
pad:0
output:5x5x16
'''

### 2x2x16 ###
filter=np.ones((16,1,2,2))
stride=2
pad=0
###5x5x16###
ofmap=np.zeros((filter.shape[0],ceil((ifmap.shape[1]+2*pad-filter.shape[2])/stride)+1,ceil((ifmap.shape[2]+2*pad-filter.shape[3])/stride)+1))
print("\n===========Start Layer 4 pooling computation===========\n")
for depth in range(ofmap.shape[0]):#depth=16,height=5,width=5,channel=1,i=2,j=2
    for height in range(ofmap.shape[1]):
        for width in range(ofmap.shape[2]):
            for channel in range(filter.shape[1]):
                for i in range(filter.shape[2]):
                    for j in range(filter.shape[3]):
                        ofmap[depth][height][width]+=ifmap[channel][i+height*stride][j+width*stride]*filter[depth][channel][i][j]
# average pooling
ifmap=ofmap/(filter.shape[2]*filter.shape[3])
#print(ofmap)

'''Layer5
Layer: Full connection
input:5x5x16
filter:5x5x120
stride:1
pad:0
output:1x1x120
'''


### 5x5x120 convolution kernel ###
filter=np.random.rand(120,1,5,5)#隨機[0,1)間小數
stride=1
pad=0
### 1x1x120 output feature maps ###
ofmap=np.zeros((filter.shape[0],floor((ifmap.shape[1]+2*pad-filter.shape[2])/stride)+1,floor((ifmap.shape[2]+2*pad-filter.shape[3])/stride)+1))

print("\n===========Start Layer 5 Full connnection===========\n")
for depth in range(ofmap.shape[0]):#depth=120,height=1,width=1,channel=1,i=5,j=5
    for height in range(ofmap.shape[1]):
        for width in range(ofmap.shape[2]):
            for channel in range(filter.shape[1]):
                for i in range(filter.shape[2]):
                    for j in range(filter.shape[3]):
                        ofmap[depth][height][width]+=ifmap[channel][i+height*stride][j+width*stride]*filter[depth][channel][i][j]
ifmap=ofmap
### activation function ###
for depth in range(ofmap.shape[0]):
    for height in range(ofmap.shape[1]):
        for width in range(ofmap.shape[2]):
            ofmap[depth][height][width]=2/(1+np.exp(-2*ofmap[depth][height][width]))-1
ifmap=ofmap
#print(ifmap)

'''Layer6
Layer: full connection
input:1x1x120
filter:1x1x84
stride:1
pad:0
output:1x1x84
'''

### 1x1x120 convolution kernel ###
filter=np.random.rand(84,120,1,1)#隨機[0,1)間小數
stride=1
pad=0
### 1x1x120 output feature maps ###
ofmap=np.zeros((filter.shape[0],floor((ifmap.shape[1]+2*pad-filter.shape[2])/stride)+1,floor((ifmap.shape[2]+2*pad-filter.shape[3])/stride)+1))

print("\n===========Start Layer 6 Full connnection===========\n")
for depth in range(ofmap.shape[0]):#depth=16,height=10,width=10,channel=1,i=5,j=5
    for height in range(ofmap.shape[1]):
        for width in range(ofmap.shape[2]):
            for channel in range(filter.shape[1]):
                for i in range(filter.shape[2]):
                    for j in range(filter.shape[3]):
                        ofmap[depth][height][width]+=ifmap[channel][i+height*stride][j+width*stride]*filter[depth][channel][i][j]
ifmap=ofmap
### activation function ###
for depth in range(ofmap.shape[0]):
    for height in range(ofmap.shape[1]):
        for width in range(ofmap.shape[2]):
            ofmap[depth][height][width]=2/(1+np.exp(-2*ofmap[depth][height][width]))-1
ifmap=ofmap
#print(ifmap)

'''Layer7
Layer: output
input:1x1x84
filter:1x1x10
stride:1
pad:0
output:1x1x10
'''
### 1x1x120 convolution kernel ###
filter=np.random.rand(10,84,1,1)#隨機[0,1)間小數
stride=1
pad=0
### 1x1x120 output feature maps ###
ofmap=np.zeros((filter.shape[0],floor((ifmap.shape[1]+2*pad-filter.shape[2])/stride)+1,floor((ifmap.shape[2]+2*pad-filter.shape[3])/stride)+1))

print("\n===========Start Layer 7 Output===========\n")
for depth in range(ofmap.shape[0]):#depth=16,height=10,width=10,channel=1,i=5,j=5
    for height in range(ofmap.shape[1]):
        for width in range(ofmap.shape[2]):
            for channel in range(filter.shape[1]):
                for i in range(filter.shape[2]):
                    for j in range(filter.shape[3]):
                        ofmap[depth][height][width]+=ifmap[channel][i+height*stride][j+width*stride]*filter[depth][channel][i][j]

### softmax ###
ofmap = np.exp(ofmap)/sum(np.exp(ofmap))
print("execution time = ",t.time()-start,"\n")
print(ofmap)
