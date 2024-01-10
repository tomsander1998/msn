import os
import numpy as np
l = []
n = 10
path = "/datasets01/imagenet_full_size/061417/train"

for dir in os.listdir(path):
    for i in range(n):
        for img_name in os.listdir(path+"/"+dir):
            l.append(img_name)

with open("subset10.txt", "w") as output:
    for img_name in l:
        output.write(img_name+ '\n')