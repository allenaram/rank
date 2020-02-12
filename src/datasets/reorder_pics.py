import numpy

filename = [line.rstrip('\n') for line in open("E:/database/tid2013/mos_with_names.txt")]
nameScoreMap = {}
for i in filename:
    nameScoreMap[i.split()[1].lower()] = float(i.split()[0])

newList = []
for k in range(9): #失真类型
    for i in range(5):#失真等级
        for j in range(25):#图片数
            K = k+1
            I = i+1
            J = j+1
            name = 'i'+ ('0'+str(J) if (J<10) else str(J))
            name += ("_" + ('0'+str(K) if (K<10) else str(K)))
            name += ("_" + str(I) + ".bmp")
            newList.append(str(nameScoreMap[name]) + " " + name)

file = open('file_name.txt','w')
for line in newList:
    file.write(line+'\n')



