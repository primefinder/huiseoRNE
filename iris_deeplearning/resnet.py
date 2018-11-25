import tensorflow as tf
import numpy as np
import cv2
import sys
import os

'''
program by Kyutae Kim
special thanks for my team members, Huiseo Kim, Jaewoo Lee

Resnet with full pre-activation

bn->relu->weight->bn->relu->weight->addition
'''
imgfile=[]
imgans0=[]
imgans1=[]
imgdata=[]

def schfile(dirname):
    global imgfile
    global imgans0
    global imgans1
    filenames=os.listdir(dirname)
    num=0
    for filename in filenames:
        tempfile=filename.split()
        tempans=[]
        if tempfile[1][0:1]=='M':
            tempans.append(0)
        else:
            tempans.append(1)
        if tempfile[1][2:3] not in ['1','2','3','4','5','6','7','8','9','0']:
            templist=list(tempfile[1])
            templist[2]='0'
            tempfile[1]=''.join(templist)
        tempans.append(int(tempfile[1][1:3])//20)
        tpans0=[int(0),int(0)]
        tpans1=[]
        for num1 in range(5):
            tpans1.append(int(0))
        tpans0[tempans[0]]=int(1)
        tpans1[tempans[1]]=int(1)
        imgans0.append(tpans0)
        imgans1.append(tpans1)
        imgans0.append(tpans0)
        imgans1.append(tpans1)
        imgt=cv2.imread(dirname+filename)
        rows, cols=imgt.shape[:2]
        imgt=cv2.cvtColor(imgt, cv2.COLOR_BGR2GRAY)
        imgfile.append(imgt)

                
schfile("./iris/")
rows, cols=imgfile[0].shape[:2]
imgsizx=128
imgsizy=128
print(rows, cols)
num=0
for img1 in imgfile:
    img3=img1[189:1723, 177:2302]
    img4=img1[1784:3318, 177:2302]
    rows1, cols1=img3.shape[:2]
    rows2, cols2=img4.shape[:2]
    if rows1!=1534 or cols1!=2125 or rows2!=1534 or cols2!=2125:
        del imgans0[num*2]
        del imgans0[num*2]
    else:
        num=num+1
        img3=cv2.resize(img3, (imgsizx, imgsizy), interpolation=cv2.INTER_AREA)
        img4=cv2.resize(img4, (imgsizx, imgsizy), interpolation=cv2.INTER_AREA)
        img3=np.reshape(img3, (imgsizx, imgsizy, 1))
        img4=np.reshape(img4, (imgsizx, imgsizy, 1))
        imgdata.append(img3)
        imgdata.append(img4)

for num1 in range(len(imgans0)):
    imgans0[num1]=np.array(imgans0[num1])
for num2 in range(len(imgans1)):
    imgans1[num2]=np.array(imgans1[num2])
    
imgans0=np.array(imgans0)
imgans1=np.array(imgans1)

imgdata=np.asarray(imgdata)

savefile=open("result.txt", "w")

trX=imgdata[0:150]
teX=imgdata[150:]
trY=imgans1[0:150]
teY=imgans1[150:]

num=0
batch_size=40
test_size=40

X=tf.placeholder("float", [None, imgsizx, imgsizy, 1])
Y=tf.placeholder("float", [None, 5])

w=[]
siz=[6,6,6]

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

w.append(init_weights([3, 3, 1, 16]))#0

for num1 in range(0, siz[0]):
    w.append(init_weights([3, 3, 16, 16]))#1~6
w.append(init_weights([3, 3, 16, 32]))#7
for num1 in range(0, siz[1]):
    w.append(init_weights([3, 3, 32, 32]))#8~13
w.append(init_weights([3, 3, 32, 64]))#14
for num1 in range(0, siz[2]):
    w.append(init_weights([3, 3, 64, 64]))#15~20
w.append(init_weights([imgsizx*imgsizy*64, 5]))#21

def layer(inlayer, w1, w2, chn):
    global imgsizx, imgsizy
    epsilon=1e-3
    batch_mean1, batch_var1=tf.nn.moments(inlayer, [0])
    scale1=tf.Variable(tf.ones([imgsizx, imgsizy, chn]))
    beta1=tf.Variable(tf.zeros([imgsizx, imgsizy, chn]))
    la1=tf.nn.batch_normalization(inlayer, batch_mean1, batch_var1, beta1, scale1, epsilon)
    la2=tf.nn.relu(la1)
    la3=tf.nn.conv2d(la2, w1, strides=[1,1,1,1], padding='SAME')
    batch_mean2, batch_var2=tf.nn.moments(la3, [0])
    scale2=tf.Variable(tf.ones([imgsizx, imgsizy, chn]))
    beta2=tf.Variable(tf.zeros([imgsizx, imgsizy, chn]))
    la4=tf.nn.batch_normalization(inlayer, batch_mean2, batch_var2, beta2, scale2, epsilon)
    la5=tf.nn.relu(la4)
    la6=tf.nn.conv2d(la5, w2, strides=[1,1,1,1], padding='SAME')
    la7=inlayer+la6
    return la7

def model(inlayer, w, siz):
    la1=tf.nn.conv2d(inlayer, w[0], strides=[1,1,1,1], padding='SAME')
    la2=[]
    la4=[]
    la6=[]
    for num1 in range(siz[0]//2):
        if num1==0:
            la2.append(layer(la1, w[num1*2+1], w[num1*2+2], 16))
        else:
            la2.append(layer(la2[num1-1], w[num1*2+1], w[num1*2+2], 16))
    la3=tf.nn.conv2d(la2[siz[0]//2-1], w[siz[0]+1], strides=[1,1,1,1], padding='SAME')
    for num1 in range(siz[1]//2):
        if num1==0:
            la4.append(layer(la3, w[siz[0]+num1*2+2], w[siz[0]+num1*2+3], 32))
        else:
            la4.append(layer(la4[num1-1], w[siz[0]+num1*2+2], w[siz[0]+num1*2+3], 32))
    la5=tf.nn.conv2d(la4[siz[1]//2-1], w[siz[0]+siz[1]+2], strides=[1,1,1,1], padding='SAME')
    for num1 in range(siz[2]//2):
        if num1==0:
            la6.append(layer(la5, w[siz[0]+siz[1]+num1*2+3], w[siz[0]+siz[1]+num1*2+4], 64))
        else:
            la6.append(layer(la6[num1-1], w[siz[0]+siz[1]+num1*2+3], w[siz[0]+siz[1]+num1*2+4], 64))
    la7=tf.reshape(la6[siz[2]//2-1], [-1, w[siz[0]+siz[1]+siz[2]+3].get_shape().as_list()[0]])
    pyx=tf.matmul(la7, w[siz[0]+siz[1]+siz[2]+3])
    return pyx


py_x = model(X, w, siz)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.02, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.initialize_all_variables().run()


    for i in range(1000):
        print(i)
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        accuracy=np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         Y: teY[test_indices]
                                                        }))
        print(accuracy)
        savefile.write(str(accuracy))
        savefile.write("\n")
    savefile.close()

    
