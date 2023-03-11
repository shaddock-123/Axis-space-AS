'''import serial # 导入模块
#接收arduino数据
ser = serial.Serial('COM12',9600,timeout=1)
# serial.Serial  的三个形参 分别对应 Arduino的串口  波特率 连接超时时间
print(ser)
while 1:
    val = ser.readline().decode('utf-8')
    # ser.readline() 读取窗串口中的数据以二进制的形式展示需要使用.decode('utf-8')进行解码
    print(val)

    parsed = val.split(',')
    parsed = [x.strip() for x in parsed]
    print(parsed)
'''  
#python向arduino发送数据
import serial
import time
import numpy as np

ser = serial.Serial('COM8',115200,timeout=1)
action =1
while 1:
    val = ser.write(str(action).encode('utf-8'))
    print("I'm writing")
    #ser.write(发送的数据需要进行编码.encode('utf-8'))
    time.sleep(0.4)
    val2 = ser.readline().decode('utf-8')
    print("I am reading")
    print(val2)
