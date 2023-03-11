__author__ = 'duguguang'

from seeker.seekersdk import *
import numpy as np
preFrmNo = 0
curFrmNo = 0

# for test
def py_msg_func(iLogLevel, szLogMessage):
    szLevel = "None"
    if iLogLevel == 4:
        szLevel = "Debug"
    elif iLogLevel == 3:
        szLevel = "Info"
    elif iLogLevel == 2:
        szLevel = "Warning"
    elif iLogLevel == 1:
        szLevel = "Error"
  
    print("[%s] %s" % (szLevel, cast(szLogMessage, c_char_p).value))

def py_data_func(data):
    if data == None:  
        print("Not get the data frame.\n")
        pass
    else:
        frameData = data.contents
        global preFrmNo, curFrmNo 
        curFrmNo = frameData.iFrame
        if curFrmNo != preFrmNo:
            preFrmNo = curFrmNo
            print( "FrameNo: %d " % (frameData.iFrame))					
            print( "nMarkerset = %d" % frameData.nBodies)

        for iBody in range(frameData.nBodies):
            pBody = pointer(frameData.BodyData[iBody])		
            print("Markerset %d: %s" % (iBody + 1, pBody.contents.szName))
		
            # Markers
            print("\tnMarkers = %d\n" % pBody.contents.nMarkers)
            print("\t{\n")
            for i in range(pBody.contents.nMarkers):
                print("\t\tMarker %d：X:%f Y:%f Z:%f\n" % (
                i + 1,
                pBody.contents.Markers[i][0],
                pBody.contents.Markers[i][1],
                pBody.contents.Markers[i][2]))

            print("\t}\n")  
		
            # Segments
            print("\tnSegments = %d\n" % pBody.contents.nSegments)
            print("\t{\n")
            for i in range(pBody.contents.nSegments):
                print("\t\tSegment %d：Tx:%lf Ty:%lf Tz:%lf Rx:%lf Ry:%lf Rz:%lf Length:%lf\n" %
                    (i + 1,
                    pBody.contents.Segments[i][0],
                    pBody.contents.Segments[i][1],
                    pBody.contents.Segments[i][2],
                    pBody.contents.Segments[i][3],
                    pBody.contents.Segments[i][4],
                    pBody.contents.Segments[i][5],
                    pBody.contents.Segments[i][6]))

            print("\t}\n")
            print("}\n")

        # Unidentified Markers
        print("nUnidentifiedMarkers = %d" % data.contents.nUnidentifiedMarkers)
        print("{\n")
        for i in range(data.contents.nUnidentifiedMarkers):
        #x_data = np.ones([1,3])
        #for i in range(1):
            print("\tUnidentifiedMarkers %d：X:%f Y:%f Z:%f\n" %
                (i + 1,
                data.contents.UnidentifiedMarkers[i][0],
                data.contents.UnidentifiedMarkers[i][1],
                data.contents.UnidentifiedMarkers[i][2]))
        print("}\n")
    return(data.contents.UnidentifiedMarkers)

def py_data_func_user(data, userData):
    print("userData:%s" % cast(userData, c_char_p).value)
    py_data_func(data)
'''
def test_seeker_func():
    cdata = np.ones([1,3])
    cdata = data.contents.UnidentifiedMarkers
    for i in range(1):
        print("\tUnidentifiedMarkers %d：X:%f Y:%f Z:%f\n"%
            (i + 1,
            cdata[i][0],
            cdata[i][1],
            cdata[i][2]))       
        print("}\n")
        return cdata
'''    
print("Started the Seeker_SDK_Client Demo")

version = (c_ubyte * 4)()
GetSdkVersion(version)
print("VERSION:%d.%d.%d.%d" % (version[0], version[1], version[2], version[3]))

SetVerbosityLevel(1)
SetErrorMsgHandlerFunc(py_msg_func)

# Change to your local ip in 10.1.1.0/24
print("Begin to init the SDK Client")
ret = Initialize(b"10.1.1.198", b"10.1.1.198")

if ret == 0:
    print("Connect to the Seeker Succeed")
else:
    print("Connect Failed: [%d]" % ret)
    Exit()
    exit(0)
  
hostInfo = HostInfo()
GetHostInfo(pointer(hostInfo))
if hostInfo.bFoundHost == 1:
    print("Founded the Seeker Server")
else:
    print("Can't find the Seeker Server")
    exit(0)
SetDataHandlerFunc(py_data_func)
#SetDataHandlerFuncUser(py_data_func_user, b"Welcom to use Seeker_Test Demo")
t=1

while(input("Press q to quit\n") == "q"):
    #pass
    break

Exit()