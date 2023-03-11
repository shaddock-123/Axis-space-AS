import numpy as np
from parl.utils import summary

x = range(100)
for i in x:
    summary.add_scalar('y=2x', i * 2, i)
'''
def test_seeker_func():
    data = np.ones([1,3])
    for i in range(1):
        print("\tUnidentifiedMarkers %d：X:%f Y:%f Z:%f\n"%
            (i + 1,
            data[i][0],
            data[i][1],
            data[i][2]))       
        print("}\n")
        return data
'''