# Author: Aqeel Anwar(ICSRL)
# Created: 9/22/2019, 6:45 PM
# Email: aqeel.anwar@gatech.edu
import numpy as np
import cv2

run_name = 'Tello_indoor'
env_type = 'VanLeer'
stat_path = '../models/'+run_name+'/'+env_type+'/stat.npy'
data_path = '../models/'+run_name+'/'+env_type+'/data_tuple.npy'
print(data_path)
data_list = np.load(data_path)

iteration = [-1.00, 1.23]
for i in iteration:
    print('Iteration: {:>+1.4f}'.format(i))


print('Total datapoints: ', len(data_list))
first=True
for data in data_list:
    if not first:
        state = data[0][0,:,:,:]
        next_state = data[2][0,:,:,:]
        conc = np.concatenate((state, next_state), axis=1)
        yy=1
        cv2.imshow('d', conc)
        print('oaky')
        cv2.waitKey(1)
        xxx=2
    else:
        first = False