import matplotlib.pyplot as plt
   
x_axis = [1,2,3,4,5,6,7,8,9,10]
#x_axis = [1,2,3,4,9]
gaze_estimator_generated = [6.7959, 6.2757 ,6.493, 5.985, 6.2791, 5.9602, 6.0028, 5.8454, 5.5412, 5.87]
gaze_estimator_real = [9.1816, 8.8344, 7.5443, 7.0526, 6.5745, 6.4363, 6.2372, 6.16, 6.0915, 5.9273]

#gaze_estimator_generated = [6.8913, 6.3796, 6.5055, 6.1899, 6.3682, 6.2284, 6.0725, 5.9959, 5.6748, 5.9804]
#gaze_estimator_real = [12.318, 11.2155, 8.7523, 7.9366, 8.1317, 8.5618, 7.8733, 7.4974, 7.6556, 7.4697]

#gaze_estimator_generated = [6.7959, 6.2757, 6.493 , 5.985, 5.5412]
#gaze_estimator_real = [9.1816, 8.8344,7.5443, 7.0526, 6.0915]
  
f = plt.figure()
plt.plot(x_axis, gaze_estimator_generated, color='red', marker='o', label ="Real samples + Generated samples")
plt.plot(x_axis, gaze_estimator_real, color='blue', marker='o', label ="Real samples")
plt.title('', fontsize=14)
plt.xlabel('Number of calibration samples', fontsize=14)
plt.ylabel('Error (degrees)', fontsize=14)
plt.grid(True)
plt.legend(loc="upper right")
plt.show()

f.savefig("figure4.pdf", bbox_inches='tight')