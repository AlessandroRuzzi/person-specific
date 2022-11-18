import matplotlib.pyplot as plt
   
x_axis = [1,2,3,4,5,6,7,8,9]
#x_axis = [1,2,3,4,9]
gaze_estimator_generated = [6.7959, 6.2757 ,6.4476, 5.9946, 6.2791, 6.2299, 6.0028, 5.8454, 5.5412, 5.87]
gaze_estimator_real = [9.1816, 8.8344, 7.5443, 7.0526, 6.5745, 6.4363, 6.2372, 6.16, 6.0915, 5.9273]

gaze_estimator_generated = [6.9404,6.4128, 6.3792, 6.1194, 6.3165, 6.2466, 6.1404, 5.9907 ,5.8149]
gaze_estimator_real = [9.251, 9.034, 8.7359, 7.9428, 6.8732, 6.9114, 6.7033, 6.7695, 6.5758]

#gaze_estimator_generated = [6.8913, 6.3796, 6.5055, 6.1899, 6.3682, 6.2284, 6.0725, 5.9959, 5.6748, 5.9804]
#gaze_estimator_real = [12.318, 11.2155, 8.7523, 7.9366, 8.1317, 8.5618, 7.8733, 7.4974, 7.6556, 7.4697]

#gaze_estimator_generated = [6.7959, 6.2757, 6.493 , 5.985, 5.5412]
#gaze_estimator_real = [9.1816, 8.8344,7.5443, 7.0526, 6.0915]
csfont = {'fontname':'Times New Roman'}
f, ax = plt.subplots()
ax.autoscale(enable=True, axis='x', tight=True)
ax.set_aspect('equal')
plt.plot(x_axis, gaze_estimator_generated, color='red', marker='o', label ="Real + Generated Samples")
plt.plot(x_axis, gaze_estimator_real, color='blue', marker='o', label ="Real Samples")
plt.title('', fontsize=14, **csfont)
plt.xticks(x_axis)
plt.xlabel('Number of real samples', fontsize=17, **csfont)
plt.ylabel('Error (degrees)', fontsize=17, **csfont)
plt.grid(True)
plt.legend(loc="upper right", prop={'size': 14, 'family': 'Times New Roman'})
plt.show()

f.savefig("personal_calibration.pdf", bbox_inches='tight')