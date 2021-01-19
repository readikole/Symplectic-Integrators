import numpy as np 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import time

#Six order symplectic integrator  - Runge-Kutta-Nystrom methods
start_time  = time.time()
#initialize vectors 
N = 50000
a = 0
b = 2500
h = (b-a)/N
H = 1/8                            #the total energy of the system

#initializing integration constants
a1= 0.123229775946271 
b1=0.0414649985182624
b2= 0.198128671918067 
a2= 0.290553797799558 
b3= -0.0400061921041533 
a3= -0.127049212625417 
b4= 0.0752539843015807 
a4= -0.246331761062075 
b5= -0.0115113874206879 
a5= 0.357208872795928
b6 = 1/2 - (b1+b2+b3+b4+b5)
a6 = 1 - 2*(a1+a2+a3+a4+a5)

#intializing phase space vectors
x_a1 = np.zeros(N, dtype='float')
y_a1 = np.zeros(N, dtype='float')
px_b1 = np.zeros(N, dtype='float')
py_b1 = np.zeros(N, dtype='float')
x_a2 = np.zeros(N, dtype='float')
y_a2 = np.zeros(N, dtype='float')
px_b2 = np.zeros(N, dtype='float')
py_b2 = np.zeros(N, dtype='float')
x_a3 = np.zeros(N, dtype='float')
y_a3 = np.zeros(N, dtype='float')
px_b3 = np.zeros(N, dtype='float')
py_b3 = np.zeros(N, dtype='float')
x_a4 = np.zeros(N, dtype='float')
y_a4 = np.zeros(N, dtype='float')
px_b4 = np.zeros(N, dtype='float')
py_b4 = np.zeros(N, dtype='float')
x_a5 = np.zeros(N, dtype='float')
y_a5 = np.zeros(N, dtype='float')
px_b5 = np.zeros(N, dtype='float')
py_b5 = np.zeros(N, dtype='float')
x_a6 = np.zeros(N, dtype='float')
y_a6 = np.zeros(N, dtype='float')
px_b6 = np.zeros(N, dtype='float')
py_b6 = np.zeros(N, dtype='float')
end_time = np.zeros(N, dtype='float')
H_new  = np.zeros(N, dtype='float')

#intial conditions 
y0 = -0.20
x0 = 0
py0 = 0
#defint dunction to find px0
def initial_px(x0, y0, py0):
    return np.sqrt(2*H - py0**2  - y0**2 - 2*x0**2*y0 + (2/3)*y0**2)
px0 = initial_px(x0, y0, py0)




i=0
error = np.zeros(N, dtype='float')
while i<N-1:
    # handle the first iteration from initial conditions
    px_b1[0] = px0 - b1*h*(x0 + 2*x0*y0)
    py_b1[0] = py0 - b1*h*(y0 + x0**2 + y0**2)

    x_a1[0] = x0 + a1*h*px_b1[0]
    y_a1[0] = y0 + a1*h*py_b1[0]

    px_b2[i] = px_b1[i] - b2*h*(x_a1[i] + 2*x_a1[i]*y_a1[i])
    py_b2[i] = py_b1[i] - b2*h*(y_a1[i] + x_a1[i]**2 - y_a1[i]**2)

    x_a2[i] = x_a1[i]  + a2*h*px_b2[i]
    y_a2[i] = y_a1[i] + a2*h*py_b2[i]

    px_b3[i] = px_b2[i] - b3*h*(x_a2[i] + 2*x_a2[i]*y_a2[i])
    py_b3[i] = py_b2[i] - b3*h*(y_a2[i] + x_a2[i]**2 - y_a2[i]**2)

    x_a3[i] = x_a2[i] + a3*h*px_b3[i]
    y_a3[i] = y_a2[i] + a3*h*py_b3[i]

    px_b4[i] = px_b3[i] - b4*h*(x_a3[i] + 2*x_a3[i]*y_a3[i])
    py_b4[i] = py_b3[i] - b4*h*(y_a3[i] + x_a3[i]**2 - y_a3[i]**2)

    x_a4[i] = x_a3[i] + a4*h*px_b4[i]
    y_a4[i] = y_a3[i] + a4*h*py_b4[i]

    px_b5[i] = px_b4[i] - b5*h*(x_a4[i] + 2*x_a4[i]*y_a4[i])
    py_b5[i] = py_b4[i] - b5*h*(y_a4[i] + x_a4[i]**2 - y_a4[i]**2)

    x_a5[i] = x_a4[i] + a5*h*px_b5[i]
    y_a5[i] = y_a4[i] + a5*h*py_b5[i]

    px_b6[i] = px_b5[i] - b6*h*(x_a5[i] + 2*x_a5[i]*y_a5[i])
    py_b6[i] = py_b5[i] - b6*h*(y_a5[i] + x_a5[i]**2 - y_a5[i]**2)

    x_a5[i+1] = x_a5[i] + a5*h*px_b6[i]
    y_a5[i+1] = y_a5[i] + a5*h*py_b6[i]

    px_b5[i+1] = px_b6[i] - b5*h*(x_a5[i+1] + 2*x_a5[i+1]*y_a5[i+1])
    py_b5[i+1] = py_b6[i] - b5*h*(y_a5[i+1] + x_a5[i+1]**2 - y_a5[i+1]**2)

    x_a4[i+1] = x_a5[i+1] + a4*h*px_b5[i+1]
    y_a4[i+1] = y_a5[i+1] + a4*h*py_b5[i+1]

    px_b4[i+1] = px_b5[i+1] - b4*h*(x_a4[i+1] + 2*x_a4[i+1]*y_a4[i+1])
    py_b4[i+1] = py_b5[i+1] - b4*h*(y_a4[i+1] + x_a4[i+1]**2 - y_a4[i+1]**2)

    x_a3[i+1] = x_a4[i+1]  + a3*h*px_b4[i+1]
    y_a3[i+1] = y_a4[i+1] + a3*h*py_b4[i+1]

    px_b3[i+1] = px_b4[i+1] - b3*h*(x_a3[i+1] + 2*x_a3[i+1]*y_a3[i+1])
    py_b3[i+1] = py_b4[i+1] - b3*h*(y_a3[i+1] + x_a3[i+1]**2 - y_a3[i+1]**2)

    x_a2[i+1] = x_a3[i+1]  + a2*h*px_b3[i+1]
    y_a2[i+1] = y_a3[i+1] + a2*h*py_b3[i+1]

    px_b2[i+1] = px_b3[i+1] - b2*h*(x_a2[i+1] + 2*x_a2[i+1]*y_a2[i+1])
    py_b2[i+1] = py_b3[i+1] - b2*h*(y_a2[i+1] + x_a2[i+1]**2 - y_a2[i+1]**2)

    x_a1[i+1] = x_a2[i+1]  + a1*h*px_b2[i+1]
    y_a1[i+1] = y_a2[i+1] + a1*h*py_b2[i+1]

    px_b1[i+1] = px_b2[i+1] - b1*h*(x_a1[i+1] + 2*x_a1[i+1]*y_a1[i+1])
    py_b1[i+1] = py_b2[i+1] - b1*h*(y_a1[i+1] + x_a1[i+1]**2 - y_a1[i+1]**2)

    #calculate the total energy at each time step
    H_new[i] = (1/2)*(px_b1[i]**2 + py_b1[i]**2) + (1/2)*(x_a1[i]**2 + y_a1[i]**2) + x_a1[i]**2 *y_a1[i] - (1/3)*y_a1[i]**3
    end_time[i] = time.time() - start_time
    error[i] = abs((H - H_new[i]))
    i+=1

print('CPU Time: ', time.time() - start_time)
#print(end_time)

#sampling the error
n = 100             #number of data points to be sampled
new_error = np.zeros(n)
for i in range(n):
    if i*100>=N:
        break
    else:
        new_error[i] = error[int(100*i)]


plt.scatter(y_a1, py_b1, s=0.1)
plt.xlabel('y')
plt.ylabel('py')
plt.show()
plt.scatter(x_a1, px_b1, s=0.1)
plt.xlabel('x')
plt.ylabel('px')
plt.show()
plt.scatter(x_a1, y_a1,marker='o',  s=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#plt.loglog(np.linspace(0,N, N), Energy)
#plt.loglog(np.linspace(0,N, n), new_error)
plt.grid()
plt.loglog(end_time,error)
plt.show()