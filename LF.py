import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from mpl_toolkits.mplot3d import Axes3D 
import time 


#leapfrog integrator of the Henon-Heiles Hamiltonian

#initialize the number of iterations and time step
start_time  = time.time()
print('Starting time : ', start_time)
N = 50000
a = 0
b = 2500
h = 0.05#(b-a)/N

#initialize the four phase variables with a vector of zeros
x = np.zeros(N, dtype='float')
y = np.zeros(N, dtype='float')
px = np.zeros(N, dtype='float')
py = np.zeros(N, dtype='float')
t = np.zeros(N, dtype='float')
#initialize other vetors that will be used for half time steps
x_half = np.zeros(N, dtype='float')
y_half = np.zeros(N, dtype='float')

#define a new set of variables for the poincare surface of section
x_new =  np.zeros(N, dtype='float')
y_new =  np.zeros(N, dtype='float')
px_new =  np.zeros(N, dtype='float')
py_new =  np.zeros(N, dtype='float')

#half time steps
xnew_half = np.zeros(N, dtype='float')
ynew_half = np.zeros(N, dtype='float')
t = np.zeros(N, dtype='float')


#create an empty array to see the conservation of H for each iteration
H_new = np.zeros(N, dtype='float')

E = 1/8
#define a function for calculating the initial px
def initial_px(x0, y0, py0):
    return np.sqrt(2*E - py0**2  - y0**2 - 2*x0**2*y0 + (1/3)*y0**2)
#px[0] = initial_px(x[0], y[0], py[0])

#propagate the initial conditions
i = 0
cpu_time = []               #cpu time for running each initial condition
#define a list of initial conditions
#y_vals = np.sin(np.linspace(-0.3, 0.3, 70))
#py_vals = np.linspace(-0.25, 0.25, 50)
y_vals = np.array([-0.25, -0.251, -0.252, -0.253, -0.254, -0.255, -0.256, -0.257, -0.258, -0.259, 
 -0.3, -0.31, -0.32, -0.33, -0.34, -0.35, -0.36, -0.37, -0.25, -0.26])
#py_vals = np.array([0, 0, -0.13, -0.14, -0.15, -0.16, -0.18 , -0.19, -0.20, -0.21, -0.22,
 #0.23, -0.24, -0.25, -0.26, -0.27, -0.28, -0.29,0, 0])
#x_vals = [-0.2, -0.21, 0.25]
for idx in range(len(y_vals)):
    #initialize initial conditions
    x[0] = 0
    y[0] = y_vals[idx]
    py[0] = 0#py_vals[i]
    px[0] = initial_px(x[0], y[0], py[0])
    for i in range(N-1):
        #fill the half-time steps
        x_half[i] = x[i] + (h/2)*px[i]
        y_half[i] = y[i] + (h/2)*py[i]
        #fill the momenta
        px[i+1] = px[i]  + h*(-x_half[i] - 2*x_half[i]*y_half[i])
        py[i+1] = py[i]  + h*(-y_half[i] - x_half[i]**2 + y_half[i]**2)

        #now calculate the intiger positions 
        x[i+1] = x_half[i] + (h/2)*(px[i] + px[i+1])
        y[i+1] = y_half[i] + (h/2)*(py[i] + py[i+1])
   
        #find the surface of section

        
        if x[i]*x[i+1]<0:                                   #if there is a change of sign, then there is a crossing
            #use either x_i or x_i+1 as an initial condition
            #the half time steps
            xnew_half[i] = x[i] + (-abs(x[i+1] - x[i])/2)*px_new[i]*(1/px[i+1])
            ynew_half[i] = y[i] + (-abs(x[i+1] - x[i])/2)*py_new[i]*(1/px[i+1])

            #fill the momenta
            px_new[i+1] = px[i]  + abs(x[i+1] - x[i])*(xnew_half[i] + 2*xnew_half[i]*ynew_half[i])*(1/px[i+1])
            py_new[i+1] =py[i]  + abs(x[i+1] - x[i])*(ynew_half[i] + xnew_half[i]**2 - ynew_half[i]**2)*(1/px[i+1])


            x_new[i+1] = x[i] + (-abs(x[i+1] - x[i])/2)*px_new[i+1]/px[i+1]
            y_new[i+1] = y[i] + (-abs(x[i+1] - x[i])/2)*py_new[i+1]/px[i+1]
      
     
    
        #calculate the total energy
        H_new[i] = (1/2)*(px[i]**2 + py[i]**2) + ((x[i+1] - [x[i]])/2)*(x[i]**2 + y[i]**2) + (x[i]**2)*y[i] - (1/3)*(y[i]**2)
        i+=1
    else:
        end_time = time.time() - start_time
        cpu_time.append(end_time)
        print('Ending time: ', idx , end_time)
        plt.scatter(y_new, py_new, s=0.01)
        plt.ylim(-0.6, 0.6)
        plt.xlim(-0.6, 0.8)
plt.show()  
