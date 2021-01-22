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
h = (b-a)/N

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
px[0] = initial_px(x[0], y[0], py[0])

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

        
        if x[i]*x[i+1]<0 and x[i]>0:                                   #if there is a change of sign, then there is a crossing
            #use either x_i or x_i+1 as an initial condition
            #the half time steps
            xnew_half[i] = x[i] + (-abs(x[i+1] - x[i])/2)*px_new[i]*(1/px[i+1])
            ynew_half[i] = y[i] + (-abs(x[i+1] - x[i])/2)*py_new[i]*(1/px[i+1])

            #fill the momenta
            px_new[i+1] = px[i]  + abs(x[i+1] - x[i])*(xnew_half[i] + 2*xnew_half[i]*ynew_half[i])*(1/px[i+1])
            py_new[i+1] =py[i]  + abs(x[i+1] - x[i])*(ynew_half[i] + xnew_half[i]**2 - ynew_half[i]**2)*(1/px[i+1])


            x_new[i+1] = xnew_half[i] + (-abs(x[i+1] - x[i])/2)*px_new[i+1]/px[i+1]
            y_new[i+1] = ynew_half[i] + (-abs(x[i+1] - x[i])/2)*py_new[i+1]/px[i+1]
      
     
    
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

'''

end_time = time.time() - start_time
print('Ending time: ', end_time)
plt.scatter(y, py, s=0.3)
plt.show()
plt.scatter(x, px, s=0.3)
plt.show()
plt.scatter(np.linspace(0,N, N),(H_new - E)/E, s=0.01)
#plt.xlim(0, 10**(4))
plt.show()
plt.scatter(x_half, y_half,marker='o',  s=0.3)
plt.show()



#for the hamiltonian of assignment 2
start_time2 = time.time()
#initialize the four phase variables with a vector of zeros
x2 = np.zeros(N, dtype='float')
y2 = np.zeros(N, dtype='float')
px2 = np.zeros(N, dtype='float')
py2 = np.zeros(N, dtype='float')
t2 = np.zeros(N, dtype='float')
#initialize other vetors that will be used for half time steps
x2_half = np.zeros(N, dtype='float')
y2_half = np.zeros(N, dtype='float')
H2_new = np.zeros(N, dtype='float')


y20 = -0.25
x20 = 0
py20 = 0
H2 = 1
#defint dunction to find px0
def init_px(x20, y20, py20):
    #to be corrected
    return np.sqrt(2*H2 - py20**2 - (1/12)*(np.exp(2*y20 + 2*np.sqrt(3)*x20) + np.exp(2*y20 - 2*np.sqrt(3)*x20) + np.exp(-4*y20)  + 1/4))

px20 = init_px(x20, y20, py20)

a1 = 1/2
b1 = 1
a2 = 1/2
error2 = np.zeros(N)
H_new = np.zeros(N, dtype='float')
i=0
while i<N-1:
    #handle the first iteration i = 0
    x2[0] = x20 + a1*h*px20
    y2[0] = y20 + a1*h*py20
    px2[0] = px20 - h*(np.sqrt(3)/12)*b1*h*(np.exp(2*y20 + 2*np.sqrt(3)*x20) - np.exp(2*y20 - 2*np.sqrt(3)*x20))
    py2[0] = py20 - (1/12)*h*(np.exp(2*y20 + 2*np.sqrt(3)*x20 + np.exp(2*y20 - 2*np.sqrt(3)*x20) - 2*np.exp(-4*y20)))

    #propagate forward for all time steps
    x2_half[i] = x2[i]  + a2*h*px2[i]
    y2_half[i] = y2[i] + a2*h*py2[i]
    
    px2[i+1] = px2[i]  - (np.sqrt(3)/12)*h*(np.exp(2*y2_half[i] + 2*np.sqrt(3)*x2_half[i]) - np.exp(2*y2_half[i] - 2*np.sqrt(3)*x2_half[i]))
    py2[i+1] = py2[i] - (1/12)*h*(np.exp(2*y2_half[i] + 2*np.sqrt(3)*x2_half[i]) + np.exp(2*y2_half[i] - 2*np.sqrt(3)*x2_half[i]) - 2*np.exp(-4*y2_half[i]))
    
    
    x2[i+1] = x2[i] + (h/2)*(px2[i] + px2[i+1])
    y2[i+1] = y2[i] + (h/2)*(py2[i] + py2[i+1])
    

    #calculate the total energy
    H2_new[i] = (1/2)*(px2[i]**2 + py2[i]**2) + (1/24)*(np.exp(2*y2[i] + 2*np.sqrt(3)*x2[i]) + np.exp(2*y2[i] - 2*np.sqrt(3)*x2[i])  + np.exp(-4*y2[i]))   - 1/8
    error2[i] = (H2 - H2_new[i])/H2
    i+=1
    pass
#sample data points for the error
n = 1000             #number of data points to be sampled
new_error = np.zeros(n)
for i in range(n):
    if i*50>N:
        break
    else:
        new_error[i] = error2[int(50*i)]
new_error = np.array(new_error)
end_time2 = time.time() - start_time2
print('CPU Time: ', end_time2)
plt.scatter(y2, py2, s=0.1)
plt.xlabel('y')
plt.ylabel('py')
plt.show()
plt.scatter(x2, px2, s=0.1)
plt.xlabel('x')
plt.ylabel('px')
plt.show()
plt.scatter(x2, y2,  s=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.grid()
plt.loglog(np.linspace(0,N, n), new_error, marker='o', color='r')
plt.show()


#for the hamiltonian of assignment 3

#for the hamiltonian of assignment 2
start_time3 = time.time()
#initialize the four phase variables with a vector of zeros
x3 = np.zeros(N, dtype='float')
y3 = np.zeros(N, dtype='float')
px3 = np.zeros(N, dtype='float')
py3 = np.zeros(N, dtype='float')
t2 = np.zeros(N, dtype='float')
#initialize other vetors that will be used for half time steps
x3_half = np.zeros(N, dtype='float')
y3_half = np.zeros(N, dtype='float')
H3_new = np.zeros(N, dtype='float')

a1 = 1/2
b1 = 1
a2 = 1/2

A = 0.25
B = 0.75
E = -10**(-8)
H3 = 2

#define a function to find the intial px0
def function(x0, y0,py0):
    return np.sqrt(2*H3 - py0**2 + 2*E*(x0**2 *y0**2)  - 2*A*(x0**6 + y0**6) - 2*B*(x0**4*(y0**2) + x0**2*(y0**4)))
#set initial conditions 
y30 = -0.8
x30 = 0
py30 = 0
px30 = function(x30, y30, py30)

i = 0
error3 = np.zeros(N)
while i<N-1:
    #propage the intial conditions over one time step
    x3[0] = x30 + a1*h*px30
    y3[0] = y30 + a1*h*py30
    px3[0] = px30 - b1*h*(-2*E*x30 + 6*A*x30**5 + 4*B*x30**3 *y30**2 + 2*B*x30*(y30**4))
    py3[0] = py30 - b1*h*(-2*E*x30 + 6*A*y30**5  + 4*B*x30**4 *y30**3 + 2*B*y30*(x30**4))

    #propagate forward for all time steps
    x3_half[i] = x3[i] + a1*h*px3[i]
    y3_half[i] = y3[i] + a1*h*py3[i]

    px3[i+1] = px3[i] - h*(-2*E*x3_half[i] + 6*A*x3_half[i]**5 + 4*B*x3_half[i]**3*(y3_half[i]**2) + 2*B*x3_half[i]*(y3_half[i]**4))
    py3[i+1] = py3[i] - h*(-2*E*x3_half[i] + 6*A*y3_half[i]**5  + 4*B*x3_half[i]**4*(y3_half[i]**3) + 2*B*y3_half[i]*(x3_half[i]**4))

    x3[i+1] = x3[i]  + a1*h*(px3[i] + px3[i+1])
    y3[i+1] = y3[i] + a1*h*(py3[i] + py3[i+1])

    #calculate the total energy
    H3_new[i] = (1/2)*(px3[i]**2 + py3[i]**2) + (1/2)*(x3[i]**2 + y3[i]**2) + x3[i]**2 *y3[i] - (1/3)*y3[i]**3
    error3[i] = (H3 - H3_new[i])/H3
    i+=1


end_time3 = time.time() - start_time3
print('CPU Time: ', end_time3)
plt.scatter(y3, py3, s=0.1)
plt.xlabel('y')
plt.ylabel('py')
plt.show()
plt.scatter(x3, px3, s=0.1)
plt.xlabel('x')
plt.ylabel('px')
plt.show()
plt.scatter(x3, y3,  s=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#plt.loglog(np.linspace(0,N, N), Energy)
#plt.loglog(np.linspace(0,N, N), H_new)
plt.grid()
plt.loglog(np.linspace(0,N, N), (H3 - H3_new)/H3)
#plt.loglog(np.linspace(0,N, N),error1, np.linspace(0,N, N), error2, np.linspace(0,N, N), error3)
plt.legend(['Error in H1', 'Error in H2', 'Error in H3'])
#plt.xlim(0, 10**3)
plt.show()
'''