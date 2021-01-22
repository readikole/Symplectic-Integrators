import numpy as np 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import time
# is there a way of updating the Hamiltonian
#for the Hamiltonian of assignment 1
start_time  = time.time()
#initialize vectors 
N = 50000
a = 0
b = 2500
h = (b-a)/N
H = 1/8                             #the total energy of the system

x_a1 = np.zeros(N, dtype='float')
y_a1 = np.zeros(N, dtype='float')
px_half = np.zeros(N, dtype='float')
py_half = np.zeros(N, dtype='float')
x_a2 = np.zeros(N, dtype='float')
y_a2 = np.zeros(N, dtype='float')
H_new = np.zeros(N, dtype='float')
#set initial conditions 
y0 = -0.25
x0 = 0
py0 = 0
#defint dunction to find px0
def initial_px(x0, y0, py0):
    return np.sqrt(2*H - py0**2  - y0**2 - 2*x0**2*y0 + (2/3)*y0**2)
px0 = initial_px(x0, y0, py0)

#set constants for the SABA integrator
a1 = 1/2 - 1/(2*np.sqrt(3))
a2 = 1/np.sqrt(3)
b1 = 1/2

#propagate the initial conditions forward
#via successive canonical transformations
i = 0
error1 = np.zeros(N)
while i<N-1:
    #propage the intial conditions over one time step
    x_a1[0] = x0 + a1*h*px0
    y_a1[0] = y0 + a1*h*py0
    px_half[0] = px0 - b1*h*(x0 + 2*x0*y0)
    py_half[0] = py0 - b1*h*(y0 + x0**2 + y0**2)
    H_new[0] = (1/2)*(x_a1[0]**2 + y_a1[0]**2) + (1/2)*(px_half[0]**2 + py_half[0]**2) +(x_a1[0]**2)*y_a1[0] - (1/3)*y_a1[0]**3
    #compute first error term 
    error1[0] = abs((H - H_new[0])/H)
    #propagate forward for all time steps
    x_a2[i] = x_a1[i] + a2*h*px_half[i]
    y_a2[i] = y_a1[i] + a2*h*py_half[i]

    px_half[i+1] = px_half[i] - b1*h*(x_a2[i] + 2*x_a2[i]*y_a2[i])
    py_half[i+1] = py_half[i] - b1*h*(y_a2[i] + x_a2[i]**2 - y_a2[i]**2)

    x_a1[i+1] = x_a2[i]  + a1*h*px_half[i+1]
    y_a1[i+1] = y_a2[i] + a1*h*py_half[i+1]

    #calculate the total energy
    H_new[i+1] = (1/2)*(px_half[i+1]**2 + py_half[i+1]**2) + (1/2)*(x_a1[i+1]**2 + y_a1[i+1]**2) +(x_a1[i+1]**2)*y_a1[i+1] - (1/3)*y_a1[i+1]**3
    error1[i+1] = abs((H - H_new[i+1])/H)
    i+=1
    
Energy = np.zeros(N, dtype='float')
for i in range(N):
    Energy[i] = H

end_time = time.time() - start_time
print('End time for H1: ', end_time)
plt.scatter(y_a1, py_half, s=0.1)
plt.xlabel('y')
plt.ylabel('py')
plt.show()
plt.scatter(x_a1, px_half, s=0.1)
plt.xlabel('x')
plt.ylabel('px')
plt.show()
plt.scatter(x_a1, y_a1,marker='o',  s=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#plt.loglog(np.linspace(0,N, N), Energy)
#plt.loglog(np.linspace(0,N, N), H_new)
plt.grid()
plt.loglog(np.linspace(0,N, N),error1)#, s=0.1)

plt.show()


start_time2 = time.time()
#for the Hamiltonian  of assignment 2
#initialize vectors 
#N = 50000
a = 0
b = 2500
h = (b-a)/N
H2 = 256                             #the total energy of the system
x2_a1 = np.zeros(N, dtype='float')
y2_a1 = np.zeros(N, dtype='float')
px2_half = np.zeros(N, dtype='float')
py2_half = np.zeros(N, dtype='float')
x2_a2 = np.zeros(N, dtype='float')
y2_a2 = np.zeros(N, dtype='float')
H2_new = np.zeros(N, dtype='float')

#set constants for the SABA integrator
a1 = 1/2 - 1/2*np.sqrt(3)
a2 = 1/np.sqrt(3)
b1 = 1/2
#define a function to find the intial px0
def initi_px(x10,y10,py10):
   return np.sqrt(2*H2 - py10**2 - (1/12)*(np.exp(2*y10 + 2*np.sqrt(3)*x10) + np.exp(2*y10 - 2*np.sqrt(3)*x10) + np.exp(-4*y10)  + 1/4))


#set initial conditions 
y10 = -0.5
x10 = 0
py10 = 0
px10 = initi_px(x10,y10,py10)

#propagate the initial conditions forward
#via successive canonical transformations
i = 0
error2 = np.zeros(N)
while i<N-1:
    #propage the intial conditions over one time step
    x2_a1[0] = x10 + a1*h*px10
    y2_a1[0] = y10 + a1*h*py10
    px2_half[0] = px10 - (np.sqrt(3)/12)*b1*h*(np.exp(2*y10 + 2*np.sqrt(3)*x10) - np.exp(2*y10 - 2*np.sqrt(3)*x10))
    py2_half[0] = py10 - (1/12)*b1*h*(np.exp(2*y10 + 2*np.sqrt(3)*x10 + np.exp(2*y10 - 2*np.sqrt(3)*x10) - 2*np.exp(-4*y10)))

    #propagate forward for all time steps
    x2_a2[i+1] = x2_a1[i] + a2*h*px2_half[i]
    y2_a2[i+1] = y2_a1[i] + a2*h*py2_half[i]

    px2_half[i+1] = px2_half[i] - (np.sqrt(3)/12)*b1*h*(np.exp(2*y2_a2[i+1] + 2*np.sqrt(3)*x2_a2[i+1]) - np.exp(2*y2_a2[i+1] - 2*np.sqrt(3)*x2_a2[i+1]))
    py2_half[i+1] = py2_half[i] - (1/12)*b1*h*(np.exp(2*y2_a2[i+1] + 2*np.sqrt(3)*x2_a2[i+1]) + np.exp(2*y2_a2[i+1] - 2*np.sqrt(3)*x2_a2[i+1]) - 2*np.exp(-4*y2_a2[i+1]))

    x2_a1[i+1] = x2_a2[i+1]  + a1*h*px2_half[i+1]
    y2_a1[i+1] = y2_a2[i+1] + a1*h*py2_half[i+1]

    #calculate the total energy
    H2_new[i] = (1/2)*(px2_half[i]**2 + py2_half[i]**2) + (1/24)*(np.exp(2*y2_a1[i] + 2*np.sqrt(3)*x2_a1[i]) + np.exp(2*y2_a1[i] - 2*np.sqrt(3)*x2_a1[i])  + np.exp(-4*y2_a1[i]))   - 1/8
    error2[i] = (H2 - H2_new[i])/H2
    i+=1
end_time2 = time.time() - start_time2
print('End time for H2: ', end_time2)

plt.scatter(y2_a1, py2_half, s=0.1)
plt.xlabel('y')
plt.ylabel('py')
plt.show()
plt.scatter(x2_a1, px2_half, s=0.1)
plt.xlabel('x')
plt.ylabel('px')
plt.show()
plt.scatter(x2_a1, y2_a1,  s=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#plt.loglog(np.linspace(0,N, N), Energy)
#plt.loglog(np.linspace(0,N, N), H_new)
#plt.grid()
plt.loglog(np.linspace(0,N, N), (H2 - H2_new)/H2)
#plt.loglog(np.linspace(0,N, N),(H - H_new)/H, np.linspace(0,N, N),(H2 - H2_new)/H2)
#plt.legend(['Error in H1', 'Error in H2'])
plt.show()



#for the Hamiltonian of assignement 3
start_time3 = time.time()
x3_a1 = np.zeros(N, dtype='float')
y3_a1 = np.zeros(N, dtype='float')
px3_half = np.zeros(N, dtype='float')
py3_half = np.zeros(N, dtype='float')
x3_a2 = np.zeros(N, dtype='float')
y3_a2 = np.zeros(N, dtype='float')
H3_new = np.zeros(N, dtype='float')
#set constants for H_3
A = 0.25
B = 0.75
E = -10**(-8)
H3 = 2
#set constants for the SABA integrator
a1 = 1/2 - 1/2*np.sqrt(3)
a2 = 1/np.sqrt(3)
b1 = 1/2
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
    x3_a1[0] = x30 + a1*h*px30
    y3_a1[0] = y30 + a1*h*py30
    px3_half[0] = px30 - b1*h*(-2*E*x30 + 6*A*x30**5 + 4*B*x30**3 *y30**2 + 2*B*x30*(y30**4))
    py3_half[0] = py30 - b1*h*(-2*E*x30 + 6*A*y30**5  + 4*B*x30**4 *y30**3 + 2*B*y30*(x30**4))

    #propagate forward for all time steps
    x3_a2[i+1] = x3_a1[i] + a2*h*px3_half[i]
    y3_a2[i+1] = y3_a1[i] + a2*h*py3_half[i]

    px3_half[i+1] = px3_half[i] - b1*h*(-2*E*x3_a2[i+1] + 6*A*x3_a2[i+1]**5 + 4*B*x3_a2[i+1]**3*(y3_a2[i+1]**2) + 2*B*x3_a2[i+1]*(y3_a2[i+1]**4))
    py3_half[i+1] = py3_half[i] - b1*h*(-2*E*x3_a2[i+1] + 6*A*y3_a2[i+1]**5  + 4*B*x3_a2[i+1]**4*(y3_a2[i+1]**3) + 2*B*y3_a2[i+1]*(x3_a2[i+1]**4))

    x3_a1[i+1] = x3_a2[i+1]  + a1*h*px3_half[i+1]
    y3_a1[i+1] = y3_a2[i+1] + a1*h*py3_half[i+1]

    #calculate the total energy
    H3_new[i] = (1/2)*(px3_half[i]**2 + py3_half[i]**2) + (1/2)*(x3_a1[i]**2 + y3_a1[i]**2) + x3_a1[i]**2 *y3_a1[i] - (1/3)*y3_a1[i]**3
    error3[i] = (H3 - H3_new[i])/H3
    i+=1


end_time3 = time.time() - start_time3
plt.scatter(y3_a1, py3_half, s=0.1)
plt.xlabel('y')
plt.ylabel('py')
plt.show()
plt.scatter(x3_a1, px3_half, s=0.1)
plt.xlabel('x')
plt.ylabel('px')
plt.show()
plt.scatter(x3_a1, y3_a1,  s=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#plt.loglog(np.linspace(0,N, N), Energy)
#plt.loglog(np.linspace(0,N, N), H_new)
plt.grid()
plt.loglog(np.linspace(0,N, N), (H3 - H2_new)/H3)
#plt.loglog(np.linspace(0,N, N),error1, np.linspace(0,N, N), error2, np.linspace(0,N, N), error3)
plt.legend(['Error in H1', 'Error in H2', 'Error in H3'])
#plt.xlim(0, 10**3)
plt.show()