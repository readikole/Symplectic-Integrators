import numpy as np 
import matplotlib.pyplot as plt 
import time
#for the Hamiltonian of assignment 1
start_time  = time.time()
N = 50000
a = 0
b = 2500
h = (b-a)/N
E = 1/8                            #the total energy of the system
'''
#intialize vectors
x_a1 = np.zeros(N, dtype='float')
x_a2 = np.zeros(N, dtype='float')
x_a3 = np.zeros(N, dtype='float')
y_a1 = np.zeros(N, dtype='float')
y_a2 = np.zeros(N, dtype='float')
y_a3 = np.zeros(N, dtype='float')
px_b1 = np.zeros(N, dtype='float')
px_b2 = np.zeros(N, dtype='float')
py_b1 = np.zeros(N, dtype='float')
py_b2= np.zeros(N, dtype='float')
H_new = np.zeros(N, dtype='float')
#set intial conditions
y0 = -0.2
x0 = 0
py0 = 0
#defint dunction to find px0
def initial_px(x0, y0, py0):
    return np.sqrt(2*E - py0**2  - y0**2 - 2*x0**2*y0 + (2/3)*y0**2)
px0 = initial_px(x0, y0, py0)

# set then ABA82 integrator constants
a1 = 1/2  - np.sqrt(525 + 70*np.sqrt(30))/70
a2  = (np.sqrt(525 + 70*np.sqrt(30)) - np.sqrt(525 - 70*np.sqrt(30)))/70
a3 = np.sqrt(525 - 70*np.sqrt(30))/35
b1 = 1/4 - np.sqrt(30)/72
b2 = 1/4 + np.sqrt(30)/72

#propagate the intial conditions forward
i = 0                   #first itegration
while i<N-1:
    #handle the first iteration i = 0
    x_a1[0] = x0 + a1*h*px0
    y_a1[0] = y0 + a1*h*py0
    px_b1[0] = px0 - b1*h*(x0 + 2*x0*y0)
    py_b1[0] = py0 - b1*h*(y0 + x0**2 - y0**2)

    #propagate forward for all time steps
    x_a2[i+1] = x_a1[i]  + a2*h*px_b1[i]
    y_a2[i+1] = y_a1[i] + a2*h*py_b1[i]
    px_b2[i+1] = px_b1[i] - b2*h*(x_a2[i+1] + 2*x_a2[i+1]*y_a2[i+1])
    py_b2[i+1] = py_b1[i] - b2*h*(y_a2[i+1] + x_a2[i+1]**2 - y_a2[i+1]**2)
    x_a3[i+1] = x_a2[i+1] + a3*h*px_b2[i+1]
    y_a3[i+1] = y_a2[i+1] + a3*h*py_b2[i+1]
    px_b2[i+1] = px_b2[i+1] - b2*h*(x_a3[i+1] + 2*x_a3[i+1]*y_a3[i+1])
    py_b2[i+1] = py_b2[i+1] - b2*h*(y_a3[i+1] + x_a3[i+1]**2 - y_a3[i+1]**2)
    x_a2[i+1] = x_a3[i+1]  + a2*h*px_b2[i+1]
    y_a2[i+1] = y_a3[i+1] + a2*h*py_b2[i+1]
    px_b1[i+1] = px_b2[i+1] - b1*h*(x_a2[i+1] + 2*x_a2[i+1]*y_a2[i+1])
    py_b1[i+1] = py_b2[i+1] - b1*h*(y_a2[i+1] + x_a2[i+1]**2 - y_a2[i+1]**2)
    x_a1[i+1] = x_a2[i+1]  + a1*h*px_b1[i+1]
    y_a1[i+1] = y_a2[i+1] + a1*h*py_b1[i+1]

    #calculate the total energy
    H_new[i] = (1/2)*(px_b1[i]**2 + py_b1[i]**2) + (1/2)*(x_a1[i]**2 + y_a1[i]**2) + x_a1[i]**2 *y_a1[i] - (1/3)*y_a1[i]**3
    i+=1
    pass

end_time = time.time() - start_time
print(end_time)

Energy = np.zeros(N, dtype='float')
for i in range(N):
    Energy[i] = E

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
#plt.loglog(np.linspace(0,N, N), H_new)
plt.loglog(np.linspace(0,N, N),abs(Energy - H_new)/Energy)
plt.show()
'''

#for the Hamiltonian of assignment 2
start_time2 = time.time()
#intialize vectors
x2_a1 = np.zeros(N, dtype='float')
x2_a2 = np.zeros(N, dtype='float')
x2_a3 = np.zeros(N, dtype='float')
y2_a1 = np.zeros(N, dtype='float')
y2_a2 = np.zeros(N, dtype='float')
y2_a3 = np.zeros(N, dtype='float')
px2_b1 = np.zeros(N, dtype='float')
px2_b2 = np.zeros(N, dtype='float')
py2_b1 = np.zeros(N, dtype='float')
py2_b2= np.zeros(N, dtype='float')
H2_new = np.zeros(N, dtype='float')
#set intial conditions
y20 = -0.25
x20 = 0
py20 = 0
H2 = 1
#defint dunction to find px0
def init_px(x20, y20, py20):
    #to be corrected
    return np.sqrt(2*H2 - py20**2 - (1/12)*(np.exp(2*y20 + 2*np.sqrt(3)*x20) + np.exp(2*y20 - 2*np.sqrt(3)*x20) + np.exp(-4*y20)  + 1/4))

px20 = init_px(x20, y20, py20)

# set then ABA82 integrator constants
a1 = 1/2  - np.sqrt(525 + 70*np.sqrt(30))/70
a2  = (np.sqrt(525 + 70*np.sqrt(30)) - np.sqrt(525 - 70*np.sqrt(30)))/70
a3 = np.sqrt(525 - 70*np.sqrt(30))/35
b1 = 1/4 - np.sqrt(30)/72
b2 = 1/4 + np.sqrt(30)/72

#propagate the intial conditions forward
i = 0                   #first itegration
error2 = np.zeros(N)
while i<N-1:
    #handle the first iteration i = 0
    x2_a1[0] = x20 + a1*h*px20
    y2_a1[0] = y20 + a1*h*py20
    px2_b1[0] = px20 - b1*h*(np.sqrt(3)/12)*b1*h*(np.exp(2*y20 + 2*np.sqrt(3)*x20) - np.exp(2*y20 - 2*np.sqrt(3)*x20))
    py2_b1[0] = py20 - (1/12)*b1*h*(np.exp(2*y20 + 2*np.sqrt(3)*x20 + np.exp(2*y20 - 2*np.sqrt(3)*x20) - 2*np.exp(-4*y20)))

    #propagate forward for all time steps
    x2_a2[i+1] = x2_a1[i]  + a2*h*px2_b1[i]
    y2_a2[i+1] = y2_a1[i] + a2*h*py2_b1[i]
    px2_b2[i+1] = px2_b1[i]  - (np.sqrt(3)/12)*b2*h*(np.exp(2*y2_a2[i+1] + 2*np.sqrt(3)*x2_a2[i+1]) - np.exp(2*y2_a2[i+1] - 2*np.sqrt(3)*x2_a2[i+1]))
    py2_b2[i+1] = py2_b1[i] - (1/12)*b2*h*(np.exp(2*y2_a2[i+1] + 2*np.sqrt(3)*x2_a2[i+1]) + np.exp(2*y2_a2[i+1] - 2*np.sqrt(3)*x2_a2[i+1]) - 2*np.exp(-4*y2_a2[i+1]))
    x2_a3[i+1] = x2_a2[i+1] + a3*h*px2_b2[i+1]
    y2_a3[i+1] = y2_a2[i+1] + a3*h*py2_b2[i+1]
    px2_b2[i+1] = px2_b2[i+1]  - (np.sqrt(3)/12)*b2*h*(np.exp(2*y2_a3[i+1] + 2*np.sqrt(3)*x2_a3[i+1]) - np.exp(2*y2_a3[i+1] - 2*np.sqrt(3)*x2_a3[i+1]))
    py2_b2[i+1] = py2_b2[i+1] - (1/12)*b2*h*(np.exp(2*y2_a3[i+1] + 2*np.sqrt(3)*x2_a3[i+1]) + np.exp(2*y2_a3[i+1] - 2*np.sqrt(3)*x2_a3[i+1]) - 2*np.exp(-4*y2_a3[i+1]))
    x2_a2[i+1] = x2_a3[i+1]  + a2*h*px2_b2[i+1]
    y2_a2[i+1] = y2_a3[i+1] + a2*h*py2_b2[i+1]
    px2_b1[i+1] = px2_b2[i+1] - (np.sqrt(3)/12)*b1*h*(np.exp(2*y2_a2[i+1] + 2*np.sqrt(3)*x2_a2[i+1]) - np.exp(2*y2_a2[i+1] - 2*np.sqrt(3)*x2_a2[i+1]))
    py2_b1[i+1] = py2_b2[i+1] - (1/12)*b1*h*(np.exp(2*y2_a2[i+1] + 2*np.sqrt(3)*x2_a2[i+1]) + np.exp(2*y2_a2[i+1] - 2*np.sqrt(3)*x2_a2[i+1]) - 2*np.exp(-4*y2_a2[i+1]))
    x2_a1[i+1] = x2_a2[i+1]  + a1*h*px2_b1[i+1]
    y2_a1[i+1] = y2_a2[i+1] + a1*h*py2_b1[i+1]

    #calculate the total energy
    H2_new[i] = (1/2)*(px2_b1[i]**2 + py2_b1[i]**2) + (1/24)*(np.exp(2*y2_a2[i] + 2*np.sqrt(3)*x2_a2[i]) + np.exp(2*y2_a2[i] - 2*np.sqrt(3)*x2_a2[i])  + np.exp(-4*y2_a2[i]))   - 1/8
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
plt.scatter(y2_a1, py2_b1, s=0.1)
plt.xlabel('y')
plt.ylabel('py')
plt.show()
plt.scatter(x2_a1, px2_b1, s=0.1)
plt.xlabel('x')
plt.ylabel('px')
plt.show()
plt.scatter(x2_a1, y2_a1,  s=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.grid()
plt.loglog(np.linspace(0,N, n), new_error, marker='o', color='r')
plt.show()


#for the Hamiltonian of assignment 3
#for the Hamiltonian of assignment 2
start_time3 = time.time()
#intialize vectors
x3_a1 = np.zeros(N, dtype='float')
x3_a2 = np.zeros(N, dtype='float')
x3_a3 = np.zeros(N, dtype='float')
y3_a1 = np.zeros(N, dtype='float')
y3_a2 = np.zeros(N, dtype='float')
y3_a3 = np.zeros(N, dtype='float')
px3_b1 = np.zeros(N, dtype='float')
px3_b2 = np.zeros(N, dtype='float')
py3_b1 = np.zeros(N, dtype='float')
py3_b2= np.zeros(N, dtype='float')
H3_new = np.zeros(N, dtype='float')


A = 0.25
B = 0.75
E = -10**(-8)
H3 = 2

# set then ABA82 integrator constants
a1 = 1/2  - np.sqrt(525 + 70*np.sqrt(30))/70
a2  = (np.sqrt(525 + 70*np.sqrt(30)) - np.sqrt(525 - 70*np.sqrt(30)))/70
a3 = np.sqrt(525 - 70*np.sqrt(30))/35
b1 = 1/4 - np.sqrt(30)/72
b2 = 1/4 + np.sqrt(30)/72

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
    px3_b1[0] = px30 - b1*h*(-2*E*x30 + 6*A*x30**5 + 4*B*x30**3 *y30**2 + 2*B*x30*(y30**4))
    py3_b1[0] = py30 - b1*h*(-2*E*x30 + 6*A*y30**5  + 4*B*x30**4 *y30**3 + 2*B*y30*(x30**4))

    #propagate forward for all time steps
    x3_a2[i] = x3_a1[i] + a2*h*px3_b1[i]
    y3_a2[i] = y3_a1[i] + a2*h*py3_b1[i]

    px3_b2[i] = px3_b1[i] - b2*h*(-2*E*x3_a2[i] + 6*A*x3_a2[i]**5 + 4*B*x3_a2[i]**3*(y3_a2[i]**2) + 2*B*x3_a2[i]*(y3_a2[i]**4))
    py3_b2[i] = py3_b1[i] - b2*h*(-2*E*x3_a2[i] + 6*A*y3_a2[i]**5  + 4*B*x3_a2[i]**4*(y3_a2[i]**3) + 2*B*y3_a2[i]*(x3_a2[i]**4))

    x3_a3[i] = x3_a2[i]  + a3*h*px3_b2[i]
    y3_a3[i] = y3_a2[i] + a3*h*py3_b2[i] 

    px3_b2[i+1] = px3_b2[i] - b2*h*(-2*E*x3_a3[i] + 6*A*x3_a3[i]**5 + 4*B*x3_a3[i]**3*(y3_a3[i]**2) + 2*B*x3_a3[i]*(y3_a3[i]**4))
    py3_b2[i+1] = py3_b2[i] - b2*h*(-2*E*x3_a3[i] + 6*A*y3_a3[i]**5  + 4*B*x3_a3[i]**4*(y3_a3[i]**3) + 2*B*y3_a3[i]*(x3_a3[i]**4))
 
    x3_a2[i+1] = x3_a3[i] + a2*h*px3_b2[i+1]
    y3_a2[i+1] = y3_a3[i] + a2*h*py3_b1[i+1]

    px3_b1[i+1] = px3_b2[i+1] - b1*h*(-2*E*x3_a2[i+1] + 6*A*x3_a2[i+1]**5 + 4*B*x3_a2[i+1]**3*(y3_a2[i+1]**2) + 2*B*x3_a2[i+1]*(y3_a2[i+1]**4))
    py3_b1[i+1] = py3_b2[i+1] - b1*h*(-2*E*x3_a2[i+1] + 6*A*y3_a2[i+1]**5  + 4*B*x3_a2[i+1]**4*(y3_a2[i+1]**3) + 2*B*y3_a2[i+1]*(x3_a2[i+1]**4))
 

    x3_a1[i+1] = x3_a2[i+1] + a1*h*px3_b1[i+1]
    y3_a1[i+1] = y3_a2[i+1] + a1*h*py3_b1[i+1]
 
    #calculate the total energy
    H3_new[i] = (1/2)*(px3_b1[i]**2 + py3_b1[i]**2) + (1/2)*(x3_a1[i]**2 + y3_a1[i]**2) + x3_a1[i]**2 *y3_a1[i] - (1/3)*y3_a1[i]**3
    error3[i] = (H3 - H3_new[i])/H3
    i+=1


end_time3 = time.time() - start_time3
print('CPU Time: ', end_time3)
plt.scatter(y3_a1, py3_b1, s=0.1)
plt.xlabel('y')
plt.ylabel('py')
plt.show()
plt.scatter(x3_a1, px3_b1, s=0.1)
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
plt.loglog(np.linspace(0,N, N), (H3 - H3_new)/H3)
#plt.loglog(np.linspace(0,N, N),error1, np.linspace(0,N, N), error2, np.linspace(0,N, N), error3)
plt.legend(['Error in H1', 'Error in H2', 'Error in H3'])
#plt.xlim(0, 10**3)
plt.show()