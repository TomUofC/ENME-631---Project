import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def func_eval(funcName, *args):
    return eval(funcName)(*args)

def RK2(function, y, t, tau):
    """Solves a given function with initial conditions using the 2nd order Runge-Kutta method \n
    Inputs:\n
    function -> the function or ODE to be solved\n
    y_init -> the initial condition for the function\n
    x_range -> the range of values to solve for\n
    tau -> the step between iterations
    """
    k1 = function(y, t)                 #evaluates the function at the current step
    
    y_star = y + (tau/2)*(k1)           #finds the half step y value
    
    k2 = function(y_star, t+tau/2)      #evaluates the function at the half step
    
    y_new = y + tau*k2                  #find the next iteration y value

    return(y_new)


def RK4(function, y, t, tau):
    """Solves a given function with initial conditions using the 4th order Runge-Kutta method \n
    Inputs:\n
    function -> the function or ODE to be solved\n
    y_init -> the initial condition for the function\n
    x_range -> the range of values to solve for\n
    tau -> the step between iterations
    """
    k1 = function(y, t)                 #evaluates the function at the current step

    k2 = function(y + (tau/2)*k1,(t+tau/2))

    k3 = function(y + (tau/2)*k2,(t+tau/2))

    k4 = function(y + (tau/2)*k3,(t+tau))

    k = tau*(k1+(2*k2)+(2*k3)+k4)/6
    
    y_new = y + k                  #find the next iteration y value

    return(y_new)

#Constants
mu_e = 2                                        #number of nucleons per electron 

p_slash_c = np.geomspace(0.1, 2.5e6, num=50)    #range of density values to be solved for
p_init = p_slash_c                              #initial density value
p_0 = 9.74e5*mu_e                               #standard white dwarf density value
p_RK2 = np.zeros((len(p_slash_c)))
p_RK4 = np.zeros((len(p_slash_c)))


tau = 0.1                                      #step between radius values
R_0 = 7.72e8/mu_e                               #standard size of a white dwarf
R_sun = 69600000000.0                           #radius of the sun [cm]
r = np.arange(tau, 4+tau,tau)                   #range of r values to be iterated over

m_init_RK2 = 0.0                                    #initial mass
m_init_RK4 = 0.0                                    #initial mass
M_0 = 5.67e33/(mu_e**2)                         #standard white dwarf mass
M_sun = 1.989e+33                               #mass of the sun [g]
M_limit = 1.46                                  #Chandrasekhar limit in solar masses


def system_RK2(y,r):
    """
    Describes the system of ODEs for white dwarf masses and radii
    Inputs: \n y -> state vector for the system 
    \n r -> radius value for the system
    Returns: values for dpdr and dmdr
    """
    p = y[0]                                                #density of the white dwarf
    m = y[1]                                                #mass of the white dwarf
    #print("p:",p,"m:", m)
    dpdr = -m*p/(p**(2/3)*r**2/(3*(1+p**(1/3))**(0.5)))     #differential equation for the density
    dmdr = r**2*p                                           #differential equation for the mass
    return np.array([dpdr, dmdr])

radius_RK2 = []                                                 #list for final radii values
mass_RK2 = []                                                   #list for final mass values


for p_RK2 in np.geomspace(0.1, 2.5e6, num=50):                  #loops through range of central density values
    y = np.array([p_RK2, m_init_RK2])                               #states initial conditions for the state vector
    
    for i in range(len(r)):                                 #loops through range of radii values
        y_prev = y                                          #saves the previous state vector
        y = RK2(system_RK2,y,r[i],tau)                          #RK2 function calculate the new state vector value

        if np.isnan(y[0]) == True:                          #checks if the density (y[0]) is not a number (i.e. divide by zero)
            radius_RK2.append(r[i-1])          #if the density is zero then the radius and mass lists are appended with the previous value
            mass_RK2.append(y_prev[1])
            break                                           #ends the loop iteration for that given central density

radius_solar_RK2 = np.array(radius_RK2)*R_0/R_sun                   #converts dimensionless radius in terms of solar radius
mass_solar_RK2 = np.array(mass_RK2)*M_0/M_sun                       #converts dimensionless mass in terms of solar mass

def system_RK4(z,r):
    """
    Describes the system of ODEs for white dwarf masses and radii
    Inputs: \n y -> state vector for the system 
    \n r -> radius value for the system
    Returns: values for dpdr and dmdr
    """
    p = z[0]                                                #density of the white dwarf
    m = z[1]                                                #mass of the white dwarf
    print("p:",p,"m:", m)
    dpdr = -m*p/(p**(2/3)*r**2/(3*(1+p**(1/3))**(0.5)))     #differential equation for the density
    dmdr = r**2*p                                           #differential equation for the mass
    return np.array([dpdr, dmdr])

radius_RK4 = []                                                 #list for final radii values
mass_RK4 = []                                                   #list for final mass values

for p in np.geomspace(0.1, 2.5e6, num=50):                  #loops through range of central density values
    z = np.array([p_RK4, m_init_RK4])                               #states initial conditions for the state vector
    
    for i in range(len(r)):                                 #loops through range of radii values
        z_prev = z                                          #saves the previous state vector
        z = RK4(system_RK4,z,r[i],tau)                          #RK2 function calculate the new state vector value

        if np.isnan(z[0]) == True:                          #checks if the density (z[0]) is not a number (i.e. divide by zero)
            radius_RK4.append(r[i-1])          #if the density is zero then the radius and mass lists are appended with the previous value
            mass_RK4.append(z_prev[1])
            break                                           #ends the loop iteration for that given central density

radius_solar_RK4 = np.array(radius_RK4)*R_0/R_sun                   #converts dimensionless radius in terms of solar radius
mass_solar_RK4 = np.array(mass_RK4)*M_0/M_sun                       #converts dimensionless mass in terms of solar mass

#Plots mass vs radius


plt.plot(mass_solar_RK2,radius_solar_RK2, 'b', label="2nd Order RK - Tau = {}".format(tau))     #plots the RK2 function
plt.plot(mass_solar_RK4,radius_solar_RK4, 'g', label="4th Order RK - Tau = {}".format(tau))     #plots the RK4 function
plt.axvline(x=M_limit ,linestyle='--',label="Chandrasekhar limit (1.46 M⊙)")
plt.legend()
plt.ylabel('Radius (R⊙)')
plt.xlabel('Mass (M⊙)')
plt.xlim([0, 20])
plt.title("White Dwarf Radii vs Mass")
plt.show()

#Plotting vs example stars
stars = np.loadtxt('wd_data.txt',usecols=(1,2,3,4))                                 #gets star data from file
star_names = np.genfromtxt('wd_data.txt', dtype=str, usecols=(0)).tolist()          #gets star names from file

plt.plot(mass_solar_RK4,radius_solar_RK4, 'b', label="4th Order RK - Tau = {}".format(tau)) #plots star data and names with RK2 and ivp functions
plt.plot(stars[:,0],stars[:,2], '*', label="Star Data")
plt.errorbar(x=stars[:,0],y=stars[:,2],yerr=stars[:,3],xerr=stars[:,1], fmt=' ',color='b', elinewidth=0.5)
for i in range(len(star_names)): plt.annotate(star_names[i],xy=(stars[i,0],stars[i,2]))     #adds star names to star data
plt.plot(mass_solar_RK2,radius_solar_RK2, 'g', label="solve_ivp")
plt.axvline(x=M_limit ,linestyle='--',label="Chandrasekhar limit (1.46 M⊙)")
plt.legend()
plt.ylabel('Radius (R⊙)')
plt.xlabel('Mass (M⊙)')
plt.xlim([0, 1.5])
plt.title("White Dwarf Radii vs Mass")
plt.show()