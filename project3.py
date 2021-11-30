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

#Question 1 - Testing

def myFunc(x, y):
    """Function used for testing the RK2 function
    """
    dy = np.zeros((len(y)))
    dy[0] = np.exp(-2 * x) - 2 * y[0]
    return dy

# -----------------------
"""
tau = 0.01
x = np.array([0, 2])
yinit = np.array([1.0/10])


[ts, ys] = RK2('myFunc', yinit, x, tau)


dt = int((x[-1]-x[0])/tau)
t = [x[0]+i*tau for i in range(dt+1)]
yexact = []
for i in range(dt+1):
    ye = (1.0/10)*np.exp(-2*t[i]) + t[i]*np.exp(-2*t[i])
    yexact.append(ye)

plt.plot(ts, ys, 'r')
plt.plot(t, yexact, 'b')
plt.xlim(x[0], x[1])
plt.legend(["2nd Order RK", "Exact solution"], loc=1)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
"""
#Question 2
mu_e = 2                                        #number of nucleons per electron 

p_slash_c = np.geomspace(0.1, 2.5e6, num=50)    #range of density values to be solved for
p = np.zeros((len(p_slash_c)))      
p_init = p_slash_c                              #initial density value
p_0 = 9.74e5*mu_e                               #standard white dwarf density value

tau = 0.01                                      #step between radius values
R_0 = 7.72e8/mu_e                               #standard size of a white dwarf
R_sun = 69600000000.0                           #radius of the sun [cm]
r = np.arange(tau, 4+tau,tau)                   #range of r values to be iterated over

m_init = 0.0                                    #initial mass
M_0 = 5.67e33/(mu_e**2)                         #standard white dwarf mass
M_sun = 1.989e+33                               #mass of the sun [g]
M_limit = 1.46                                  #Chandrasekhar limit in solar masses


def system(y,r):
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

radius = []                                                 #list for final radii values
mass = []                                                   #list for final mass values
for p in np.geomspace(0.1, 2.5e6, num=50):                  #loops through range of central density values
    y = np.array([p, m_init])                               #states initial conditions for the state vector
    
    for i in range(len(r)):                                 #loops through range of radii values
        y_prev = y                                          #saves the previous state vector
        y = RK2(system,y,r[i],tau)                          #RK2 function calculate the new state vector value

        if np.isnan(y[0]) == True:                          #checks if the density (y[0]) is not a number (i.e. divide by zero)
            radius.append(r[i-1])          #if the density is zero then the radius and mass lists are appended with the previous value
            mass.append(y_prev[1])
            break                                           #ends the loop iteration for that given central density

radius_solar = np.array(radius)*R_0/R_sun                   #converts dimensionless radius in terms of solar radius
mass_solar = np.array(mass)*M_0/M_sun                       #converts dimensionless mass in terms of solar mass

#Question 3

plt.plot(mass_solar,radius_solar, 'b', label="2nd Order RK")                    #plots the mass vs radius in terms of solar units
plt.axvline(x=M_limit ,linestyle='--',label="Chandrasekhar limit (1.46 M⊙)")   #label for the Chandradekhar limit
plt.legend()
plt.ylabel('Radius (R⊙)')
plt.xlabel('Mass (M⊙)')
plt.title("White Dwarf Radii vs Mass")
plt.show()

#Question 4
def system_ivp(r,y):                                    #redefines the system to use in ivp (note the arguments are swapped)
    p = y[0]                                            #density
    m = y[1]                                            #mass
    #print("p:",p,"m:", m)
    dpdr = -m*p/(p**(2/3)*r**2/(3*(1+p**(1/3))**(0.5)))
    dmdr = r**2*p
    return np.array([dpdr, dmdr])   
radius_ivp = []                                         #list for final radii_ivp values
mass_ivp = []                                           #list for final mass_ivp values
for p in np.geomspace(0.1, 2.5e6, num=50):              #range for central density values
    y = [p, m_init]
    solution = solve_ivp(system_ivp, [0.000001,4], y,dense_output=True)     #solution to the ivp function
    radius_ivp.append(solution.t[np.argmin(solution.y[0])])                 #grabs the radius value
    mass_ivp.append(solution.y[1,np.argmin(solution.y[0])])                 #grabs the mass value

radius_solar_ivp = np.array(radius_ivp)*R_0/R_sun                           #converts radius to solar units 
mass_solar_ivp = np.array(mass_ivp)*M_0/M_sun                               #converts mass to solar units

plt.plot(mass_solar,radius_solar, 'b', label="2nd Order RK - Tau = {}".format(tau))     #plots the RK2 function with the ivp function for comparison
plt.plot(mass_solar_ivp,radius_solar_ivp, 'g', label="solve_ivp")
plt.axvline(x=M_limit ,linestyle='--',label="Chandrasekhar limit (1.46 M⊙)")
plt.legend()
plt.ylabel('Radius (R⊙)')
plt.xlabel('Mass (M⊙)')
plt.xlim([0, 20])
plt.title("White Dwarf Radii vs Mass")
plt.show()

#Question 5
stars = np.loadtxt('wd_data.txt',usecols=(1,2,3,4))                                 #gets star data from file
star_names = np.genfromtxt('wd_data.txt', dtype=str, usecols=(0)).tolist()          #gets star names from file

plt.plot(mass_solar,radius_solar, 'b', label="2nd Order RK - Tau = {}".format(tau)) #plots star data and names with RK2 and ivp functions
plt.plot(stars[:,0],stars[:,2], '*', label="Star Data")
plt.errorbar(x=stars[:,0],y=stars[:,2],yerr=stars[:,3],xerr=stars[:,1], fmt=' ',color='b', elinewidth=0.5)
for i in range(len(star_names)): plt.annotate(star_names[i],xy=(stars[i,0],stars[i,2]))     #adds star names to star data
plt.plot(mass_solar_ivp,radius_solar_ivp, 'g', label="solve_ivp")
plt.axvline(x=M_limit ,linestyle='--',label="Chandrasekhar limit (1.46 M⊙)")
plt.legend()
plt.ylabel('Radius (R⊙)')
plt.xlabel('Mass (M⊙)')
plt.xlim([0, 1.5])
plt.title("White Dwarf Radii vs Mass")
plt.show()