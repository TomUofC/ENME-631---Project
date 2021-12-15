import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def func_eval(funcName, *args):
    return eval(funcName)(*args)


def Euler(function, y, t, tau):

    slope = function(y, t)

    y_new = y +tau*slope

    return(y_new)

def RK4(function, y, t, tau):
    
    k1 = function(y, t)                 
    
    k2 = function(y + (tau/2)*k1, t+tau/2)      

    k3 = function(y+(tau/2)*k2, t+tau/2)

    k4 = function(y+(tau/2)*k3, t+tau/2)
    
    y_new = y + (tau/6)*(k1+(2*k2)+(2*k3)+k4)           

    return(y_new)


#Constants
mu_e_c = 12/6
mu_e_fe = 56/26

p_slash_c = np.geomspace(0.1, 2.5e6, num=100)
p_init = p_slash_c
p_0_c = 9.74e5*mu_e_c
p_0_fe = 9.74e5*mu_e_fe
p_Eu = np.zeros((len(p_slash_c)))
p_RK4 = np.zeros((len(p_slash_c)))
p_Rel_RK4 = np.zeros((len(p_slash_c)))
p_AB = np.zeros((len(p_slash_c)))

tau = 0.0001

R_0_c = 7.72e8/mu_e_c
R_0_fe = 7.72e8/mu_e_fe
R_sun = 69600000000.0 
r = np.arange(tau, 4+tau,tau) 

m_init_Eu = 0.0 
m_init_RK4 = 0.0                  
m_init_Rel_RK4 = 0.0                                   
m_init_AB = 0.0

M_0_c = 5.67e33/(mu_e_c**2) 
M_0_fe = 5.67e33/(mu_e_fe**2)
M_sun = 1.989e+33 
M_limit = 1.46


def system(y,r):
    
    p = y[0] 
    m = y[1] 

    dpdr = -m*p/(p**(2/3)*r**2/(3*(1+p**(1/3))**(0.5)))
    dmdr = r**2*p
    return np.array([dpdr, dmdr])


def gamma(y):

    gamma = y**(2/3)/(3*(1+y**(2/3))**0.5)

    return gamma

def RelCase(y,r):

    p = y[0]
    m = y[1]

    dpdr = -m*p/(r**2*(gamma(p)))
    dmdr = r**2*p

    return np.array([dpdr, dmdr])


radius_RK4 = [] 
mass_RK4 = [] 


#RK4 METHOD----------------------------------------------------

#NON-RELATIVISTIC----------------------------------------------
for p_RK4 in np.geomspace(0.1, 2.5e6, num=50): 
    y = np.array([p_RK4, m_init_RK4]) 
    
    for i in range(len(r)): 
        y_prev = y 
        y = RK4(system,y,r[i],tau) 

        if np.isnan(y[0]) == True:
            radius_RK4.append(r[i-1])
            mass_RK4.append(y_prev[1])
            break

radius_solar_RK4 = np.array(radius_RK4)*R_0_c/R_sun 
mass_solar_RK4 = np.array(mass_RK4)*M_0_c/M_sun 


#RELATIVISTIC---------------------------------------------------
radius_Rel_RK4 = [] 
mass_Rel_RK4 = []

for p_Rel_RK4 in np.geomspace(0.1, 2.5e6, num=50):
    y = np.array([p_Rel_RK4, m_init_Rel_RK4]) 
    
    for i in range(len(r)): 
        y_prev = y 
        y = RK4(RelCase,y,r[i],tau) 

        if np.isnan(y[0]) == True: 
            radius_Rel_RK4.append(r[i-1]) 
            mass_Rel_RK4.append(y_prev[1])
            break 

#Carbon Core
radius_solar_Rel_RK4 = np.array(radius_Rel_RK4)*R_0_c/R_sun
mass_solar_Rel_RK4 = np.array(mass_Rel_RK4)*M_0_c/M_sun 

#Iron Core
radius_solar_Rel_fe_RK4 = np.array(radius_Rel_RK4)*R_0_fe/R_sun
mass_solar_Rel_fe_RK4 = np.array(mass_Rel_RK4)*M_0_fe/M_sun 



#EULER METHOD--------------------------------------------------

radius_Eu = [] 
mass_Eu = [] 

for p_Eu in np.geomspace(0.1, 2.5e6, num=50): 
    y = np.array([p_Eu, m_init_Eu])                
    
    for i in range(len(r)):                            
        y_prev = y                                    
        y = Euler(system,y,r[i],tau)                    

        if np.isnan(y[0]) == True:                    
            radius_Eu.append(r[i-1])       
            mass_Eu.append(y_prev[1])
            break                                     

radius_solar_Eu = np.array(radius_Eu)*R_0_c/R_sun      
mass_solar_Eu = np.array(mass_Eu)*M_0_c/M_sun                 


plt.rcParams.update({'font.size': 20})
#Plots mass vs radius------------------------------------------
plt.plot(mass_solar_RK4,radius_solar_RK4, 'b',linewidth=3, label="4th Order RK") 
plt.plot(mass_solar_Rel_RK4,radius_solar_Rel_RK4,'c',linestyle='--',linewidth=3 ,label="Relativistic 4th Order RK (Carbon)")
plt.plot(mass_solar_Rel_fe_RK4,radius_solar_Rel_fe_RK4,'r',linestyle='--',linewidth=3, label="Relativistic 4th Order RK (Iron)")  
plt.plot(mass_solar_Eu,radius_solar_Eu, 'g',linewidth=3, label="Euler Method")
plt.axvline(x=M_limit ,linestyle='--',label="Chandrasekhar limit (1.46 M⊙)")
plt.legend()
plt.ylabel('Radius (R⊙)')
plt.xlabel('Mass (M⊙)')
plt.xlim([0, 1.5])
#plt.title("White Dwarf Radii vs Mass")
plt.show()

#Plotting vs example stars-------------------------------------
stars = np.loadtxt('wd_data.txt',usecols=(1,2,3,4))                               
star_names = np.genfromtxt('wd_data.txt', dtype=str, usecols=(0)).tolist()   

plt.plot(stars[:,0],stars[:,2], '*', label="Star Data")
plt.errorbar(x=stars[:,0],y=stars[:,2],yerr=stars[:,3],xerr=stars[:,1], fmt=' ',color='b', elinewidth=0.5)
for i in range(len(star_names)): plt.annotate(star_names[i],xy=(stars[i,0],stars[i,2]),fontsize=12)   
plt.plot(mass_solar_RK4,radius_solar_RK4, 'b',linewidth=3, label="4th Order RK")
plt.plot(mass_solar_Rel_RK4,radius_solar_Rel_RK4,'c',linestyle='--',linewidth=3, label="Relativistic 4th Order RK (Carbon)")    
plt.plot(mass_solar_Rel_fe_RK4,radius_solar_Rel_fe_RK4,'r',linestyle='--',linewidth=3, label="Relativistic 4th Order RK (Iron)")

plt.plot(mass_solar_Eu,radius_solar_Eu, 'g',linewidth=3, label="Euler Method")
plt.axvline(x=M_limit ,linestyle='--',label="Chandrasekhar limit (1.46 M⊙)")
plt.legend()
plt.ylabel('Radius (R⊙)')
plt.xlabel('Mass (M⊙)')
plt.xlim([0, 1.5])
#plt.title("White Dwarf Radii vs Mass")
plt.show()