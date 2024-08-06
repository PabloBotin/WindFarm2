from Fluid import Fluid
from Visualize import Visualize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def main():
    numY = 80 # Resolution. H=1 by default. 
    L = 4

    # Initialize sim.
    sim = Fluid(numY, L) # Create instance of the class. (numY, ratio)
    print ('Fluid initialized.')
    ###### Modifiable parameters ######
    steps = 1000000
    iterations = 5
    radio = 3
    sim.AI= False
    sim.walls = False
    sim.Poiseuille = False
    F_turbines = True    # Set constant force turbines. 

    ### Add (constant force) Turbines ### 
    if F_turbines == True:
        print ('Adding 3 F turbines...')
        # Assign body force to the turbine locations. 
        sim.set_Turbine(int(numY//2), int(numY), -1000, 3) # (Y,X, F, radio(#cells))
        #sim.set_Turbine(45,100, -1000, 3) # (Y,X, F, radio(#cells)) 
        #sim.set_Turbine(15,100, -1000, 3) # (Y,X, F, radio(#cells))

    #### SOLVER ####
    i=0
    p= sim.p
    for t in range(steps):
        print ('step: ', t)
        sim.dt= sim.compute_time_step(sim.u, sim.v, sim.dx, sim.dy, sim.nu, cfl_number=0.2)
        # print ('timestep: ', sim.dt) 
        sim.Momentum() # Get half-step velocities. 
        p = sim.poisson(p, iters= iterations) # Solve Poisson. 
        sim.Correct() # Correct half step velocities.  
        # Show simulation progress.  
        if (t + 1) % (steps/100) == 0:
            i= i+1
            #print (i, '/100')
            Visualize.plot_Vel(sim.u, sim.v, sim.numX, sim.numY)
            Visualize.plot_divergence(sim.u, sim.v, sim.numX, sim.numY)
            # Poiseuille validation
            if sim.Poiseuille==True:
                # Define pipe slices. 
                u0 = sim.u[:, 0] # 1D velocity array at x=0
                u1 = sim.u[:, 2*sim.numY-1] # 1D velocity array at x=H
                u2= sim.u[:, 4*sim.numY-1] # 1D velocity array at x=10H
                dpdx= (sim.p[int(numY/2), -2]-sim.p[int(numY/2), -1])/sim.dx # Compute pressure gradient at outlet. 
                Visualize.PoiseuilleValidation (u0, u1, u2, sim.numY, dpdx)
                #dpdx= sim.dp_dx(sim.p) # Compute the pressure gradient field. 
                #Visualize.plot_dPdx(dpdx, sim.numX, sim.numY)

if __name__ == "__main__": # Only run the functions defined in this code. 
    main()
