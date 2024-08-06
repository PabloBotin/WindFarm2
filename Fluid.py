import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Fluid:

    def __init__(self, numY, ratio):

        # Sets square domain. 
        self.numY = numY # Num vertical cells. 
        self.ratio = ratio # Ratio numX/numY. 
        self.numX = numY*ratio # Num horizontal cells. 
        self.H = 1 # Height lof the domain. 
        self.L = ratio*self.H # Length of the domain. 
        self.dx = self.L / (self.numX) # Horizontal spatial step size. 
        self.dy = self.H / (self.numY) # Vertical spatial step size. 

        # Initialize arrays. 
        self.u = np.zeros((self.numY, self.numX))  # First, fill the entire array with zeros 
        self.v = np.zeros((self.numY, self.numX))  # First, fill the entire array with zeros
        self.p = np.ones((self.numY, self.numX))  # Pressure field
        self.s = np.ones((self.numY, self.numX))  # Sets Boundaries.  1:Fluid/0:solid.
        self.F = np.zeros((self.numY, self.numX)) # Body Force array. 

        # Define physical parameters 
        self.nu = 0.05
        self.rho = 1
        self.inVel = 10
        self.radio = 3
        self.A= 3.14*self.radio*self.radio
        self.Ct= 3/4

        # Define numerical parameters 
        self.walls = False
        self.Poiseuille = False
        self.AI = False
        #self.dt= .0000001 
        self.dt= self.compute_time_step(self.u, self.v, self.dx, self.dy, self.nu, cfl_number=0.2)

    def Momentum (self):
        """
        Solves the explicit form of the x-momentum and y-momentum eq for both u anv v.
        Without the pressure term!!!

        Does:
            Update u and v arrays. 
        """
        # Set parameters
        nu= self.nu
        # Initialize arrays 
        u, v, dt, dx, dy, s, walls, F = self.u, self.v, self.dt, self.dx, self.dy, self.s, self.walls, self.F
        un = u.copy()
        vn = v.copy()
    
        # X-Momentum body term is different for axial induction. Removed p-term. 
        u[1:-1, 1:-1] = ((un[1:-1, 1:-1]- 
                        s[1:-1, 1:-1]*un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                        s[1:-1, 1:-1]*vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) +
                        s[1:-1, 1:-1]* nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))+
                        F[1:-1, 1:-1]*self.dt)
        # Y-Momentum. Removed p-term.
        v[1:-1,1:-1] = (vn[1:-1, 1:-1] - 
                        s[1:-1, 1:-1]*un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        s[1:-1, 1:-1]*vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) +
                        s[1:-1, 1:-1]*nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Enforce BCs
        # BC's. Non-Slip (Dirichlet)
        # Inlet, Dirichlet BC.
        u[:, 0] = self.inVel
        v[:, 0] = 0
        # Outlet, Neumann. du/dx=0.
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        if walls == True: 
            # Bottom 
            u[0, :]  = 0
            v[0, :]  = 0
            # Top
            u[-1, :] = 0    
            v[-1, :] = 0
        else: 
            # Zero gradient (Neumann) BC. 
            # Bottom 
            u[0, :]  = u[1, :]
            v[0, :]  = v[1, :]
            # Top
            u[-1, :] = u[-2, :]    
            v[-1, :] = v[-2, :]

    def Correct(self):
        """
        Corrects u and v velocities by applying the pressure term. 

        Does:
            Update u and v arrays. 
        """
        # Initialize arrays 
        u, v, dt, rho, p = self.u, self.v, self.dt, self.rho, self.p
        un = u.copy()
        vn = v.copy()
        
        # u ← u − Δt/ρ ∂p/∂x
        dpdx = self.dp_dx(p)
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]- dt/rho*dpdx[1:-1, 1:-1])
        
        # v ← v − Δt/ρ ∂p/∂y
        dpdy = self.dp_dy(p)
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1]- dt/rho*dpdy[1:-1, 1:-1])

        # Enforce BCs. 
        # BC's. Non-Slip (Dirichlet)
        # Inlet, Dirichlet BC.
        u[:, 0] = self.inVel
        v[:, 0] = 0
        # Outlet, Neumann. du/dx=0.
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        if self.walls == True: 
            # Bottom 
            u[0, :]  = 0
            v[0, :]  = 0
            # Top
            u[-1, :] = 0    
            v[-1, :] = 0
        else: 
            # Zero gradient (Neumann) BC. 
            # Bottom 
            u[0, :]  = u[1, :]
            v[0, :]  = v[1, :]
            # Top
            u[-1, :] = u[-2, :]    
            v[-1, :] = v[-2, :]

        return 

    def build_b(self): 
        """
        Builds the b term of the poisson equation. No Force term. 

        Parameters:
            rho (float 2D array): current pressure field.

        Returns:
            p (float 2D array): updated pressure field. 
        """
        rho, dt, u, v, dx, dy = self.rho, self.dt, self.u, self.v, self.dx, self.dy
        Ct= 3/4
        b = np.zeros_like(self.u)
        b[1:-1, 1:-1] = (1 / dt * 
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                        (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                        2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                            (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2)
        return b
    
    def poisson(self, p, iters=5):
        """
        Solves the Poisson equation for pressure, which is intended to ensure a divergence free flow. 
        Does not account for the body force term. 
        
        Parameters:
            p (float 2D array): current pressure field.
            iters (int): Number of iterations. 

        Returns:
            p (float 2D array): updated pressure field. 
        """
        dx, dy, rho = self.dx, self.dy, self.rho
        # Build b term. 
        b= self.build_b()

        for i in range(iters): # Adjust number of iterations. 
            pn = p.copy()
            # Check this: 
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                            (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                            (2 * (dx**2 + dy**2)) -
                            rho * dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                            b[1:-1,1:-1])

            # Inlet. Neumann BC: dp/dx = 0 at x = 0
            p[:, 0] = p[:, 1]
            # Outlet, Dirichlet BC: p = 2 at x = 2
            p[:, -1] = 0
                
            # Top and Bottom. Neumann dp/dy=0 
            p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
            p[-1, :] = p[-2, :]   # dp/dy = 0 at y = yMax
            
        return p 

    def CFL_Check(self):
        """
        Checks if CFL condition is met and prints an error if not. 
        """ 
        cfl= self.dt / self.dx * max(np.max(np.abs(self.u)), np.max(np.abs(self.v)))
        if cfl > 1.0:
            raise ValueError("CFL condition violated. Consider reducing the time step size self.dt.")

    def Div_print(self):
        """
        Compute and print:
            1. Maximum absolute magnitude of divergence. 
            2. Summation of the absolute values of the diveregences at all cells. 
        """
        # Compute the partial derivatives
        dudx = np.gradient(self.u, axis=1) / self.dx
        dvdy = np.gradient(self.v, axis=0) / self.dy
        # Compute the divergence
        div = dudx + dvdy
        max_div = np.max(np.abs(div))
        sum_div = np.sum(np.abs(div))
        print ("Max divergence: ", max_div)
        print ("Sum divergence: ", sum_div)

    def AddSquare(self, L):
        """
        Updates the s array adding solid conditions to a square shape
        of side length L at the middle of the domain. 
        """
        # Calculate the center start indices for symmetric placement
        i = int((self.numX * 3/7) - (L / 2))
        j = (self.numY - L) // 2
        # Set the specified square area to solid (s = 0)
        self.s[j:j + L, i:i + L] = 0

    def compute_time_step(self, u, v, dx, dy, nu, cfl_number=0.1):
        """
        Compute the maximum allowable time step for a 2D incompressible Navier-Stokes simulation.
        I could use this function to automatically adjust the timestep based on the current velocity field
        It uses the msot conservative among CFL and viscosity stability conditions.  
        
        Parameters:
            nu (float): Kinematic viscosity.
            cfl_number (float): CFL number (typically < 1 for stability).
            
        Returns:
            float: Maximum allowable time step.
        """
        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        
        dt_conv_x = dx / u_max if u_max != 0 else np.inf
        dt_conv_y = dy / v_max if v_max != 0 else np.inf
        dt_diff_x = dx**2 / (2 * nu)
        dt_diff_y = dy**2 / (2 * nu)
        
        dt = cfl_number * min(dt_conv_x, dt_conv_y, dt_diff_x, dt_diff_y)
        
        return dt
        
    def Re(self):
        """
        Returns: 
            Reynolds number. Computed using the maximum absolute value of the horizontal velocity
                at the outter layer. Taken from this layer to avoid unrealistic velocities that 
                can be found at the entrance due to unsolved divergence. 
        """
        # Compute the maximum velocity at the outlet
        max_velocity = max(abs(self.u[:, -1])) # Take max velocity from the outlet layer. 
        # Compute the Reynolds number
        Re = int((self.rho * max_velocity * self.H) / self.nu)
        print ('Re: ', Re)
        return 
    
    def set_Turbine(self,y, x, F, radio): 
        """
        Parameters: 
            y (int): vertical coordinate. 
            x (int): horizontal coordinate. 
            F (float): Force magnitude. 
            radio (int): number of cells spanned by the radious. 
        """
        self.F[(y - radio):(y + radio),x] = F # (Y,X, F, #cells)

    def dp_dx(self, p):
        # Initialize dp_dx array with the same shape as P
        dp_dx = np.zeros_like(p)

        # Central difference for interior cells
        dp_dx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * self.dx)

        # Forward difference for left boundary cells
        dp_dx[:, 0] = (p[:, 1] - p[:, 0]) / self.dx

        # Backward difference for right boundary cells
        dp_dx[:, -1] = (p[:, -1] - p[:, -2]) / self.dx

        return dp_dx 

    def dp_dy(self, p):
        # Initialize dp_dx array with the same shape as P
        dp_dy = np.zeros_like(p)

        # Central difference for interior cells
        dp_dy[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * self.dy)

        # Forward difference for bottom boundary cells
        dp_dy[0, :] = (p[1, :] - p[0, :]) / self.dy

        # Backward difference for top boundary cells
        dp_dy[-1, :] = (p[-1,:] - p[-2,:]) / self.dy

        return dp_dy
    
    def Divergence(self, F): 
        """
        Computation of the divergence of a magnitude field F. 
        Parameters:
            F (float 2D array): Magnitude array. 
        Returns: 
            divF (float 2D array): divergence of F. 
        """
        # Compute the partial derivatives
        dFdx = np.gradient(self.F, axis=1) / self.dx
        dFdy = np.gradient(self.F, axis=0) / self.dy
        # Compute the divergence
        divF = dFdx + dFdy 
        return divF
    

    