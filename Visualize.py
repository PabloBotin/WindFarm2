import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Visualize:

    def __init__(self, fluid_instance):
        self.fluid_instance = fluid_instance

    def plot_Turbine(p, u, v, nx, ny, AI):
        L=1
        Ratio=nx/ny
        x = np.linspace(0, Ratio*L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        plot_type = "quiver"  # "quiver" or "streamplot"
        
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        
        # Plotting the pressure field as a contour
        contourf = ax.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
        fig.colorbar(contourf, ax=ax)
        
        # Plotting the pressure field outlines
        ax.contour(X, Y, p, cmap=cm.viridis)
        
        # Plotting velocity field
        if plot_type == "quiver":
            ax.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
        elif plot_type == "streamplot":
            ax.streamplot(X, Y, u, v)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Mask and overlay the region with non-zero AI values
        masked_AI = np.ma.masked_where(AI == 0, AI)
        black_cmap = ListedColormap(['black'])
        
        ax.imshow(masked_AI, cmap=black_cmap, origin='lower', alpha=1, extent=[0, 2, 0, 2])
        
        plt.show()

    def plot_Quiver(p, u, v, nx, ny, s):
        L = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        plot_type = "quiver"  # "quiver" or "streamplot"
        
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        
        # Plotting the pressure field as a contour
        contourf = ax.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
        
        # Creating a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Adding the colorbar
        fig.colorbar(contourf, cax=cax)
        
        # Plotting the pressure field outlines
        ax.contour(X, Y, p, cmap=cm.viridis)
        
        # Plotting velocity field
        if plot_type == "quiver":
            ax.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
        elif plot_type == "streamplot":
            ax.streamplot(X, Y, u, v)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Mask and overlay the region
        masked_s = np.ma.masked_where(s != 0, s)
        black_cmap = ListedColormap(['orange'])
        
        # Adjust extent to match the domain
        extent = [0, Ratio * L, 0, L]
        ax.imshow(masked_s, cmap=black_cmap, origin='lower', alpha=1, extent=extent)
        
        # Ensure the plot has an equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.show()

    def plot_Vel(u, v, nx, ny):
        L = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        
        # Compute the velocity magnitude
        velocity_magnitude = np.sqrt(u**2 + v**2)
        
        # Plotting the velocity field as colored cells
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        c = ax.pcolormesh(X, Y, velocity_magnitude, shading='auto', cmap=cm.viridis)
        
        # Creating a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Adding the color bar with a label
        fig.colorbar(c, cax=cax).set_label('Velocity Magnitude')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Velocity Field')
        
        # Ensuring the plot has an equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.show()

    def plot_Vel_Black(u, v, nx, ny):
        L = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        
        # Compute the velocity magnitude
        velocity_magnitude = np.sqrt(u**2 + v**2)
        
        # Create a figure with a black background
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Plotting the velocity field as colored cells with a contrasting colormap
        c = ax.pcolormesh(X, Y, velocity_magnitude, shading='auto', cmap=cm.plasma)
        
        # Creating a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Adding the color bar with a label and white color for text
        cbar = fig.colorbar(c, cax=cax)
        cbar.set_label('Velocity Magnitude', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Setting labels and title with white color
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_title('Velocity Field', color='white')
        
        # Setting axis tick parameters with white color
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # Ensuring the plot has an equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.show()

    def plot_divergence(u, v, nx, ny):
        """
        Plots the 2D divergence field given the velocity components u and v.
        """
        L = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        dx = L / (nx - 1)
        dy = L / (ny - 1)
        
        # Compute the partial derivatives
        dudx = np.gradient(u, axis=1) / dx
        dvdy = np.gradient(v, axis=0) / dy
        
        # Compute the divergence
        divergence = dudx + dvdy
        
        # Plot the divergence field
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        cp = ax.contourf(X, Y, divergence, 20, cmap='viridis')
        
        # Creating a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Adding the color bar with a label
        fig.colorbar(cp, cax=cax).set_label('Divergence')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('2D Divergence Field')
        
        # Ensuring the plot has an equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.show()

    def plot_divergence_Black(u, v, nx, ny):
        """
        Plots the 2D divergence field given the velocity components u and v.
        """
        L = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        dx = L / (nx - 1)
        dy = L / (ny - 1)
        
        # Compute the partial derivatives
        dudx = np.gradient(u, axis=1) / dx
        dvdy = np.gradient(v, axis=0) / dy
        
        # Compute the divergence
        divergence = dudx + dvdy
        
        # Create a figure with a black background
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Plot the divergence field with a contrasting colormap
        cp = ax.contourf(X, Y, divergence, 20, cmap=cm.plasma)
        
        # Creating a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Adding the color bar with a label and white color for text
        cbar = fig.colorbar(cp, cax=cax)
        cbar.set_label('Divergence', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Setting labels and title with white color
        ax.set_xlabel('x', color='white')
        ax.set_ylabel('y', color='white')
        ax.set_title('2D Divergence Field', color='white')
        
        # Setting axis tick parameters with white color
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # Ensuring the plot has an equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.show()

    def PoiseuilleValidation (u_0, u_1, u_2, ny, dpdx):
        def compute_analytical_velocity(y, h, mu, dpdx):
            return (1 / (2 * mu)) * dpdx * (h**2 - y**2)

        def plot_velocity_profiles(y, u_analytical, u_0, u_1, u_2):
            plt.figure(figsize=(10, 6))
            plt.plot(u_analytical, y, label='Analytical Solution', linewidth=5)
            plt.plot(u_0, y, label='Numerical Solution at x=0', linewidth=4)
            plt.plot(u_1, y, label='Numerical Solution at x=2', linewidth=4)
            plt.plot(u_2, y, label='Numerical Solution at x=4', linewidth=4)
            plt.xlabel('Velocity $u$')
            plt.ylabel('Position $y$')
            plt.title('Comparison of Analytical and Numerical Velocity Profiles')
            plt.legend()
            plt.grid(True)
            plt.show()

        # Parameters
        h2 = 1/2  # Half-distance between the plates
        mu = 0.5  # Dynamic viscosity
        y = np.linspace(-h2, h2, ny)  # y-coordinates
        # Overall pressure drop from your simulation
        # Compute analytical solution
        u_analytical = compute_analytical_velocity(y, h2, mu, dpdx)
        # Plotting the velocity profiles
        plot_velocity_profiles(y, u_analytical, u_0, u_1, u_2)

    def Poiseuille_L2 (u_2, ny, dpdx):
        def compute_analytical_velocity(y, h, mu, dpdx):
            return (1 / (2 * mu)) * dpdx * (h**2 - y**2)

        def compute_L2_error(u_analytical, u_numerical):
            return np.sqrt(np.sum((u_analytical - u_numerical)**2) / len(u_analytical))
        
        # Parameters
        h2 = 1/2  # Half-distance between the plates
        mu = 0.5  # Dynamic viscosity
        y = np.linspace(-h2, h2, ny)  # y-coordinates
        # Compute analytical solution
        u_analytical = compute_analytical_velocity(y, h2, mu, dpdx)
        # Compute the L2 error and return. 
        return compute_L2_error(u_analytical, u_2)

    def plot_Turbine(p, u, v, nx, ny, AI):
        L = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * L, nx)
        y = np.linspace(0, L, ny)
        X, Y = np.meshgrid(x, y)
        plot_type = "quiver"  # "quiver" or "streamplot"

        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)

        # Plotting the pressure field as a contour
        contourf = ax.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
        fig.colorbar(contourf, ax=ax)

        # Plotting the pressure field outlines
        ax.contour(X, Y, p, cmap=cm.viridis)

        # Plotting velocity field
        if plot_type == "quiver":
            ax.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
        elif plot_type == "streamplot":
            ax.streamplot(X, Y, u, v)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Mask and overlay the region with non-zero AI values
        masked_AI = np.ma.masked_where(AI == 0, AI)
        black_cmap = ListedColormap(['black'])

        # Ensure the extent covers the entire plotting area
        ax.imshow(masked_AI, cmap=black_cmap, origin='lower', alpha=1, extent=[0, Ratio * L, 0, L], aspect='auto')

        plt.show()

    def plot_F(M, nx, ny):
        """
        Parameters: 
            - ny (int): vertical number of cells. 
            - nx (int): horizontal number of cells. 
            - M (2D float array): Magnitude field to be visualized.  
        Does: 
            - Field visualization of M.  
        """
        H = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * H, nx)
        y = np.linspace(0, H, ny)
        X, Y = np.meshgrid(x, y)
        
        # Plotting the velocity field as colored cells
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        c = ax.pcolormesh(X, Y, M, shading='auto', cmap=cm.viridis)
        
        # Creating a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Adding the color bar with a label
        fig.colorbar(c, cax=cax).set_label('dp/dx')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Force Field')
        
        # Ensuring the plot has an equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.show()

    def plot_divF(divF, nx, ny):
        """
        Parameters: 
            - ny (int): vertical number of cells. 
            - nx (int): horizontal number of cells. 
            - divF (2D float array): Divergence of force field.  
        Does: 
            - Field visualization of Force diveregence.  
        """
        H = 1
        Ratio = nx / ny
        x = np.linspace(0, Ratio * H, nx)
        y = np.linspace(0, H, ny)
        X, Y = np.meshgrid(x, y)
        
        # Plotting the velocity field as colored cells
        fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
        c = ax.pcolormesh(X, Y, divF, shading='auto', cmap=cm.viridis)
        
        # Creating a divider for existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Adding the color bar with a label
        fig.colorbar(c, cax=cax).set_label('dp/dx')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Force diveregence field')
        
        # Ensuring the plot has an equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.show()