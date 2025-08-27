# ===============================================================
#                           IMPORTS
# ===============================================================

# Numerical and array operations
import numpy as np  
# - Provides arrays (like lists but faster and more efficient)
# - Provides mathematical operations: sqrt, sin, cos, linear algebra, etc.

# Data handling and manipulation
import pandas as pd  
# - Provides DataFrame, a table-like structure
# - Makes it easy to read, manipulate, and analyze data from CSV/VTK files

# Plotting and visualization
import matplotlib.pyplot as plt  
# - Standard 2D plotting library in Python
# - Used for line plots, scatter plots, contour plots, etc.

# 3D plotting in Matplotlib
from mpl_toolkits.mplot3d import Axes3D  
# - Enables 3D plotting capabilities in Matplotlib
# - Allows creation of 3D scatter plots, surface plots, and quiver plots

# Visualization and handling of 3D meshes
import pyvista as pv  
# - Specialized library for reading, visualizing, and manipulating VTK/mesh data
# - Converts 3D point clouds or meshes into Python-friendly objects

# Continuous Wavelet Transform library
import pycwt as wavelet  
# - Used for spectral (frequency) analysis of data over space or time
# - Performs wavelet transform to see how frequency content changes

# Helper functions from PyCWT library
from pycwt.helpers import find  
# - Provides utility functions for wavelet analysis
# - E.g., to find indices or frequencies in wavelet output

# GUI for file dialogs and popups
from tkinter import filedialog, Tk, messagebox  
# - Tkinter is Python’s built-in GUI library
# - filedialog: open/save file popups
# - Tk: the main GUI window (needed to use dialogs)
# - messagebox: show pop-up messages like warnings, errors, info

# Operating system interactions
import os  
# - Provides functions to handle file paths, directories, and system operations
# - E.g., os.path.basename to get file name without path

# Colormaps for plotting
from matplotlib import cm  
# - Provides color maps for visualizations
# - Used for mapping values to colors (like vorticity or shear magnitude)

# Efficient spatial searches (k-nearest neighbors) in 3D
from scipy.spatial import cKDTree  
# - Builds a tree for fast nearest-neighbor searches in multidimensional space
# - Used to find closest points for computing gradients or vorticity

# Machine learning nearest neighbor search
from sklearn.neighbors import NearestNeighbors  
# - Another method to find k-nearest neighbors
# - Used in shear layer and pressure gradient computations

# Interpolation for scattered data
from scipy.interpolate import griddata  
# - Converts unstructured data points onto a regular grid
# - Used for contour plots, computing derivatives, and visualization

# Fast Fourier Transform (FFT) operations
from scipy.fft import fft, fftfreq  
# - fft: compute the frequency spectrum of data
# - fftfreq: generate corresponding frequency values for FFT

# Linear algebra tools
from numpy.linalg import lstsq  
# - Solves linear systems using least squares
# - Used to compute gradients, fit planes, or approximate derivatives

# Sparse matrix operations
from scipy.sparse import diags, kron, identity  
# - diags: create diagonal matrices efficiently
# - kron: Kronecker product, used to build 2D Laplacian from 1D
# - identity: identity matrix (for linear algebra operations)

# Solve sparse linear systems
from scipy.sparse.linalg import spsolve  
# - Solves large sparse linear equations efficiently
# - Used for computing 2D streamfunction from vorticity


# ---------------------------------------------------------------
#           INITIALIZE A SINGLE HIDDEN GUI ROOT WINDOW
# ---------------------------------------------------------------
# Tkinter requires a 'root' window to open file dialogs.
# We create it once and hide it immediately (withdraw) because we
# don't need a full GUI window, only the file dialog popup.
root = Tk()       # create main Tkinter window
root.withdraw()   # hide the main window


# ---------------------------------------------------------------
#               FUNCTION: load_file
# ---------------------------------------------------------------
# Purpose:
# - Opens a file dialog for the user to select a .vtk or .csv file
# - Returns the file path as a string
# - Shows a warning if no file is selected

def load_file():
    # Open a standard file selection dialog
    # filetypes = filter to only show .vtk or .csv files
    file_path = filedialog.askopenfilename(filetypes=[("VTK or CSV", "*.vtk *.csv")])
    
    # If user cancels or closes the dialog, file_path will be empty
    if not file_path:
        # Show a small popup warning the user to select a file
        messagebox.showwarning("No file selected", "Please select a .vtk or .csv file.")
        return None  # exit function without a path
    
    # Return the path of the selected file
    return file_path


# ---------------------------------------------------------------
#               FUNCTION: load_data
# ---------------------------------------------------------------
# Purpose:
# - Reads the selected file and converts it into a Pandas DataFrame
# - Handles both VTK (3D mesh) and CSV files
# - Returns a structured table with columns like x, y, z, u, v, w, velocity, etc.

def load_data(file_path):
    # If the file is a VTK file (3D mesh)
    if file_path.endswith(".vtk"):
        mesh = pv.read(file_path)  # read VTK using PyVista
        
        # Extract the point coordinates into a dictionary
        data = {
            'x': mesh.points[:, 0],  # all x coordinates
            'y': mesh.points[:, 1],  # all y coordinates
            'z': mesh.points[:, 2],  # all z coordinates
        }
        
        # Include any extra data attached to each point, like velocity vectors
        # mesh.point_data could include arrays named 'u','v','w' or 'velocity'
        for key in mesh.point_data:
            data[key] = mesh.point_data[key]
        
        # Convert dictionary to Pandas DataFrame for easier processing later
        return pd.DataFrame(data)
    
    # If the file is a CSV file (tabular data)
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)  # directly load CSV as DataFrame
    
    # If the file type is not supported, raise an error
    else:
        raise ValueError("Unsupported file type.")


# ---------------------------------------------------------------
#               FUNCTION: is_3D_data
# ---------------------------------------------------------------
# Purpose:
# - Determines if the dataset represents 3D flow
# - Checks if there is a 'z' column with multiple unique values
# - Also requires a 'w' velocity component for 3D flow
# - Returns True if the data is 3D, otherwise False

def is_3D_data(df):
    # Check if 'z' column exists AND has more than 1 unique value (i.e., variation in z)
    # AND if 'w' velocity component exists
    return ("z" in df.columns and df["z"].nunique() > 1) and ("w" in df.columns)

# ===============================================================
#                       VORTICITY FUNCTIONS
# ===============================================================

def compute_2D_vorticity(df, grid_n=100):
    """
    Function: compute_2D_vorticity
    -------------------------------
    Calculates **vorticity** in a 2D flow field.
    
    This function:
    1. Takes raw velocity data (u and v) at scattered points.
    2. Places it on a nice, regular square grid.
    3. Calculates vorticity = (rate of change of v in x direction) - (rate of change of u in y direction).
    4. Solves for something called a **streamfunction** (think of it as
       smooth curves that represent the path fluid elements follow).
    5. Returns everything so we can make pretty plots.

    Parameters:
        df : Pandas DataFrame
            Must contain columns:
                'x', 'y' → coordinates of each data point
                'u', 'v' → velocity components in x and y directions
        grid_n : int (default = 100)
            Number of grid points (resolution of the interpolation)

    Returns:
        X, Y : 2D meshgrid arrays (grid of x and y coordinates)
        vorticity : 2D array of vorticity values at each grid cell
        U, V : velocity components interpolated onto the grid
        streamfunction : 2D array representing flow streamlines
    """
    # Extract x and y coordinates from the dataframe
    x = df["x"].values
    y = df["y"].values
    
    # Check that both velocity components are present in the data.
    # If either 'u' or 'v' is missing, stop and give an error.
    if not all(c in df.columns for c in ("u", "v")):
        raise ValueError("u and v required for 2D vorticity.")
    
    # Get velocity values
    u = df["u"].values
    v = df["v"].values

    # Create a regular "grid" (like graph paper) that spans the entire dataset
    grid_x = np.linspace(x.min(), x.max(), grid_n)  # evenly spaced x-coordinates
    grid_y = np.linspace(y.min(), y.max(), grid_n)  # evenly spaced y-coordinates
    X, Y = np.meshgrid(grid_x, grid_y)             # build full 2D grid of points

    # Put the scattered velocity data (u, v) onto the grid
    U = griddata((x, y), u, (X, Y), method='cubic')   # interpolate u
    V = griddata((x, y), v, (X, Y), method='cubic')   # interpolate v

    # If cubic interpolation fails (leaves gaps), fall back to linear interpolation
    if np.any(np.isnan(U)) or np.any(np.isnan(V)):
        U = griddata((x, y), u, (X, Y), method='linear')
        V = griddata((x, y), v, (X, Y), method='linear')

    # Compute the rate of change (gradient) of U and V
    # dV_dy = change of v with respect to y
    # dV_dx = change of v with respect to x
    # dU_dy = change of u with respect to y
    # dU_dx = change of u with respect to x
    dV_dy, dV_dx = np.gradient(V, grid_y, grid_x, edge_order=2)
    dU_dy, dU_dx = np.gradient(U, grid_y, grid_x, edge_order=2)

    # Vorticity in 2D: ω = dv/dx - du/dy
    vorticity = dV_dx - dU_dy

    # ---------------------------------------------------------------
    # STREAMFUNCTION CALCULATION
    # ---------------------------------------------------------------
    # The streamfunction (ψ) satisfies the equation:
    #    ∇²ψ = -ω
    # where ∇² is the Laplacian operator (like applying "curvature").
    #
    # This is a way to turn vorticity into smooth flow lines we can plot.
    #
    # We solve this equation numerically using sparse matrices.
    # ---------------------------------------------------------------

    ny, nx = vorticity.shape  # grid dimensions
    dx = grid_x[1] - grid_x[0] if nx > 1 else 1.0
    dy = grid_y[1] - grid_y[0] if ny > 1 else 1.0

    # Build Laplacian matrices in x and y directions (second derivatives)
    ex = np.ones(nx)
    ey = np.ones(ny)
    Lx = diags([ex * (-2.0), ex[:-1], ex[:-1]], [0, -1, 1], shape=(nx, nx)) / dx**2
    Ly = diags([ey * (-2.0), ey[:-1], ey[:-1]], [0, -1, 1], shape=(ny, ny)) / dy**2

    # Identity matrices (like "do nothing" operators)
    Ix = identity(nx)
    Iy = identity(ny)
    
    # Full 2D Laplacian operator (combination of x and y second derivatives)
    Lap = kron(Iy, Lx) + kron(Ly, Ix)  # Kronecker sum = 2D Laplacian

    # Flatten vorticity into a single vector so we can solve Ax = b
    rhs = -vorticity.flatten()
    
    # Solve the linear system: Lap * ψ = -ω
    psi_flat = spsolve(Lap.tocsr(), rhs)
    
    # Reshape ψ back into a 2D grid
    streamfunction = psi_flat.reshape((ny, nx))

    # Return all the results
    return X, Y, vorticity, U, V, streamfunction


def compute_3D_vorticity(df, k=10):
    """
    Function: compute_3D_vorticity
    -------------------------------
    Calculates vorticity in a **3D flow field**.

    Approach used here:
    1. Build a "KD-Tree" to quickly find nearest neighbors for each point.
       (Think: for every point, we look at its closest neighbors.)
    2. Estimate velocity gradients using a least-squares method.
    3. Compute the curl formula:
         ωx = dw/dy - dv/dz
         ωy = du/dz - dw/dx
         ωz = dv/dx - du/dy
    4. Store results and also calculate the magnitude (overall strength) of vorticity.

    Parameters:
        df : Pandas DataFrame
            Must contain 'x','y','z','u','v','w'
            (3D positions + velocity components)
        k : int (default = 10)
            Number of nearest neighbors used in gradient calculation.

    Returns:
        x, z : coordinate arrays (for 2D slice plotting)
        vort_x, vort_z : x and z components of vorticity
        vort_mag : magnitude of the vorticity vector
    """
    # Extract coordinates and velocities
    x = df["x"].values
    y = df["y"].values
    z = df["z"].values
    u = df["u"].values
    v = df["v"].values
    w = df["w"].values

    # Combine into arrays
    points = np.vstack((x, y, z)).T        # each row = (x, y, z)
    velocity = np.vstack((u, v, w)).T      # each row = (u, v, w)
    
    # Build KD-tree (data structure for fast neighbor searching)
    tree = cKDTree(points)
    npts = len(points)
    k = min(k, npts)  # make sure k isn’t larger than total points

    # Prepare storage for vorticity values
    vorticity = np.zeros_like(velocity)

    # Loop over each point in the dataset
    for i in range(npts):
        # Find the k nearest neighbors
        dists, idx = tree.query(points[i], k=k)
        
        # If not enough neighbors, skip this point
        if len(idx) < 4:
            vorticity[i] = 0.0
            continue

        # Displacements of neighbors relative to this point
        A = points[idx] - points[i]       # (k,3) positions
        B = velocity[idx] - velocity[i]   # (k,3) velocities

        # Try to solve A * Grad ≈ B (least squares fit for velocity gradient)
        try:
            Grad, *_ = np.linalg.lstsq(A, B, rcond=None)  # 3x3 gradient tensor
            
            # Compute curl (vorticity vector) from gradient tensor
            curl = np.array([
                Grad[2, 1] - Grad[1, 2],   # dw/dy - dv/dz
                Grad[0, 2] - Grad[2, 0],   # du/dz - dw/dx
                Grad[1, 0] - Grad[0, 1],   # dv/dx - du/dy
            ])
            vorticity[i] = curl
        except Exception:
            vorticity[i] = 0.0  # if solver fails, just set zero

    # Compute vorticity magnitude (length of vector)
    vort_mag = np.linalg.norm(vorticity, axis=1)

    # Return values (mainly x,z for slice visualization)
    return x, z, vorticity[:, 0], vorticity[:, 2], vort_mag
# ===============================================================
# VORTICITY PLOTTING FUNCTIONS
# ===============================================================

def plot_2D_vorticity(X, Y, vorticity, U, V, streamfunction, save_path=None):
    """
    Function: plot_2D_vorticity
    ----------------------------
    Creates a visual plot of **2D vorticity** and flow streamlines.

    Inputs:
        X, Y : 2D coordinate grids
        vorticity : spinning strength at each grid point
        U, V : velocity components on the grid
        streamfunction : used to draw streamlines (like a map of flow paths)
        save_path : optional string
            If provided, saves the plot to this file instead of just showing it.
    """

    plt.figure(figsize=(8, 6))   # make a new figure with a good size

    # Plot the vorticity field as a color background
    plt.contourf(X, Y, vorticity, levels=50, cmap="RdBu_r")  
    #   contourf = "filled contour plot"
    #   RdBu_r = red-to-blue color scheme (good for showing positive vs negative)

    # Add streamlines (like flow paths) on top of the vorticity field
    plt.streamplot(X, Y, U, V, color="k", density=1, linewidth=0.5)
    #   density = how many lines
    #   color="k" = black lines
    #   streamplot automatically uses U and V (velocities)

    # Add labels and color bar
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    plt.title("2D Vorticity with Streamlines")
    plt.colorbar(label="Vorticity (ω)")   # side legend showing vorticity values

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # save high-quality image
        plt.close()
    else:
        plt.show()


def plot_3D_vorticity(x, z, vort_x, vort_z, vort_mag, save_path=None):
    """
    Function: plot_3D_vorticity
    ----------------------------
    Creates a **2D slice view** (picking two directions (here: x and z) and showing how vorticity looks in that plane) of 3D vorticity.

    What we do here:
    - Represent vorticity using small arrows that show its direction (x,z components).
    - Color the arrows by **magnitude** (overall strength of rotation).

    Inputs:
        x, z : coordinates of data points
        vort_x, vort_z : x and z components of vorticity vector
        vort_mag : strength (length) of vorticity vector
        save_path : optional string, if provided saves figure to file
    """

    plt.figure(figsize=(8, 6))

    # Use quiver = arrow plot
    plt.quiver(x, z, vort_x, vort_z, vort_mag, cmap="inferno", scale=50)
    #   x,z = positions of arrows
    #   vort_x, vort_z = direction of each arrow
    #   vort_mag = color (strength of spin)
    #   scale=50 makes arrows not too long

    # Labels and title
    plt.xlabel("x-position")
    plt.ylabel("z-position")
    plt.title("3D Vorticity Slice (x-z plane)")
    plt.colorbar(label="|ω| (Vorticity Magnitude)")  # side legend for magnitude

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# ===============================================================
#                   PRESSURE GRADIENT FUNCTIONS
# ===============================================================

def compute_2D_pressure_gradient(df):
    """
    Function: compute_2D_pressure_gradient
    --------------------------------------
    Calculates how fast pressure changes along the x-direction in 2D.

    Inputs:
        df : Pandas DataFrame
            Must have columns:
                'x' → x-coordinate of each data point
                'p' → pressure at that point

    Returns:
        x_unique : sorted x positions
        dp_dx : derivative of pressure with respect to x
    """

    try:
        # Check that required columns exist
        if not all(col in df.columns for col in ["x", "p"]):
            raise ValueError("Missing required columns for 2D pressure gradient: x, p.")

        # Remove duplicate x-values and sort
        df_sorted = df.sort_values("x").drop_duplicates(subset=["x"])
        x_unique = df_sorted["x"].values
        p_unique = df_sorted["p"].values

        # Need at least two points to calculate a gradient
        if len(x_unique) < 2:
            raise ValueError("Need at least 2 points for pressure gradient.")

        # Compute gradient using numpy (dp/dx)
        dp_dx = np.gradient(p_unique, x_unique)  # automatically accounts for uneven spacing

        return x_unique, dp_dx

    except Exception as e:
        # If something fails, show a message and return None
        messagebox.showinfo("2D Pressure Gradient", f"Skipped: {e}")
        return None, None


def compute_3D_pressure_gradient(df, k=6):
    """
    Function: compute_3D_pressure_gradient
    --------------------------------------
    Estimates pressure gradient magnitude at each point in a 3D dataset.

    Uses nearest neighbors and plane fitting:
        1. Find k nearest points for each data point
        2. Fit a plane: p = a*x + b*y + c*z + d
        3. Gradient magnitude = sqrt(a^2 + b^2 + c^2)

    Inputs:
        df : Pandas DataFrame
            Must have columns: 'x', 'y', 'z', 'p'
        k : int, default=6
            Number of neighbors to use for gradient calculation

    Returns:
        x : original x coordinates
        gradmag : estimated pressure gradient magnitude at each point
    """

    coords = df[['x','y','z']].values  # positions
    p = df['p'].values                 # pressure values
    n = len(df)
    k = min(k, n)                      # cannot use more neighbors than points

    # Find k nearest neighbors for each point
    nbr = NearestNeighbors(n_neighbors=k).fit(coords)
    ind = nbr.kneighbors(coords, return_distance=False)

    # Prepare storage for gradient magnitudes
    gradmag = np.zeros(n)

    for i, idx in enumerate(ind):
        # Fit a plane: p = a*x + b*y + c*z + d using least squares
        A = np.c_[coords[idx], np.ones(len(idx))]  # append ones for constant term
        try:
            coeffs, *_ = np.linalg.lstsq(A, p[idx], rcond=None)  # coefficients = [a, b, c, d]
            gradmag[i] = np.linalg.norm(coeffs[:3])             # magnitude = sqrt(a^2+b^2+c^2)
        except Exception:
            gradmag[i] = 0.0  # if fitting fails, assume zero gradient

    return df['x'].values, gradmag


def plot_2D_pressure_gradient(x, grad):
    """
    Function: plot_2D_pressure_gradient
    -----------------------------------
    Creates a line plot showing how pressure changes along x.

    Inputs:
        x : x positions
        grad : pressure gradient (dp/dx) at each x

    What it shows:
    - Peaks in the plot = regions where pressure changes quickly
    - Flat regions = pressure nearly constant
    """
    if x is None or grad is None:
        return  # do nothing if data is missing

    plt.figure()
    plt.plot(x, grad, '-o', color='blue', markersize=4)
    plt.title("Pressure Gradient Magnitude (2D)")
    plt.xlabel("x-position")
    plt.ylabel("|dp/dx|")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_3D_pressure_gradient(x, grad):
    """
    Function: plot_3D_pressure_gradient
    -----------------------------------
    Creates a scatter plot showing magnitude of pressure gradient in 3D space.

    Inputs:
        x : x-coordinate (or any chosen direction)
        grad : pressure gradient magnitude

    Visualization:
    - Points are colored according to gradient magnitude
    - Bright colors = high pressure change
    """
    plt.figure()
    sc = plt.scatter(x, grad, c=grad, cmap='viridis', edgecolor='k')
    plt.colorbar(sc, label='|∇p|')
    plt.xlabel('X')
    plt.ylabel('|∇p|')
    plt.title('3D Pressure Gradient Magnitude')
    plt.tight_layout()
    plt.show()
# ===============================================================
#                   SPECTRAL ANALYSIS FUNCTIONS
# ===============================================================

def compute_2D_spectral_analysis(df):
    """
    Function: compute_2D_spectral_analysis
    --------------------------------------
    Purpose:
    - Analyze how a flow's speed varies along a line (x-direction) in 2D.
    - Detect repeating patterns or "waves" in the velocity data.
    - Imagine a river: water speed changes as you move along the river.
    - FFT (Fast Fourier Transform) finds the main repeating patterns.
    - Wavelets show how these patterns change at different positions along the river.

    Inputs:
        df : Pandas DataFrame
            Must have:
                'x' → position along a line
                'u' → velocity component along x

    Outputs:
        Generates plots (FFT and Wavelet) to visualize patterns in the data.
    """

    # Convert DataFrame columns to NumPy arrays for faster calculations
    x = df['x'].to_numpy()  # x positions
    u = df['u'].to_numpy()  # velocity values

    # Sort x so the positions increase from left to right
    # Sorting ensures that the FFT and wavelet transformations make sense
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    u = u[sort_idx]

    # Check that there are enough points for meaningful analysis
    # Less than 8 points won't give useful frequency information
    if len(u) < 8:
        print("Too few points for meaningful 2D FFT.")
        return

    # Compute average spacing between points
    # This spacing is needed for frequency calculations
    dx = np.mean(np.diff(x))  

    # ========================
    # Fourier Transform (FFT)
    # ========================
    # FFT converts the velocity signal from space domain (x) into frequency domain
    # It tells us which "repeating patterns" are present and their strength
    fft_vals = np.fft.fft(u)                 # transform the velocity
    freqs = np.fft.fftfreq(len(u), d=dx)     # corresponding frequencies
    power = np.abs(fft_vals)**2               # power = magnitude squared of FFT

    # Plot FFT results
    plt.figure()
    plt.plot(freqs[:len(freqs)//2], power[:len(freqs)//2], color='darkblue')
    plt.xlabel("Frequency (cycles per unit distance)")
    plt.ylabel("Power (strength of pattern)")
    plt.title("2D Fourier Spectrum")
    plt.grid(True)
    plt.show()

    # ========================
    # Wavelet Transform
    # ========================
    # Wavelets allow us to see how the frequency of patterns changes along x
    import pycwt as wavelet
    mother = wavelet.Morlet(6)  # Choose a Morlet wavelet (a wave-like function)
    dt = dx                     # spacing between x points
    s0 = 2 * dt                 # smallest scale (shortest wavelength we care about)
    dj = 0.25                    # resolution in scales
    J = 7 / dj                   # number of scales

    # Compute the continuous wavelet transform (CWT)
    # wave → complex numbers representing pattern strength at each x and scale
    # scales → different "sizes" of wavelets
    # freqs_wavelet → corresponding frequencies
    wave, scales, freqs_wavelet, coi, fft_wave, fftfreqs_wavelet = wavelet.cwt(u, dt, dj, s0, J, mother)
    power_wavelet = np.abs(wave)**2  # power = magnitude squared

    # Prepare grids for plotting: x positions vs log2(scale)
    T, S = np.meshgrid(x, np.log2(scales))

    # Plot wavelet power spectrum
    plt.figure()
    plt.contourf(T, S, np.log2(power_wavelet), 100, cmap='jet')  
    # contourf = filled contour plot
    # log2(power_wavelet) = use logarithm to better see small and large patterns
    plt.xlabel("x-position")
    plt.ylabel("Log2(Scale) (smaller = short waves, larger = long waves)")
    plt.title("2D Wavelet Power Spectrum")
    plt.colorbar(label='Log2 Power (pattern strength)')
    plt.show()


def compute_3D_spectral_analysis(df):
    """
    Function: compute_3D_spectral_analysis
    --------------------------------------
    Purpose:
    - Analyze flow patterns in 3D along the x-direction.
    - Because 3D data is large, we average velocity along y and z to create a 1D signal.
    - Perform FFT and Wavelet to see repeating structures and how they vary along x.

    Inputs:
        df : Pandas DataFrame
            Must include columns: 'x','y','z','u'

    Outputs:
        Plots FFT and Wavelet power spectrum for the averaged velocity.
    """

    # Extract positions and velocity
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    u = df['u'].values

    # Remove invalid or missing values
    mask = ~np.isnan(u) & np.isfinite(u)
    x, y, z, u = x[mask], y[mask], z[mask], u[mask]

    # Average u along y-z planes to simplify data into 1D
    bins = np.linspace(x.min(), x.max(), 100)  # 100 bins along x
    u_mean = np.zeros_like(bins)              # store averaged u
    counts = np.zeros_like(bins)              # count number of points per bin

    for i in range(len(u)):
        idx = np.argmin(np.abs(bins - x[i]))  # find closest bin for this point
        u_mean[idx] += u[i]                   # sum velocities
        counts[idx] += 1                      

    counts[counts == 0] = 1    # avoid division by zero
    u_mean /= counts           # compute average in each bin

    if len(u_mean) < 10:
        print("Too few usable points after cleaning.")
        return

    dx = bins[1] - bins[0]    # spacing along x

    # ========================
    #           FFT
    # ========================
    fft_vals = np.fft.fft(u_mean - np.mean(u_mean))  # subtract mean to remove bias
    fft_freq = np.fft.fftfreq(len(u_mean), d=dx)     # corresponding frequencies
    fft_power = np.abs(fft_vals)**2                  # power of each frequency

    # ========================
    #           Wavelet
    # ========================
    import pycwt as wavelet
    dt = dx
    dj = 0.25
    s0 = 2 * dt

    try:
        wave, scales, freqs, coi, fft_res, fftfreqs = wavelet.cwt(u_mean, dt, dj, s0)
        power = np.abs(wave)**2
    except Exception as e:
        print("Wavelet error:", e)
        return

    # ========================
    # Plot FFT and Wavelet
    # ========================
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Top subplot: FFT
    axs[0].plot(fft_freq, fft_power)
    axs[0].set_xlim(0, max(fft_freq))
    axs[0].set_title("3D FFT Power Spectrum")
    axs[0].set_xlabel("Frequency (cycles/unit distance)")
    axs[0].set_ylabel("Power (pattern strength)")

    # Bottom subplot: Wavelet
    im = axs[1].contourf(bins, np.log2(scales), power, cmap='viridis')
    axs[1].set_title("Wavelet Power Spectrum")
    axs[1].set_ylabel("Scale (log2)")
    axs[1].set_xlabel("x-position")
    plt.colorbar(im, ax=axs[1])
    plt.tight_layout()
    plt.show()


def perform_spectral_analysis(df):
    """
    Function: perform_spectral_analysis
    -----------------------------------
    Automatically decides if the data is 2D or 3D and runs the correct analysis.

    Logic:
    - If a 'z' column exists with more than one unique value → 3D
    - Otherwise → 2D
    """
    is_3d = "z" in df.columns and df["z"].nunique() > 1
    if is_3d:
        compute_3D_spectral_analysis(df)
    else:
        compute_2D_spectral_analysis(df)
# ===============================================================
#                       SHEAR LAYER FUNCTIONS
# ===============================================================

def compute_2D_shear_layer(df):
    """
    Function: compute_2D_shear_layer
    ---------------------------------
    Purpose:
    - Measure how much the flow speed changes across the 2D plane.
    - This is called the "shear layer" and highlights areas where velocity changes rapidly.
    - In 2D, we focus on the x-direction and compute the average speed change along y.

    Steps:
    1. Check required columns exist ('x','y','u','v')
    2. Try to reshape the data into a 2D grid (rows=y, columns=x)
    3. Compute velocity magnitude at each grid point
    4. Average along y-axis to get a single shear value for each x
    5. If reshaping fails (unstructured points), fallback to binning approach
    """

    # Verify the input data has all necessary velocity and position columns
    if not all(col in df.columns for col in ["x", "y", "u", "v"]):
        raise ValueError("Missing columns for 2D shear layer: x, y, u, v.")

    # Extract unique x and y positions and sort them
    # Sorting ensures we have a logical order along each axis
    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())

    try:
        # Reshape velocity components into 2D grid
        # Rows = different y positions, Columns = different x positions
        # This assumes the dataset is structured like a grid (common in CFD)
        U = df.sort_values(["y", "x"])["u"].values.reshape(len(y_unique), len(x_unique))
        V = df.sort_values(["y", "x"])["v"].values.reshape(len(y_unique), len(x_unique))

        # Compute velocity magnitude at each grid point: speed = sqrt(u^2 + v^2)
        vel_mag = np.sqrt(U**2 + V**2)

        # Average along y-axis (rows) → gives a single shear value for each x
        shear_layer = vel_mag.mean(axis=0)
        return x_unique, shear_layer

    except Exception:
        # Fallback method if reshaping fails (e.g., unstructured data)
        # Step 1: Define bins along x
        bins = np.linspace(df.x.min(), df.x.max(), len(x_unique))
        
        # Step 2: Compute magnitude at all points
        mag = np.sqrt(df.u.values**2 + df.v.values**2)
        
        # Step 3: Initialize arrays to sum velocity magnitudes and counts
        shear = np.zeros(len(bins))
        counts = np.zeros(len(bins))
        
        # Step 4: Loop through all points and assign to nearest x bin
        for xi, mi in zip(df.x.values, mag):
            idx = np.argmin(np.abs(bins - xi))  # find closest bin for this x value
            shear[idx] += mi                    # add velocity magnitude to bin
            counts[idx] += 1                    # count how many points are in this bin
        
        # Step 5: Avoid division by zero (if no points in a bin)
        counts[counts == 0] = 1

        # Step 6: Average magnitude in each bin
        shear /= counts

        return bins, shear


def compute_3D_shear_layer(df, k=6):
    """
    Function: compute_3D_shear_layer
    ---------------------------------
    Purpose:
    - Measure how velocity magnitude changes locally in 3D space (local shear)
    - Imagine a 3D airflow: some regions speed up, some slow down.
    - Shear is high where velocity changes rapidly in any direction.
    - For each point, we estimate how fast velocity changes using neighbors.

    Steps:
    1. Compute velocity magnitude at all points: |V| = sqrt(u^2 + v^2 + w^2)
    2. Find the k nearest neighbors of each point (spatial neighborhood)
    3. Fit a plane to the neighborhood velocities (linear approximation)
    4. Compute the slope of the plane → magnitude = shear
    """

    # Step 1: Extract positions and compute velocity magnitudes
    coords = df[['x','y','z']].values                       # positions
    mag = np.sqrt(df['u']**2 + df['v']**2 + df['w']**2)    # speed at each point
    n = len(df)
    k = min(k, n)                                           # cannot have more neighbors than points

    # Step 2: Build nearest neighbors model
    # This finds the indices of k closest points for each point
    nbr = NearestNeighbors(n_neighbors=k).fit(coords)
    ind = nbr.kneighbors(coords, return_distance=False)

    # Step 3: Prepare array to store shear magnitudes
    shear = np.zeros(n)

    # Step 4: Loop over each point and compute local shear
    for i, idx in enumerate(ind):
        # idx = indices of the k neighbors
        # Construct matrix A for plane fitting: [x y z 1] for linear equation
        A = np.c_[coords[idx], np.ones(len(idx))]

        # Solve for plane coefficients: mag = a*x + b*y + c*z + d
        # np.linalg.lstsq → finds best-fit plane that minimizes error
        try:
            c, *_ = np.linalg.lstsq(A, mag[idx], rcond=None)  # c = [a,b,c,d]
            shear[i] = np.linalg.norm(c[:3])                  # slope magnitude = local shear
        except Exception:
            shear[i] = 0.0  # if fitting fails, set shear to 0

    # Step 5: Return positions and computed shear
    return df['x'].values, df['y'].values, df['z'].values, shear


def plot_2D_shear_layer(x, shear_layer):
    """
    Function: plot_2D_shear_layer
    ------------------------------
    Purpose:
    - Visualize 2D shear along x-axis.
    - x-axis = distance along flow
    - y-axis = speed change (shear)
    - Peaks = areas where flow changes fastest → strong shear

    Steps:
    1. Plot shear vs x
    2. Add markers, grid, and labels for clarity
    """

    plt.figure(figsize=(8, 5))
    plt.plot(x, shear_layer, marker='o', linestyle='-', color='purple')  # points + line
    plt.title("2D Shear Layer (Velocity Magnitude)")
    plt.xlabel("x-position")
    plt.ylabel("|V| (Velocity magnitude / shear)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_3D_shear_layer(x, y, z, shear):
    """
    Function: plot_3D_shear_layer
    ------------------------------
    Purpose:
    - Visualize shear in 3D space.
    - Each point = position in 3D space (x, y, z)
    - Color = how strong the local velocity change is
    - Brighter colors = higher shear → regions of rapid velocity change
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points with color = shear magnitude
    sc = ax.scatter(x, y, z, c=shear, cmap='viridis', s=25)  # s=25 → point size
    plt.colorbar(sc, pad=0.1, label='Shear magnitude')

    # Label axes for clarity
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D Shear Layer (local gradient magnitude)')

    plt.tight_layout()
    plt.show()
# ===============================================================
#           VELOCITY FIELD FUNCTION
# ===============================================================

def plot_velocity_field(df):
    """
    Function: plot_velocity_field
    -----------------------------
    Purpose:
    - Visualize the 3D velocity field of the flow using arrows.
    - Each arrow shows the direction and relative speed of flow at a point.
    - Arrow length/color indicates speed magnitude.
    - Helps identify flow patterns, regions of fast/slow flow, or vortices.

    Steps:
    1. Check if necessary velocity columns exist.
    2. Extract positions and velocity components.
    3. Compute magnitude of velocity at each point.
    4. Normalize vectors to plot arrows consistently.
    5. Use Matplotlib's 3D quiver to show arrows in 3D space.
    """

    try:
        # Step 1: Ensure required columns exist
        if not all(col in df.columns for col in ["x", "y", "u", "v"]):
            messagebox.showinfo("Velocity Field", "Missing columns: x, y, u, v.")
            return

        # Step 2: Provide default values for missing z and w components
        # This allows 2D data to be plotted in 3D space at z=0
        df["z"] = df.get("z", 0.0)
        df["w"] = df.get("w", 0.0)

        # Step 3: Extract arrays for positions and velocities
        x, y, z = df["x"].values, df["y"].values, df["z"].values
        u, v, w = df["u"].values, df["v"].values, df["w"].values

        # Step 4: Compute velocity magnitude at each point
        # magnitude = sqrt(u^2 + v^2 + w^2)
        mag = np.sqrt(u**2 + v**2 + w**2)
        max_mag = np.max(mag)

        # Step 5: If all velocities are zero, exit with info
        if max_mag == 0:
            messagebox.showinfo("Velocity Field", "All velocity magnitudes are zero.")
            return

        # Step 6: Determine arrow length for plotting
        # We scale arrows relative to the largest axis range (10%)
        dx = np.max(x) - np.min(x)
        dy = np.max(y) - np.min(y)
        dz = np.max(z) - np.min(z)
        arrow_length = 0.1 * max(dx, dy, dz)

        # Step 7: Normalize velocity vectors for plotting
        # Ensures arrows represent direction, not absolute length
        norm_u = u / max_mag
        norm_v = v / max_mag
        norm_w = w / max_mag

        # Step 8: Create 3D plot figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Step 9: Plot arrows (quiver)
        # Parameters:
        # - x,y,z: start positions of arrows
        # - norm_u,v,w: direction of arrows (normalized)
        # - length: scaling factor for arrow length
        # - color: color of arrows
        ax.quiver(x, y, z, norm_u, norm_v, norm_w, length=arrow_length, color="#2f8cae")

        # Step 10: Label axes and set plot title
        ax.set_title("3D Velocity Field (Quiver Plot)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Step 11: Automatically scale axes
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        zmin, zmax = np.min(z), np.max(z)
        if zmin == zmax:  # if flat 2D data
            zmin -= 0.5
            zmax += 0.5
        ax.set_zlim(zmin, zmax)

        # Step 12: Add grid and tidy layout
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        # Catch any unexpected error and show in messagebox
        messagebox.showinfo("Velocity Field", f"Error: {e}")
# ===============================================================
#                       MAIN FUNCTION
# ===============================================================

def main():
    """
    Function: main
    ---------------
    Purpose:
    - Orchestrates the entire workflow:
        1. Ask user to select a data file (.vtk or .csv)
        2. Load the data into a DataFrame
        3. Detect if data is 2D or 3D
        4. Compute and visualize:
           - Velocity field
           - Vorticity
           - Pressure gradient
           - Shear layer
           - Spectral analysis (FFT and wavelet)
    - Handles errors gracefully using pop-up messages.
    """

    # Step 1: Ask the user to select a file (VTK or CSV)
    file_path = load_file()  # opens a file dialog
    if not file_path:        # if user cancels
        return               # exit program

    # Step 2: Load the data from the selected file
    df = load_data(file_path)       # read into a Pandas DataFrame
    fname = os.path.basename(file_path)  # get file name without path
    print(f"Loaded: {fname}")       # print file name to console

    try:
        # Step 3: Detect if data is 3D
        is_3d = is_3D_data(df)

        # Step 3a: If the VTK file stored a combined 'velocity' array, split it
        if "velocity" in df.columns and df["velocity"].ndim == 2:
            vel = np.vstack(df['velocity'].values)
            df['u'], df['v'], df['w'] = vel[:,0], vel[:,1], vel[:,2]

        # ----------------------------
        #       3D DATA WORKFLOW
        # ----------------------------
        if is_3d:
            # Plot the 3D velocity field
            plot_velocity_field(df)

            # Compute 3D vorticity (local rotation of flow)
            x, z, vx, vz, mag = compute_3D_vorticity(df)
            plot_3D_vorticity(x, z, vx, vz, mag)

            # Compute 3D pressure gradient magnitude
            x_pg, grad_pg = compute_3D_pressure_gradient(df)
            plot_3D_pressure_gradient(x_pg, grad_pg)

            # Compute 3D shear layer
            x_sl, y_sl, z_sl, shear_sl = compute_3D_shear_layer(df)
            plot_3D_shear_layer(x_sl, y_sl, z_sl, shear_sl)

            # Perform spectral analysis (FFT and wavelet)
            perform_spectral_analysis(df)

        # ----------------------------
        #       2D DATA WORKFLOW
        # ----------------------------
        else:
            # Plot the 2D velocity field
            plot_velocity_field(df)

            # Compute 2D vorticity (rotation of flow in 2D plane)
            X, Y, vort, u_vel, v_vel, streamfn = compute_2D_vorticity(df)
            plot_2D_vorticity(X, Y, vort, u_vel, v_vel, streamfn)

            # Compute 2D pressure gradient along x
            x, grad = compute_2D_pressure_gradient(df)
            plot_2D_pressure_gradient(x, grad)

            # Compute 2D shear layer
            x_sl, shear_layer_2d = compute_2D_shear_layer(df)
            plot_2D_shear_layer(x_sl, shear_layer_2d)

            # Perform spectral analysis (FFT and wavelet)
            perform_spectral_analysis(df)

    except Exception as e:
        # If anything goes wrong in the workflow, show an error popup
        messagebox.showerror("Critical Error", str(e))


# ===============================================================
#                       RUN PROGRAM
# ===============================================================

if __name__ == "__main__":
    """
    Python standard way to run script:
    - If this file is executed directly, run main()
    - If imported as a module, main() will not run automatically
    """
    main()
