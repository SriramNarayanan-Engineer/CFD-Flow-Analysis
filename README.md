🌀 CFD Flow Analysis Toolkit

This project is a Python-based toolkit for analyzing and visualizing fluid flow data.  
It works with data saved from simulations (CFD) or experiments in '.vtk' or '.csv' format.  

---

✨ What Can This Tool Do?

  📂 Load Data  
  - Accepts '.vtk' files  
  - Accepts '.csv' files (spreadsheets with columns like 'x, y, z, u, v, w, p')  

  🔍 Flow Analysis  
  - Compute Vorticity → how much the fluid is swirling (like whirlpools)  
  - Compute Shear Layers → regions where fluid is sliding past itself  
  - Compute Pressure Gradients → where pressure increases/decreases  
  - Compute Streamfunction → flow lines that show how the fluid moves  

  📊 Frequency & Wave Analysis  
  - FFT (Fast Fourier Transform) → shows which frequencies dominate in the flow  
  - Wavelet Transform → shows how frequencies change over time/space  

  🎨 Visualizations  
  - 2D contour plots (like heat maps of vorticity, shear, etc.)  
  - 3D vector/quiver plots (arrows showing flow direction)  
  - Color maps to highlight important flow regions  

---

## 🛠️ Installation Guide (Step by Step)

1. Install Python 3.9+  
   - If you don’t have it, download from [python.org](https://www.python.org/downloads/)  

2. Download this project  
   - Option 1: Clone using Git  
     ```bash
     git clone https://github.com/SriramNarayanan-Engineer/CFD-Flow-Analysis.git
     cd CFD-Flow-Analysis
     ```  
   - Option 2: Download ZIP from GitHub → extract it → open in your terminal  

3. Install required Python packages  
   ```bash
   pip install -r requirements.txt

4. (Linux only) Install Tkinter GUI support
    sudo apt-get install python3-tk

---

▶️ How to Run

Open a terminal in the project folder.

Run the program:
python main.py

A file dialog will appear → select your .vtk or .csv file.

The program will process the data and display results as plots and analysis.

📂 Example Input Data

.vtk file (CFD mesh): contains 3D points and velocity vectors.

.csv file (spreadsheet): looks like this:

x	y	z	u	v	w	p
0.0	0.1	0.0	2.5	0.3	0.0	101325
0.1	0.1	0.0	2.7	0.4	0.0	101300

Columns:

x, y, z → position

u, v, w → velocity components

p → pressure
