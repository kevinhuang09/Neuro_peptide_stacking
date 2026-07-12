# set Agg to backend and without popping up a window
import matplotlib
matplotlib.use("Agg")

# set path 
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)