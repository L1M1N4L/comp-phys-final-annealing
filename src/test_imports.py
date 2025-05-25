try:
    import numpy
    print("NumPy is installed")
except ImportError:
    print("NumPy is NOT installed")

try:
    import matplotlib
    print("Matplotlib is installed")
except ImportError:
    print("Matplotlib is NOT installed")

try:
    import imageio
    print("ImageIO is installed")
except ImportError:
    print("ImageIO is NOT installed") 