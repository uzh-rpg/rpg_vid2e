import os
try:
    import esim_py
    print("Installation sucessful!")
except ImportError:
    print("esim_py not found, importing binaries. These do not correspond to source files in this repo")
    import sys
    binaries_folder = os.path.join(os.path.dirname(__file__), "..", "bin")
    sys.path.append(binaries_folder)
    import esim_py
    print("Import of binaries successful!")




