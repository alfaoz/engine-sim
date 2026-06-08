"""Engine Simulator — entry point.

Physically-based real-time engine model:
  * slider-crank kinematics + 0D filling/emptying cylinder thermodynamics
  * Wiebe combustion heat release
  * 1D compressible (Euler) exhaust gas dynamics — the tailpipe pressure wave
    is sent straight to the speaker as the engine note

Run:  python main.py
"""
from engine_sim.ui import main

if __name__ == "__main__":
    main()
