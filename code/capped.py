import os, sys, runpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard
guard.arm()
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name="__main__")
