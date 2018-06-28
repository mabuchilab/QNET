"""Set up the environment for tests

This file is automatically evaluated by py.test.
"""
import sys

# We want to detect unexpected recursions, so the recursion limit will be
# artificially low.
sys.setrecursionlimit(300)
