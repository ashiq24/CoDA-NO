#!/usr/bin/env python3

import pandas as pd
import sys
import h5py
h5f= h5py.File('mesh.h5','r')
print(h5f.keys())
print(h5f['mesh'])
with pd.HDFStore('mesh.h5', 'r') as d:
    df = d.get('mesh')
    df.to_csv('mesh.csv')
