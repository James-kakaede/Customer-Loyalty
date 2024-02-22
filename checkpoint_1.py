import pandas as pd
import numpy as np
import streamlit as st
data = pd.read_csv('expresso_processed.csv')
data.head()
print(data.head())