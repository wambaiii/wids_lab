@echo off
REM Runs the whole lab on Windows. Open PowerShell/CMD in this folder first.
REM   venv\Scripts\activate    (after: python -m venv venv  and  pip install -r requirements.txt)
python make_synthetic.py
python features.py data\train_frames.csv
python train.py
python detect.py
pause
