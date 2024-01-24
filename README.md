# Localised Regression with Topological Data Analysis Ball Mapper


## Setting local enviroment
It is assumed that one work under Linux distribution. It should work also under MacOS. In case of Windows a have no idea how to proceed.

1. Clone repository  
`git clone "git@github.com:RafalTopolnicki/localised-regression.git"`  
and enter inside newly created directory  
`cd localised-regression/`
2. Create local python enviroment (this requires python 3.x)  
`python3 -m venv venv`
and activate it  
`source venv/bin/activate`  
notice, that promt starts with `(venv)` now. You must activate the local python enviroment each time you work with teh code.   
**Activation command must be done in each terminal window that is used to work with the project!**
3. Install dependences. This is done only once or after each change in the `requirements.txt`  
`pip install -r requirements.txt`
4. We compare the BLMR method withn the [MARS](https://towardsdatascience.com/mars-multivariate-adaptive-regression-splines-how-to-improve-on-linear-regression-e1e7a63c5eae), among others. The MARS implementation in python is quite difficualt to install. Here is what worked for me:  
`sudo apt-get install python3-dev`  
`git clone https://github.com/scikit-learn-contrib/py-earth.git`  
`cd py-earth`  
`python setup.py install --cythonize`

4. Run notebook  
`jupyter lab`

