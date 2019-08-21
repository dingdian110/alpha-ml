# Installation Tutorial.
## 1. Install SWIG
* swig: 3.0.5 [download site](https://sourceforge.net/projects/swig/files/swig/swig-3.0.5/)
```
sudo apt-get install build-essential libpcre3-dev
tar -zxf swig-3.0.5.tar.gz
cd swig-3.0.5
./configure && make
```

## 2. Install Python Packages.
Please check the file `exp_requirements.txt`
1. `pip install -r exp_requirements.txt`
2. `pip install george`
3. `cd $lite-smac$` and run `python setup.py install`
4. `pip install lockfile smac==0.8.0 xgboost lightgbm hyperopt dill`
5. `pip uninstall pyrfr` and then `pip install pyrfr`
