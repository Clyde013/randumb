This package contains c++ and cuda bindings to python, and the entire package can be installed by running the `setup.py` file via `pip install .` in the base directory.

# cmake commands cheatsheet
compile `./main` cd csrc/build directory and run 
```
cmake --build . --config Release
```

if it is necessary to redo the build folder:
```
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
```