# HOPS README
This repository provides a Python implementation of the Hierarchy of Pure States (HOPS) method [1], including both the linear and non-linear formulations. HOPS is a numerically exact approach for solving the time evolution of open quantum systems coupled to a non-Markovian environment composed of harmonic oscillators.

## Dependencies
The implementation requires the following Python libraries:  
- [SciPy](https://scipy.org/)  
- [NumPy](https://numpy.org/)  
- [matplotlib](https://matplotlib.org/)  
- [more-itertools](https://pypi.org/project/more-itertools/)  
- [rich](https://pypi.org/project/rich/)  
- [psutil](https://pypi.org/project/psutil/)  

The easiest way to install these dependencies is with `pip`:

```shell
python3 -m pip install scipy numpy matplotlib more-itertools rich psutil
```

The code has been tested with Python versions 3.10, 3.11, and 3.12. Compatibility with other versions is not guaranteed.

## Try it out
To try it out, clone the repository and run the ```spin_boson_model_example.py``` script.
```shell
git clone https://github.com/Turku-Quantum-Optics/hops.git
cd hops/HOPS/
python3 spin_boson_model_example.py
```

## References
[1] D. Suess, A. Eisfeld, and W. T. Strunz, "Hierarchy of Stochastic Pure States for Open Quantum System Dynamics," Phys. Rev. Lett. (2014).
