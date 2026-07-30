# Dependencies

Install PyTorch separately for the actual CPU/CUDA environment by following
the official PyTorch installation guidance. The cloud training environment may
continue to use its existing compatible PyTorch installation.

This project deliberately does not pin a PyTorch wheel because the correct
wheel depends on the machine's CUDA runtime. Do not use this project setup to
install or upgrade PyTorch implicitly.

After PyTorch is available, install the small runtime and test dependency sets
as appropriate:

```bash
python -m pip install -r requirements/runtime.txt
python -m pip install -r requirements/dev.txt
```
