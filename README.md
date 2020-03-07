# ethz-safe-learning
## SIMBA: Safe Informative Model Based Agent

### Installing mujoco_py
Make sure that you're in a conda python 3.6 environment
``` 
git clone https://github.com/openai/mujoco-py
cd mujoco-py
pip install -e . --no-cache
python3
import mujoco_py
mujoco_py.__version__
>> '2.0.2.9'
```
If it says something about patchelf:
```
conda install patchelf
```
