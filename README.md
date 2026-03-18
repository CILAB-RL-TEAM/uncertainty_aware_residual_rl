# Accelerating Residual Reinforcement Learning using Uncertianty Estimation

[**Paper**](https://ieeexplore.ieee.org/document/11267054) | [**Website**](https://lakshitadodeja.github.io/uncertainty-aware-residual-rl/)

[Lakshita Dodeja](https://lakshitadodeja.github.io/website/), [Karl Schmeckpeper](https://sites.google.com/view/karlschmeckpeper), [Shivam Vats](https://shivamvats.com/), [Thomas Weng](https://thomasweng.com/), [Mingxi Jia](https://mingxi-jia.github.io/), [George Konidaris](https://cs.brown.edu/people/gdk/), [Stefanie Tellex](https://h2r.cs.brown.edu/people/)

<img src="docs/images/video_comparison_hd.gif" alt="drawing" width="50%"/>

## Installation
1. **Setup Conda Environment:**
    ```bash
    conda create -n uarl python=3.10 -y
    conda activate uarl
    ```

2. **Install DPPO:**

    We use the diffusion policies from [Diffusion Policy Policy Optimization (DPPO)](https://github.com/irom-princeton/dppo) to keep the comparison consistent. Clone and install DPPO outside of the current repo : 

    ``` bash
    git clone https://github.com/irom-princeton/dppo.git
    cd ~/dppo
    pip install -e .
    ```
    Environment specific dependencies:
    ```bash
    pip install -e .[robomimic]  
    ```

    Add dppo to PYTHONPATH
    ```
    export PYTHONPATH=$HOME/dppo:$PYTHONPATH
    ```
3. **Install Dependencies:**
    ```
    cd ~/uncertainty_aware_residual_rl
    pip install -r requirements.txt
    ```
3. **Install the serl_launcher**
    ```bash
    cd serl_launcher
    pip install -e .
    ```
    This codebase is developed from [SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning](https://github.com/rail-berkeley/serl). 
    We change the sac agent of SERL for our method in the file "assymetric_sac".
    
    
## Base Policies

Base diffusion policies and training data for can and square tasks are available [here](https://drive.google.com/drive/folders/1lQXcmddQQ4ZVAbDoNdeZMdXJ-WFvQL-Q?usp=sharing). 

Downlod the policies and move them in `base_policies` folder in the repo.   

## Running Experiments

Example scripts for running the repo with both distance-to-data and ensemble variance uncertainty metric are in folder `examples/residual_sim/scripts`

Replace the `test_checkpoints` folder in the scripts with your custom folder for storing checkpoints and logs. 

## Citation
If you find this repo useful, please cite our work using : 
```bibtex
    @ARTICLE{11267054,
    author={Dodeja, Lakshita and Schmeckpeper, Karl and Vats, Shivam and Weng, Thomas and Jia, Mingxi and Konidaris, George and Tellex, Stefanie},
    journal={IEEE Robotics and Automation Letters}, 
    title={Accelerating Residual Reinforcement Learning With Uncertainty Estimation}, 
    year={2026},
    volume={11},
    number={1},
    pages={970-977},
    keywords={Uncertainty;Stochastic processes;Reinforcement learning;Imitation learning;Training;Robustness;Tuning;Transforms;Training data;Robot control;Reinforcement learning (RL);deep learning methods;machine learning for robot control},
    doi={10.1109/LRA.2025.3636808}}
```





