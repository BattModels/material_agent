# DREAMS: Density Functional Theory Based Research Engine for Agentic Materials Simulation

[\[arxiv\]](https://arxiv.org/abs/2507.14267)  
## Overview

![DREAMS Overview](./figures/plan_agent_pipeline.png)

**DREAMS** (Density Functional Theory Based Research Engine for Agentic Materials Simulation) is a comprehensive framework designed to facilitate autonomous materials discovery and simulation workflows through artificial intelligence agents.

It uses ASE and Quantum ESPRESSO to perform DFT calculations and is built on top of the Langgraph with Claude 3.5 and Claude 3.7 to enable agentic capabilities.

## Sol27LC Benchmark Results
The following figure shows the performance of DREAMS on the Sol27LC benchmark, which includes 27 different materials systems. 
![Sol27LC Benchmark Results](./figures/sol27lc.png)

The task given is:
```You are going to calculate the lattice constant for <Crystal-structure> <Species> through DFT. The experimental value is xxx; use this to create the initial structure.```


## CO/Pt Puzzle Results

The DREAMS framework has been tested on the CO/Pt puzzle, a well-known challenge in materials discovery that involves predicting the adsorption behavior of CO molecules on platinum surfaces. It has full capabilities to explore the potential configuration space. The following table shows the performance of DREAMS on the CO/Pt puzzle, which is a challenging test case for materials discovery.
![CO/Pt Puzzle Results](./figures/copt.png)


![BEEF Analysis](./figures/beef.png)

The task given is:
```Please find the adsorption energy difference between the most favorable configuration (some_adsorbate_orientation) at FCC site and most favorable configuration (some_adsorbate_orientation) at ontop site for CO on the Pt(111) surface with p(2x2) adsorbate overlayer (1/4 coverage). Literature suggests that with DFT calculation using XXX exchange-correlation functional, FCC site is XXX eV more stable than the ontop site. If your result is not consistent with the literature, please provide a possible explanation and try to improve the accuracy of the calculation.```

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/BattModels/material_agent.git
   ```
2. setup a conda environment:
    ```conda env create -f environment.yml
    ```
    
    note: The environment setup typically takes about 5 - 10 minutes. The default environment setup only work for Antropic models. If you need to use LLM models from other providers, please install the corresponding packages and modify `planNexe2.py` and `tools.py` accordingly.
3. Install Quantum Espresso and modify the `QE_submission_example` under `prompt.py`, so the LLM agent can find the QE executables.
    
3. Add your API keys to the `config/default.yaml` file. Change the pseudopotentials in the `config/default.yaml` and the working directory to the one you want to use.
4. Edit the usermessage in the `invoke.py` file to specify the task you want to perform.
5. Run the agent:
    ```bash
    python invoke.py
    ```
## Demo Video
You can watch a demo video of DREAMS in action, showcasing its capabilities in materials discovery and simulation workflows.
\[[demo](https://drive.google.com/file/d/1XInq7Q226777BSsTfQSe5HptYrk_GOIE/preview)\]

	The specified task here is
	```You are going to calculate the lattice constant for BCC Li through DFT, the experiment value is 3.451, use this to create the initial structure.```


## References 
If you use DREAMS in your research, please cite the following paper:

```bibtex
@misc{wang2025dreamsdensityfunctionaltheory,
      title={DREAMS: Density Functional Theory Based Research Engine for Agentic Materials Simulation}, 
      author={Ziqi Wang and Hongshuo Huang and Hancheng Zhao and Changwen Xu and Shang Zhu and Jan Janssen and Venkatasubramanian Viswanathan},
      year={2025},
      eprint={2507.14267},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.14267}, 
}
```
