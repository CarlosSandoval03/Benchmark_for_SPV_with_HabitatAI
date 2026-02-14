# Habitat-Lab Modifications for End-to-End Prosthetic Vision (SPV)

This document summarizes the modifications made to the upstream Habitat-Lab repository to enable **end-to-end (E2E) training and evaluation under Simulated Prosthetic Vision (SPV)**.

The `habitat-lab/` directory contains a version of Habitat-Lab (v2.4) that was extended to:

- Support SPV-based observation pipelines
- Train PPO agents directly on phosphene representations
- Integrate reconstruction-based auxiliary losses
- Enable end-to-end evaluation of prosthetic vision navigation performance based on Ruyter van Steveninck et al. (2022) and Küçükoglu et al. (2022)

Upstream project:
Habitat-Lab (Meta)

---

# Overview of Modifications

The goal of these modifications was to enable full end-to-end training of navigation agents under prosthetic vision.

Core extensions include:

1. Custom observation transforms for SPV
2. PPO training loop modifications to incorporate reconstruction loss
3. Reward shaping adjustments
4. Policy modifications to operate on SPV representations
5. Distributed training compatibility

---

# Modified Files:

The following files were intervened to support SPV-based training and evaluation:

### Core PPO & Training Pipeline
- habitat-baselines/habitat_baselines/agents/ppo_agents.py
- habitat-baselines/habitat_baselines/rl/ppo/policy.py
- habitat-baselines/habitat_baselines/rl/ppo/ppo.py
- habitat-baselines/habitat_baselines/rl/ppo/ppo_trainer.py
- habitat-baselines/habitat_baselines/rl/ppo/single_agent_access_mgr.py

### Policy & Representation
- habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py
- habitat-baselines/habitat_baselines/rl/ddppo/policy/running_mean_and_var.py

### Distributed & Utilities
- habitat-baselines/habitat_baselines/rl/ddppo/ddp_utils.py
- habitat-baselines/habitat_baselines/rl/ver/inference_worker.py
- habitat-baselines/habitat_baselines/utils/timing.py

### Observation & Environment Integration
- habitat-baselines/habitat_baselines/common/construct_vector_env.py
- habitat-baselines/habitat_baselines/common/obs_transformers.py
- habitat-baselines/habitat_baselines/common/rollout_storage.py

### Configuration
- habitat-baselines/habitat_baselines/config/default_structured_configs.py
- habitat-baselines/habitat_baselines/run.py

### Visualization
- habitat-lab/habitat/utils/visualizations/utils.py

---

## Nature of Changes (High-Level)

Across the above modules, changes were made to:

- Introduce SPV-based observation transforms
- Adjust policy input handling for phosphene representations
- Integrate reconstruction loss into the PPO training loop
- Enable optional reward shaping based on reconstruction signals
- Ensure compatibility of custom-made transformations with Habitat-AI

These modifications were implemented as part of a time-constrained research workflow and were validated within the dependency stack available at that time.

---

## Reproducibility Note

Habitat-Lab evolves rapidly.  
This modified version corresponds to the ecosystem available during development and may require manual adaptation when used with newer upstream versions.

For a conceptual overview of the SPV pipeline and results, see the top-level README.

---

# References
- de Ruyter van Steveninck, J., Güçlü, U., vanWezel, R., & van Gerven,M. (2022). End-to-end optimization of prosthetic vision. Journal of Vision, 22(2), 20. https://doi.org/10.1167/jov.22.2.20
- Küçükoglu, B., Rueckauer, B., Ahmad, N., de Ruyter van Steveninck, J., Güçlü, U., & van Gerven,M. (2022). Optimization of neuroprosthetic vision via end-to-end deep reinforcement learning. International Journal of Neural Systems. https://doi.org/10.1142/s0129065722500526
