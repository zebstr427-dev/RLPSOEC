# RLPSOEC UAV Emergency Communication Simulation

This project contains the companion simulation and result-organization code for the RLPSOEC method. It supports UAV emergency communication networking and real-time mapping tasks, including relay deployment, link-capacity evaluation, on-demand network adaptation, and ablation analysis. Communication capability is represented by measurement-driven effective capacity, and relay-assisted links use two-hop bottleneck capacity.

## Project Structure

`models/`  
Link-capacity models. This part follows the communication-model description in the paper: measured throughput is treated as effective capacity, a LOS capacity curve is built from node distance, and DSM occlusion information is used to correct NLoS links. Equivalent SNR is used only as a monitoring and state variable.

`config/`  
Experiment configuration and reference metrics, including RLPSOEC, RLPSOEC w/o PPO, RLPSOEC w/o Trigger, and Centralized deployment.

`simulation/`  
Core simulation modules. This part builds task scenarios, DSM environments, mapping UAV trajectories, base-station position, relay-state updates, on-demand triggering, PSO search, PPO/strategy-assisted parameter adjustment, and experiment logging.

`metrics/`  
Result statistics and consistency checks. This part normalizes column names across log files and computes average total capacity, trigger rate, relay jump, optimization time, and success rate.

`experiments/`  
Experiment organization layer. It separates the full RLPSOEC method, the ablation version without PPO, the ablation version without the trigger mechanism, and the centralized/static deployment baseline. This directory keeps only high-level experiment organization.

`visualization/`  
Paper-figure generation and label normalization. It organizes relay jump, instantaneous total capacity, and average-performance comparison outputs.

`speed_modeling_data/`  
Communication-modeling data area. CSV files in the root directory are original measurement inputs and are not overwritten by simulation outputs. 

`test-noPPO/`, `test-noTrigger/`, `test-static/`  
Legacy-compatible directories for ablation experiments and baseline experiments.

`tests/`  
Tests and regression checks. They cover effective-capacity variation with distance, DSM occlusion penalty, equivalent SNR derivation, trigger thresholds, and PSO two-hop bottleneck capacity.

The project uses the method names from the paper:

- `RLPSOEC`
- `RLPSOEC w/o PPO`
- `RLPSOEC w/o Trigger`
- `Centralized deployment`

## Outputs

Project outputs mainly include simulation logs, experiment summary JSON files, performance-evolution figures, paper-figure input CSV files, and related images.
