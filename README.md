# Federal Recommendation

This project implements a federated recommendation experiment pipeline with privacy protection, attack evaluation, batch execution, and report generation.

## Highlights

- Federated optimization with `FedAvg` and `FedProx`
- Shared backbone plus local personalized head
- Privacy modes: `PLAIN`, `CDP`, `LDP`
- Adaptive differential privacy scheduling
- Membership inference attack evaluation with `ASR`, `AUC`, `Precision`, `Recall`, and `F1`
- Batch cloud experiments and automatic result reporting

## Project Structure

- `src/`: core training, model, privacy, dataset, and attack code
- `main.py`: interactive training entrypoint
- `run_cloud_batch.py`: batch runner for cloud experiments
- `generate_final_report.py`: generate final figures and report
- `reports/`: generated experiment reports
- `data/README.txt`: dataset note

## Main Experiment Groups

- `G0-G2`: plain federated recommendation baselines
- `G3-G6`: centralized DP branches
- `G7-G9`: local DP branches
- `A1-A4`: ablation and sensitivity experiments

## Reproduce

1. Prepare the dataset files under `data/`.
2. Install dependencies from `requirements_cloud.txt`.
3. Run a single interactive experiment with `python main.py`.
4. Run batch experiments with `python run_cloud_batch.py --rounds 1000 --seeds 42,52`.
5. Generate the final report with `python generate_final_report.py`.

## Notes

- Large generated artifacts, pulled cloud results, private keys, and raw dataset CSV files are excluded through `.gitignore`.
- The repository is intended to publish source code and documentation only, not private credentials or bulky experiment outputs.
