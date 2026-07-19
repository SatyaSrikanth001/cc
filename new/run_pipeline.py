"""
Master orchestrator that runs all phases in sequence.
Ensures clean state execution and absolute process traceability.
"""

import logging
import os
import shutil
import subprocess
import sys
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_phase(script_name: str):
    """Run a phase script as a subprocess, propagating errors and logs."""
    logging.info(f"{'='*70}")
    logging.info(f"LAUNCHING PHASE: {script_name}")
    logging.info(f"{'='*70}")
    
    # sys.executable ensures the subprocess uses the exact same virtual environment
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False
    )
    if result.returncode != 0:
        logging.error(f"Execution failed: {script_name} returned exit code {result.returncode}")
        raise RuntimeError(f"Pipeline execution aborted. {script_name} failed.")
    logging.info(f"SUCCESS: {script_name} completed cleanly.\n")


def main():
    # 1. Verification of Workspace Configuration
    if not os.path.exists('config.yaml'):
        raise FileNotFoundError("Critical Error: config.yaml not found in active working directory.")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    output_dir = config['output'].get('dir', 'out')
    
    # Traceability Metadata
    logging.info("Initializing Master Pipeline Orchestrator...")
    logging.info(f"Python Interpreter: {sys.executable}")
    logging.info(f"Working Directory: {os.getcwd()}")
    logging.info(f"Target Output Directory: {output_dir}")

    # 2. Experimental Sanitization (Cache Busting)
    # Automatically clear target output folder to guarantee no stale files contaminate downstream phases.
    if os.path.exists(output_dir):
        logging.info(f"Sanitizing workspace: Clearing previous execution results in '{output_dir}'...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Explicit Sequence Definition
    # Phase 3 is executed as an alternate feature selection path. 
    # Phase 5 evaluates Phase 4 by default (final_selected_features).
    phases = [
        'phase2a_noise_purge.py',
        'phase2b_elite_rank.py',
        'phase3_mrmr.py',        # Alternative Greedy Selection Path (Outputs preserved in output dir)
        'phase4_consensus.py',   # Primary Weighted Consensus Path
        'phase5_ocsvm_validation.py',  # Leakage-Free Validation of Phase 4 (Consensus)
    ]

    # 4. Strict Validation of Pipeline Integrity
    for script in phases:
        if not os.path.exists(script):
            # We crash immediately if any script is missing rather than running on stale data
            logging.error(f"Critical Pipeline Interruption: '{script}' is missing from workspace!")
            sys.exit(1)

    # 5. Sequential Execution Run
    try:
        for phase in phases:
            run_phase(phase)
        logging.info("=" * 70)
        logging.info("PIPELINE EXECUTION COMPLETE: All phases completed without error.")
        logging.info("=" * 70)
    except Exception as exc:
        logging.error(f"Pipeline aborted prematurely due to an unhandled runtime error:\n{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()