from classes import (
    Cipher,
    StatisticalProfile,
    RelaxationSolver,
    SolverAnalytics,
    SolverConfig,
)
from utils.data import load_ciphers_list
from utils.logging import get_colored_logger
from tqdm import tqdm
from argparse import ArgumentParser
import os
from datetime import datetime
import pandas as pd


log = get_colored_logger("Relaxation Solver")


def solve_cipher(
    cipher_name: str,
    config: SolverConfig | None = None,
    restarts: int = 1,
    p_bar: tqdm | None = None,
) -> list[SolverAnalytics]:
    """Solve a single cipher.

    Args:
        cipher_name (str): Name of the cipher to solve
        config (SolverConfig, optional): Solver configuration. Defaults to None.
        restarts (int, optional): Number of restarts to run. Defaults to 1.
        p_bar (tqdm, optional): Global progress bar. Defaults to None.

    Returns:
        SolverAnalytics: Solver analytics object

    """
    config = SolverConfig() if config is None else config
    cipher = Cipher(cipher_name)
    cipher.load_data()

    cipher_profile = StatisticalProfile.from_cipher(cipher, save=True)
    english_profile = StatisticalProfile.from_english_corpus(save=True)

    analytics = []

    for _ in range(restarts):
        solver = RelaxationSolver(english_profile, cipher_profile, cipher, config)
        analytics.append(SolverAnalytics(solver))

        solver.run()
        solver.decode()

        if p_bar:
            p_bar.update(1)

    if not analytics:
        log.error("Could not decode ciphertext. Check for symbol mismatches.")
        exit(1)

    return analytics


if __name__ == "__main__":
    results = []

    parser = ArgumentParser()

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disables use of cached results and overwrites them. (default: False)",
    )

    args = parser.parse_args()

    data_records = []

    RESTARTS = 10
    ciphers = load_ciphers_list()

    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    output_csv = f"results/relaxation_experiment_{timestamp}.csv"

    log.info(f"Solving {len(ciphers)} ciphers with 10 restarts each...")

    # Global progress bar
    p_bar = tqdm(total=len(ciphers) * RESTARTS)

    for cipher_name in ciphers:
        # Run the batch
        batch_analytics = solve_cipher(cipher_name, restarts=RESTARTS, p_bar=p_bar)

        # Parse Name for Plotting Dimensions (e.g., "c_400_5")
        try:
            parts = cipher_name.split("_")
            # Handle standard format: c_400_5
            if len(parts) == 3 and parts[0] == "c":
                length = int(parts[1])
                difficulty = int(parts[2])
            # Handle z408 or mono
            elif "z408" in cipher_name:
                length = 408
                difficulty = 0  # Special flag
            elif "mono" in cipher_name:
                length = 4000
                difficulty = 0  # Special flag
            else:
                length = 0
                difficulty = 0
        except Exception:
            length = 0
            difficulty = 0

        # Process each restart
        for run_id, run in enumerate(batch_analytics):
            record = {
                "cipher_name": cipher_name,
                "run_id": run_id,
                "length": length,
                "difficulty": difficulty,
                # Metrics (Ensure SolverAnalytics exposes these)
                "time": run.solver.time,
                "ser": getattr(run, "ser", 1.0),  # Default to 1.0 if missing
                "mer": getattr(run, "mer", 1.0),
                "score": getattr(run, "score", 0.0),
            }
            data_records.append(record)

    p_bar.close()

    # 2. Save to CSV
    df = pd.DataFrame(data_records)
    df.to_csv(output_csv, index=False)

    log.info(f"Results saved to {output_csv}")

    # --- Quick Preview for your sanity ---
    # Group by cipher and find the BEST SER for each
    best_results_ser = df.loc[df.groupby("cipher_name")["ser"].idxmin()]
    best_results_score = df.loc[df.groupby("cipher_name")["score"].idxmax()]
    log.info("\nBest results per cipher:")
    log.info(best_results_ser[["cipher_name", "ser", "time"]].to_string())

    log.info("\nBest results per cipher (score):")
    log.info(best_results_score[["cipher_name", "score", "time"]].to_string())
