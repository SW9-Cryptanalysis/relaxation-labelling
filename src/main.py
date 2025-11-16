from classes import (
	Cipher,
	StatisticalProfile,
	RelaxationSolver,
	SolverAnalytics,
	SolverConfig,
)
from utils.data import load_ciphers_list, save_results
from utils.logging import get_colored_logger
from tqdm import tqdm
from argparse import ArgumentParser

log = get_colored_logger("Relaxation Solver")


def solve_cipher(cipher_name: str, config: SolverConfig = SolverConfig()):
	cipher = Cipher(cipher_name)
	cipher.load_data()

	cipher_profile = StatisticalProfile.from_cipher(cipher, save=True)
	english_profile = StatisticalProfile.from_english_corpus(save=True)

	solver = RelaxationSolver(english_profile, cipher_profile, cipher, config)
	analytics = SolverAnalytics(solver, cipher)

	solver.run()
	solver.decode()
	if not solver.decoded:
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

	log.info("Solving ciphers...")
	for cipher in tqdm(load_ciphers_list()):
		log.debug(f"Solving cipher: {cipher}")
		analytics = solve_cipher(cipher)
		results.append(analytics)
	if not args.no_cache:
		log.info("Saving results as json...")
		save_results(results)

	log.info("Done!")
