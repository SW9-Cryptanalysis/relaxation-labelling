import time
from classes.statistical_profile import StatisticalProfile
from classes.ngram_scorer import NGramScorer
from classes.mcmc_solver import McmcSolver
from classes.cipher import Cipher
from classes.relaxation_solver import RelaxationSolver
from classes.solver_analytics import SolverAnalytics
from utils.logging import get_colored_logger
from utils.constants import RESULT_PATH_MCMC, RANDOM_RESTARTS


log = get_colored_logger("Relaxation Solver")

MCMC_ITERATIONS = 50000
CIPHER_NAMES_TO_TEST = [
	"c_6000_30",  # The "good" one
	# "c_10000_25", # The "bad" one
	# ... add all other cipher names here
]


def run_analysis(
	cipher_name: str,
	iters: int = 50000,
	relaxation: bool = False,
) -> SolverAnalytics | None:
	"""Run the full analysis pipeline for a single cipher."""
	log.debug(f"\n--- Starting analysis for: {cipher_name} ---")
	log.debug(f"Loading cipher data for {cipher_name}...")
	try:
		cipher = Cipher(cipher_name)
		cipher.load_data()
		if not cipher.ciphertext or not cipher.key:
			log.error(f"Error: Cipher class did not load data for {cipher_name}.")
			return None
	except Exception as e:
		log.error(f"Failed to load Cipher object: {e}")
		return None

	log.debug("Loading/Building Statistical Profiles...")
	eng_profile = StatisticalProfile.from_english_corpus(no_cache=False)
	cip_profile = StatisticalProfile.from_cipher(cipher, save=True, no_cache=False)

	log.debug("Initializing n-gram scorer...")
	try:
		scorer = NGramScorer(4, "english_quadgrams.txt")
	except FileNotFoundError as e:
		log.error(f"Fatal Error: {e}")
		log.warning(
			"Please download 'english_quadgrams.txt' and place it in the "
			"ngram directory.",
		)
		return None

	if relaxation:
		log.debug("Running Relaxation Solver...")
		rel_solver = RelaxationSolver(eng_profile, cip_profile, cipher)
		solver = McmcSolver(eng_profile, cip_profile, cipher, scorer)
		start_time = time.time()
		rel_solver.run()
		rel_key = rel_solver.guesses
		log.debug("Running MCMC Solver...")
		solver.run(max_iters=iters, initial_key=rel_key)
	else:
		log.debug("Running MCMC Solver...")
		solver = McmcSolver(eng_profile, cip_profile, cipher, scorer)
		start_time = time.time()
		solver.run(max_iters=iters)

	end_time = time.time()
	solver.time = end_time - start_time

	log.info(f"--- Run finished in {solver.time:.2f} seconds ---")

	analytics = SolverAnalytics(cipher=cipher, solver=solver)
	return analytics


def run_restarts(
	cipher_name: str,
	iters: int = 50000,
	relaxation: bool = False,
	restarts: int = 1,
) -> SolverAnalytics:
	"""Run the full analysis pipeline for a single cipher."""
	best_analytics: SolverAnalytics | None = None
	for i in range(RANDOM_RESTARTS):
		log.info(
			f"Starting analysis for {cipher_name} (Restart {i + 1}/"
			f"{RANDOM_RESTARTS})...",
		)
		analytics: SolverAnalytics | None = run_analysis(
			cipher_name,
			iters,
			relaxation,
		)
		if not analytics or not isinstance(analytics.solver, McmcSolver):
			log.error(
				f"Failed to run analysis for {cipher_name} (Restart {i + 1}"
				f"/{RANDOM_RESTARTS}).",
			)
			continue
		if (
			best_analytics is None
			or analytics.solver.best_score > best_analytics.solver.best_score  # type: ignore
		):  # type: ignore
			best_analytics = analytics
	if not best_analytics:
		raise ValueError("No results found.")
	return best_analytics


def main() -> None:
	"""Start the main execution script.

	Define the list of ciphers you want to test in CIPHER_NAMES_TO_TEST.

	"""
	results: list[SolverAnalytics] = []
	for name in CIPHER_NAMES_TO_TEST:
		best_analytics: SolverAnalytics = run_restarts(name, MCMC_ITERATIONS)
		results.append(best_analytics)

	if results:
		results.sort(key=lambda r: r.cipher.name)
		for r in results:
			log.info(
				f"Result for {r.cipher.name}: {r.solver.decoded}"
				f"(SER: {r.ser:.4f}, MER: {r.mer:.4f})",
			)
			r.save(RESULT_PATH_MCMC + f"{r.cipher.name}.json")
	else:
		log.error("No results to save.")


if __name__ == "__main__":
	main()
