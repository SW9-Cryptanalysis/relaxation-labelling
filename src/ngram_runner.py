from classes.mcmc_solver import McmcSolver
from classes.relaxation_solver import RelaxationSolver
from classes.cipher import Cipher
from classes.solver_analytics import SolverAnalytics
from classes.ngram_scorer import NGramScorer
from classes.statistical_profile import StatisticalProfile
import time
from utils.logging import get_colored_logger
from utils.data import load_ciphers_list
from latex import save_results_as_latex
from utils.constants import RESULT_PATH_MCMC
import os
import json


log = get_colored_logger("Relaxation Solver")


def run_analysis(
	cipher_name: str,
	relaxation: bool = False,
	iters: int = 50000,
) -> SolverAnalytics | None:
	"""Run the full analysis pipeline for a single cipher.

	Args:
		cipher_name (str): Name of the cipher to analyze
		relaxation (bool, optional): Whether to run the relaxation
			solver or not. Defaults to False.
		iters (int, optional): Number of iterations to run. Defaults to 50000.

	Returns:
		SolverAnalytics | None: Solver analytics object or None if an error occurred

	"""
	log.info(f"\n=== Starting analysis for: {cipher_name} ===")
	log.info(f"Loading cipher data for {cipher_name}...")
	try:
		cipher = Cipher(cipher_name)
		cipher.load_data()
		if not cipher.ciphertext or not cipher.key:
			log.error(f"Error: Cipher class did not load data for {cipher_name}.")
			return None
	except Exception as e:
		log.error(f"Failed to load Cipher object: {e}")
		return None

	eng_profile = StatisticalProfile.from_english_corpus(no_cache=False)
	cip_profile = StatisticalProfile.from_cipher(cipher, save=True, no_cache=False)
	scorer = NGramScorer(4, "english_quadgrams.txt")

	if relaxation:
		rel_solver = RelaxationSolver(eng_profile, cip_profile, cipher)
		solver = McmcSolver(eng_profile, cip_profile, cipher, scorer)
		start_time = time.time()
		rel_solver.run()
		rel_key = rel_solver.guesses
		solver.run(max_iters=iters, initial_key=rel_key)
	else:
		solver = McmcSolver(eng_profile, cip_profile, cipher, scorer)
		start_time = time.time()
		solver.run(max_iters=iters)

	end_time = time.time()
	solver.time = end_time - start_time

	analytics = SolverAnalytics(solver=solver)
	return analytics


def run_restarts(
	cipher_name: str,
	relaxation: bool = False,
	iters: int = 50000,
	restarts: int = 1,
) -> SolverAnalytics:
	"""Run the full analysis pipeline for a single cipher.

	Args:
		cipher_name (str): Name of the cipher to analyze
		relaxation (bool, optional): Whether to run the relaxation
			solver or not. Defaults to False.
		iters (int, optional): Number of iterations to run. Defaults to 50000.
		restarts (int, optional): Number of restarts to run. Defaults to 1.

	Returns:
		SolverAnalytics: Solver analytics object

	"""
	best_analytics: SolverAnalytics | None = None
	for i in range(restarts):
		log.info(f"Starting analysis for {cipher_name} (Restart {i + 1}/{restarts})...")
		analytics: SolverAnalytics | None = run_analysis(
			cipher_name,
			relaxation,
			iters,
		)
		if not analytics or not isinstance(analytics.solver, McmcSolver):
			log.error(
				f"Failed analysis for {cipher_name} (Restart {i + 1}/{restarts}).",
			)
			continue
		if (
			best_analytics is None
			or analytics.solver.best_score > best_analytics.solver.best_score  # type: ignore
		):
			best_analytics = analytics
	if not best_analytics:
		raise ValueError("No results found.")
	return best_analytics


def itereate_ciphers(
	ciphers: list[str],
	relaxation: bool = False,
	iters: int = 50000,
) -> list[SolverAnalytics]:
	"""Run the full analysis pipeline for a list of ciphers.

	Args:
		ciphers (list[str]): List of cipher names
		relaxation (bool, optional): Whether to run the relaxation solver.
			Defaults to False.
		iters (int, optional): Number of iterations to run. Defaults to 50000.

	Returns:
		list[SolverAnalytics]: List of solver analytics objects

	"""
	results: list[SolverAnalytics] = []
	for name in ciphers:
		best_analytics: SolverAnalytics = run_restarts(name, relaxation, iters)
		results.append(best_analytics)

	if results:
		results.sort(key=lambda r: r.solver.cipher.name)
		for r in results:
			log.info(
				f"Result for {r.solver.cipher.name}: {r.solver.decoded}"
				f" (SER: {r.ser:.4f}, MER: {r.mer:.4f})",
			)
			r.save(
				f"{RESULT_PATH_MCMC}{r.solver.cipher.name}-mcmc"
				f"{'-relaxation' if relaxation else ''}.json",
			)
	else:
		log.error("No results to save.")

	return results


if __name__ == "__main__":
	ciphers = load_ciphers_list()
 
	results = itereate_ciphers(ciphers, relaxation=False, iters=50000)	
	save_results_as_latex(results, "mcmc-results-no-relaxation.tex")
	results = itereate_ciphers(ciphers, relaxation=True, iters=50000)
	save_results_as_latex(results, "mcmc-results-relaxation.tex")
