import os
from utils.constants import RESULT_PATH
from classes.solver_analytics import SolverAnalytics
from utils.data import load_results
from utils.logging import get_colored_logger

log = get_colored_logger(__name__)


def save_results_as_latex(
	results: list["SolverAnalytics"], name: str = "result_table",
) -> None:
	"""Save results as a LaTeX table.

	Args:
		results (list[SolverAnalytics]): The results to save
		name (str, optional): The name of the LaTeX table. Defaults to "result_table".

	Returns:
		None

	"""
	table_str = "\\begin{table}[]\n"

	prebody_str = indent(
		"\\centering\n"
		"\\label{tab:results}\n"
		"\\caption{Results of the relaxation solver on various ciphers.}\n"
		"\\begin{tabular}{l r r r}\n",
		1,
	) + indent("\\toprule\nCipher & SER & MER & Time \\\\ \n\\midrule\n", 2)

	postbody_str = (
		indent("\\bottomrule\n", 2) + indent("\\end{tabular}\n", 1) + "\\end{table}\n"
	)

	tabular_str = ""

	for result in results:
		tabular_str += indent(
			f"\\texttt{{{result.solver.cipher.name.replace('_', '\\_')}}} & "
			f"{result.ser:.4f} & {result.mer:.4f} & {result.solver.time:.4f} \\\\ \n",
			2,
		)

	table_str += prebody_str
	table_str += tabular_str
	table_str += postbody_str

	log.debug("Saving Latex table...")
	# Save as easy copy-pastable LaTeX table
	try:
		if not os.path.exists(RESULT_PATH):
			os.makedirs(RESULT_PATH, exist_ok=True)
			
		full_path = os.path.join(RESULT_PATH, f"{name}.txt")
		log.debug(f"Saving table to {full_path}")
		
		with open(full_path, "w") as f:
			f.write(table_str)
			
	except Exception as e:
		# Catch any file system or permission error and log it clearly
		log.error(f"FATAL ERROR: Could not save LaTeX table to disk. Permission denied or path error. Error: {e}")
		# Log the table content so the user can copy-paste it manually if disk access fails
		print("\n[Copy-Paste Fallback LaTeX Content Below]:\n" + table_str + "\n")


def indent(text: str, level: int = 1) -> str:
	"""Indent a string by a given level.

	Args:
		text (str): The text to indent
		level (int, optional): The level of indentation. Defaults to 1.

	Returns:
		str: The indented text

	"""
	lines = text.splitlines()
	return "\n".join(["  " * level + line for line in lines[: len(lines)]]) + "\n"


if __name__ == "__main__":
	results = load_results()
	save_results_as_latex(results)
