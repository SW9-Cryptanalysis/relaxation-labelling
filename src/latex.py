import os
from utils.constants import RESULT_PATH
from classes.solver_analytics import SolverAnalytics
from utils.data import load_results


def save_results_as_latex(results: list["SolverAnalytics"], name: str = "result_table") -> None:
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
			f"\\texttt{{{result.cipher.name.replace('_', '\\_')}}} & {result.ser:.4f} & {result.mer:.4f} & {result.solver.time:.4f} \\\\ \n",
			2,
		)

	table_str += prebody_str
	table_str += tabular_str
	table_str += postbody_str

	# Save as easy copy-pastable LaTeX table
	if not os.path.exists(RESULT_PATH):
		os.makedirs(RESULT_PATH)
	with open(f"{RESULT_PATH}/{name}.txt", "w") as f:
		f.write(table_str)
		print("Saved latex")


def indent(text: str, level: int = 1) -> str:
	lines = text.splitlines()
	return "\n".join(["  " * level + line for line in lines[: len(lines)]]) + "\n"


if __name__ == "__main__":
	results = load_results()
	save_results_as_latex(results)
