import os
import ast

from rizaio import Riza
from langgraph.prebuilt import create_react_agent
from core.llm_models import default_llm_models

riza = Riza(
    api_key=os.environ.get("RIZA_API_KEY"),
)


def check_compile(code_string: str) -> tuple[bool, str]:
    """
    Test if a Python code string can be compiled.

    Args:
        code_string (str): Python code to test

    Returns:
        tuple: (bool, str) - (True if compiles, error message if any)
    """
    try:
        ast.parse(code_string)
        compile(code_string, "<string>", "exec")
        return True, "Code compiles successfully"
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except (ValueError, TypeError) as e:
        return False, f"Compilation error: {str(e)}"


def run_python_tool(code: str) -> str:
    """Executes Python code.

    Args:
        code (str): The Python code to execute.

    Returns:
        str: The output of the executed code.
    """
    print("Running code:", code)
    is_valid, error_message = check_compile(code)
    if not is_valid:
        return f"Code compilation failed: {error_message}"
    response = riza.command.exec(
        runtime_revision_id=os.environ.get("RIZA_RUNTIME_ID"),
        language="python",
        code=code,
    )
    print("Riza response:", response)
    return response


model = default_llm_models.coding_model

coding_agent = create_react_agent(
    model=model,
    tools=[run_python_tool, check_compile],
    prompt=(
        """
                You are an expert Python agent to write, run, and debug code efficiently, including math and data tasks.

                Supervisor
                - Strictly follow only the latest Supervisor instruction.
                - If a key detail is missing, ask one concise question; otherwise proceed with a reasonable assumption.

                Environment
                - Available: Python standard library + numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, requests, regex.
                - Anything else is not installed and cannot be installed here.

                Tools
                - run_python_tool: executes Python.
                    MANDATORY: All results you want captured must be printed to stdout using print(...).
                    Returning values or leaving bare expressions will NOT appear in the tool output.
                    Always include at least one print that clearly shows the final answer.
                    Tips:
                    - Arrays/Numbers: print(result)
                    - pandas: print(df.head() or summary)
                    - Multiple values: print a concise summary line
                    - Plots: plt.show() and also print a short text summary
                - check_compile: validates Python syntax. If a dependency is outside the allowed list, use this to verify syntax only and do not execute.

                Debugging
                - If errors indicate a missing package (ImportError/ModuleNotFoundError), do not try to execute again or install; state it’s not executable here.
                - Otherwise diagnose, fix, and re-run.

                Math
                - Use built-ins and modules like math, cmath, fractions, decimal, statistics, random. Show formulas and results when helpful.
                - For complex calculations, use numpy and scipy.

                Standards & Workflow
                - Follow PEP 8; write clear, efficient, well-commented code with robust error handling.
                - Outline briefly, implement incrementally, quick tests, run, verify, refine.

                Output
                - You MUST output: a brief outline your code, the full original code, and execution results, error raised if any.
    """
    ),
    name="coding_agent",
)
