import subprocess
import sys
import os
import importlib.util


# ---------------------------------------------------------------------------
# Installations
# ---------------------------------------------------------------------------


def install_requirements(file_path: str) -> bool:
    """
    Installs Python packages from a requirements file.

    The function executes pip using the current Python interpreter and installs
    all dependencies listed in the provided requirements file.

    Behavior:
    - Runs 'pip install -r <file_path>'.
    - Returns True if installation completes successfully.
    - Handles installation failures and missing requirements files gracefully.

    :param file_path: Path to the requirements file containing package dependencies.
    :type file_path: str
    :return: True if all requirements were installed successfully, otherwise False.
    :rtype: bool
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", file_path])
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during installation: {e}")
        return False
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return False


def create_project_dirs():
    directories = ['data', 'output']
    for folder in directories:
        os.makedirs(folder, exist_ok=True)
        print(f"Directory '{folder}' is ready.")    


def run_setup():
    create_project_dirs()
    print("installiation res: ", install_requirements("requirements.txt") )


def check_installed_packages():
    packages = ['scipy', 'sweetviz', 'matplotlib', 'seaborn', 'sklearn',
                'statsmodels', 'pmsampsize', 'dcurves']
    all_installed = True
    for pkg in packages:
        if importlib.util.find_spec(pkg) is not None:
            print(f"[INSTALLED] {pkg}")
        else:
            print(f"[MISSING]   {pkg}")
            all_installed = False
    return all_installed