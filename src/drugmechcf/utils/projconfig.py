"""
Project specific configuration
"""

import datetime
import os
from pathlib import Path

from drugmechcf.utils.misc import in_databricks_and_dbfs


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

# Shell vars containing path to dir containing DrugMechCounterfactuals clone.
SHELLVAR_PROJDIR = "PROJDIR"
SHELLVAR_DRUGMECHCFDIR = "DRUGMECHCFDIR"

# Shell Var for getting DataBricks UserName
SHELLVAR_DATABRICKS_USER_NAME = "DATABRICKS_USER_NAME"


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------


class InvalidEnvironmentError(TypeError):
    pass


class ProjectConfig:

    def __init__(self, project_name: str):
        assert project_name is not None, "Must provide arg `project_name`"
        self.project_name = project_name
        return

    def get_input_basedir(self, verbose=True) -> str:
        """
        Get the base-dir for input files created for this project.

        In Databricks, it will be "/dbfs/Users/{user}/{proj_name}/".
        Otherwise, it will use "$PROJDIR/".
        """
        if in_databricks_and_dbfs():
            basedir = get_project_data_basedir_databricks(self.project_name)
        else:
            basedir = get_project_base_dir()

        if verbose and not os.path.isdir(basedir):
            print(f"WARNING:  Project base dir  `{basedir}`  does not exist!")

        return basedir

    def get_output_basedir(self, create_it=False, verbose=True) -> str:
        """
        Get the base-dir for output files created for this project.
        Create it, if it does not already exist.

        In Databricks, it will be "/dbfs/Users/{user}/{proj_name}".
        Otherwise, it will use "$PROJDIR/Temp".
        """
        if in_databricks_and_dbfs():
            basedir = get_project_data_basedir_databricks(self.project_name)
        else:
            basedir = get_project_local_temp_dir()

        if create_it and not os.path.isdir(basedir):
            os.makedirs(basedir)
            if verbose:
                print("Created project base dir:", basedir)

        return basedir

    def get_input_data_dir(self, verbose=True) -> str:
        """
        Where input data is kept.
        $input_dir/Data
        """
        basedir = self.get_input_basedir(verbose=verbose)
        data_dir = os.path.join(basedir, "Data")
        return data_dir

    def pp_config(self):
        print("ProjectConfig:")
        print("    project_name =", self.project_name)
        print("    input_basedir =", self.get_input_basedir())
        print("    input_data_dir =", self.get_input_data_dir())
        print("    output_basedir =", self.get_output_basedir())
        print()
        return
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------

def get_project_base_dir():
    """
    Path to $PROJDIR
    """
    if projdir := os.environ.get(SHELLVAR_DRUGMECHCFDIR):
        return projdir
    elif projdir := os.environ.get(SHELLVAR_PROJDIR):
        return projdir

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    try:
        python_base_idx = dir_path.index("src")
        base_dir = dir_path[:python_base_idx]
    except ValueError:
        # based on module path $PROJDIR/framework/projconfig.py
        base_dir = Path(dir_path).parent

    return base_dir


def get_project_local_temp_dir():
    """
    Path to $PROJDIR/Temp
    """
    temp_dir = os.path.join(get_project_base_dir(), "Temp")
    return temp_dir


def get_project_data_basedir_databricks(proj_name: str) -> str:
    """
    /dbfs/Users/{user}/{proj_name}
    """
    if not in_databricks_and_dbfs():
        raise InvalidEnvironmentError("Not in Databricks")

    # This would get set in the Cluster Configuration
    user = os.environ.get(SHELLVAR_DATABRICKS_USER_NAME, "UnknownUser")

    basedir = f"/dbfs/Users/{user}/{proj_name}"

    return basedir


def get_project_config(proj_name=None):
    if proj_name is None:
        # Use basename of $PROJDIR
        proj_name = os.path.basename(get_project_base_dir())
    return ProjectConfig(project_name=proj_name)
