"""
This script updates and executes batch files for model optimization and generation.
"""

import os
from subprocess import Popen
import argparse


def update_bat_file(batch_file_path, var_update_dict):
    """
    Updates specific variables in a batch file with new values provided in a dictionary.
    Example:
        update_bat_file("path/to/batch_file.bat", {
            "SRC_DIR": "new/source/dir",
            "SRC_FILE": "new/source/file",
            "OPTIMISE_FILE": "new/optimise/file",
            "GEN_DIR": "new/gen/dir"
        })
    """

    # read the contents of the batch file
    with open(batch_file_path, "r", encoding="utf-8") as f:
        content = f.readlines()

    # loop through the lines of the batch file and update the variables
    for i, cont in enumerate(content):
        if cont.startswith("set MODEL_SRC_DIR="):
            content[i] = f'set MODEL_SRC_DIR={var_update_dict["SRC_DIR"]}\n'
        elif cont.startswith("set MODEL_SRC_FILE="):
            content[i] = f'set MODEL_SRC_FILE={var_update_dict["SRC_FILE"]}\n'
        elif cont.startswith("set MODEL_OPTIMISE_FILE="):
            content[i] = f'set MODEL_OPTIMISE_FILE={var_update_dict["OPTIMISE_FILE"]}\n'
        elif cont.startswith("set GEN_SRC_DIR="):
            content[i] = f'set GEN_SRC_DIR={var_update_dict["GEN_DIR"]}\n'

    # write the updated content to the batch file
    with open(batch_file_path, "w", encoding="utf-8") as f:
        f.writelines(content)

    print("Batch file content updated successfully.")


if __name__ == "__main__":

    def str2bool(v):
        """
        Convert a string representation of truth to a boolean.
        Valid true values are: "yes", "true", "t", "y", "1".
        Valid false values are: "no", "false", "f", "n", "0".
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()

    parser.add_argument("--SRC_DIR", type=str, default="..\\workspace\\catsdogs\\tflite_model", help="The model source dir")
    parser.add_argument("--SRC_FILE", type=str, default="mobilenet_v2_int8quant.tflite", help="The model source file name")
    parser.add_argument("--GEN_DIR", type=str, default="..\\workspace\\catsdogs\\tflite_model\\vela", help="The generated dir for *vela.tflite")
    parser.add_argument("--VELA_EN", type=str2bool, default=False, help="Vela compiler enable or not, default is normal C.")
    args = parser.parse_args()

    # Change to ../deployment folder
    old_cwd = os.getcwd()
    batch_cwd = os.path.join(old_cwd, "deployment")
    os.chdir(batch_cwd)

    if args.VELA_EN:
        # Get the MODEL_OPTIMISE_FILE
        if args.SRC_FILE.count(".tflite"):
            MODEL_OPTIMISE_FILE = args.SRC_FILE.split(".tflite")[0] + "_vela.tflite"
        else:
            raise OSError("Please input .tflite file!")

        # Update the variables.bat
        BATCH_VAR_FILE_PATH = "variables.bat"
        batch_var_update_dict = {}
        batch_var_update_dict["SRC_DIR"] = args.SRC_DIR
        batch_var_update_dict["SRC_FILE"] = args.SRC_FILE
        batch_var_update_dict["OPTIMISE_FILE"] = MODEL_OPTIMISE_FILE
        batch_var_update_dict["GEN_DIR"] = args.GEN_DIR
        update_bat_file(BATCH_VAR_FILE_PATH, batch_var_update_dict)

        # Execute the bat file
        print(f'Executing the {os.path.join(batch_cwd, "gen_model_cpp.bat")}.')
        print("Please wait...")
        p = Popen("gen_model_cpp.bat")
        stdout, stderr = p.communicate()
        # subprocess.call(["gen_model_cpp.bat"])

        vela_output_path = os.path.join(old_cwd, args.GEN_DIR.split("..\\")[1], MODEL_OPTIMISE_FILE)
        print(f"Finish, the vela file is at: {vela_output_path}")

    else:
        # Get the MODEL_OPTIMISE_FILE
        if not args.SRC_FILE.count(".tflite"):
            raise OSError("Please input .tflite file!")

        # Update the variables.bat
        BATCH_VAR_FILE_PATH = "variables_no_vela.bat"
        batch_var_update_dict = {}
        batch_var_update_dict["SRC_DIR"] = args.SRC_DIR
        batch_var_update_dict["SRC_FILE"] = args.SRC_FILE
        batch_var_update_dict["GEN_DIR"] = args.GEN_DIR
        update_bat_file(BATCH_VAR_FILE_PATH, batch_var_update_dict)

        # Execute the bat file
        print(f'Executing the {os.path.join(batch_cwd, "gen_model_cpp_no_vela.bat")}.')
        print("Please wait...")
        p = Popen("gen_model_cpp_no_vela.bat")
        stdout, stderr = p.communicate()
        # subprocess.call(["gen_model_cpp.bat"])

        vela_output_path = os.path.join(old_cwd, args.GEN_DIR.split("..\\")[1], args.SRC_FILE)
        print(f"Finish, the c source file is at: {vela_output_path}")

    os.chdir(old_cwd)
