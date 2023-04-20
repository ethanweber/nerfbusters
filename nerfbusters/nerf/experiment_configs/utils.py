import copy
import glob
import os
import random
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import GPUtil
import tyro
from typing_extensions import Annotated

from nerfstudio.configs.base_config import PrintableConfig
from nerfstudio.utils.scripts import run_command

@dataclass
class Argument:
    """Arg name."""

    name: str
    arg_string: str
    exclude: Optional[set] = None


def get_experiment_name_and_argument_combinations(arguments_list_of_lists: List[List[Argument]]) -> List[str]:
    """Get the experiment name and argument combinations."""
    list_of_lists = [[]]
    while arguments_list_of_lists:
        arguments_list = arguments_list_of_lists.pop(0)
        # check which lists we add this to
        new_list_of_lists = []
        for list_ in list_of_lists:
            new_list_ = copy.deepcopy(list_)  # will be split as needed
            arguments_to_add = []
            for argument in arguments_list:
                arguments_to_add.append(argument)
            if len(arguments_to_add) == 0:
                new_list_of_lists.append(new_list_)
            else:
                for argument in arguments_to_add:
                    new_list = copy.deepcopy(new_list_)
                    new_list.append(argument)
                    new_list_of_lists.append(new_list)
        list_of_lists = new_list_of_lists

    experiment_names = []
    argument_combinations = []
    for list_ in list_of_lists:
        experiment_name = "---".join([argument.name for argument in list_])
        argument_string = " ".join([argument.arg_string for argument in list_])
        experiment_names.append(experiment_name)
        argument_combinations.append(argument_string)

    return experiment_names, argument_combinations