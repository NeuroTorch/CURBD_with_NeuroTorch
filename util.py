import os


def save_str_to_file(save_path: str, string: str):
    """
    Save a string to a file.

    :param save_path: The path to save the string to.
    :param string: The string to save.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w+") as f:
        f.write(string)
