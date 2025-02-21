import os
import random

import numpy as np
import tensorflow as tf


def optional_global_determinism(func):
    def wrapper(*args, **kwargs):
        seed = kwargs.get(
            "seed", None
        )  # Get seed from keyword arguments, default to None
        if seed is not None:
            # Set the environment variable and seed values for repeatable results
            os.environ["PYTHONHASHSEED"] = str(seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        return func(*args, **kwargs)

    return wrapper
