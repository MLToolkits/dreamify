import functools


def validate_positive(value, name, allow_zero=True, value_type=(int, float)):
    """Helper function to validate that a value is positive (or non-negative if allow_zero=True)."""
    if value is not None and (
        not isinstance(value, value_type) or (value < 0 if allow_zero else value <= 0)
    ):
        raise ValueError(
            f"{name} must be a {'non-negative' if allow_zero else 'positive'} {value_type[0].__name__}."
        )


def validate_dream(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Validate parameters using the helper function
        validate_positive(kwargs.get("learning_rate"), "learning_rate", allow_zero=True)
        validate_positive(
            kwargs.get("num_octave"), "num_octave", allow_zero=False, value_type=(int,)
        )
        validate_positive(
            kwargs.get("octave_scale"),
            "octave_scale",
            allow_zero=False,
            value_type=(float,),
        )
        validate_positive(
            kwargs.get("iterations"), "iterations", allow_zero=False, value_type=(int,)
        )
        validate_positive(kwargs.get("max_loss"), "max_loss", allow_zero=False)
        validate_positive(kwargs.get("duration"), "duration", allow_zero=True)

        return func(*args, **kwargs)

    return wrapper


__all__ = [validate_dream]
