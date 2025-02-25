import functools


def validate_positive(value, name, allow_zero=True, value_type=(int, float)):
    """Helper function to validate that a value is positive (or non-negative if allow_zero=True)."""
    if value is None:
        return

    if isinstance(value, slice):
        if not all(
            isinstance(v, (int, type(None)))
            for v in (value.start, value.stop, value.step)
        ):
            raise ValueError(f"{name} must be a slice with integer (or None) values.")
        return

    if not isinstance(value, value_type) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(
            f"{name} must be a {'non-negative' if allow_zero else 'positive'} {value_type[0].__name__}."
        )


def validate_save_media(will_save_name, will_save_val, duration_name, duration_val):
    if duration_val != 0 and will_save_val is False:
        raise ValueError(
            f"{will_save_name} is set to False while {duration_name} is set to {duration_val}. Set {will_save_name} to True."
        )


def validate_dream_params(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
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
        validate_positive(kwargs.get("vid_duration"), "vid_duration", allow_zero=False)
        validate_positive(kwargs.get("gif_duration"), "gif_duration", allow_zero=False)
        validate_positive(kwargs.get("duration"), "duration", allow_zero=False)

        validate_save_media("save_vid", kwargs.get("save_vid"), "vid_duration", kwargs.get("vid_duration"))
        validate_save_media("save_gif", kwargs.get("save_gif"), "gif_duration", kwargs.get("gif_duration"))
        
        return func(*args, **kwargs)

    return wrapper


__all__ = [validate_dream_params]
