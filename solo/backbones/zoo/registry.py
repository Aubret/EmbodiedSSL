__all__ = ['list_models', 'register_model', 'model_registry']

model_registry = {}


def register_model():
    def inner_decorator(fn):
        # add entries to registry dict/sets
        model_name = fn.__name__
        model_registry[model_name] = fn
        return fn
    return inner_decorator


def list_models():
    """ Return list of available model names
    """
    return list(model_registry.keys())

