def opt_ease(*dargs, **dkargs):
    # get model dir.
    # deal with the parameters in decorators.
    # print(dargs)

    def _dec(func):
        def dec(*args, **kwargs):
            # deal with the parameters in the function decorated.
            result = func(args[0])
            return result
        return dec
    return _dec
