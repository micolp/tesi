import ge_config as cfg

def log(message, verbose_level):
    if verbose_level <= cfg.verbose_level:
        print(message)

