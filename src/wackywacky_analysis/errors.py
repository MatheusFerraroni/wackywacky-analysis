class WackyWackyError(RuntimeError):
    """Erro seguro para ser exibido pela CLI."""


class ConfigurationError(WackyWackyError):
    pass


class SourceChangedError(WackyWackyError):
    pass


class ReviewRequired(WackyWackyError):
    pass


class ReviewRejected(WackyWackyError):
    pass
