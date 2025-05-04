from typing import Awaitable, TypeVar

R = TypeVar("R")


class ObjectRef(Awaitable[R]): # type: ignore
    pass


class ObjectID(Awaitable[R]): # type: ignore
    pass


class ObjectRefGenerator(Awaitable[R]): # type: ignore
    """A generator to obtain object references
    from a task in a streaming manner.
    """
    pass


class DynamicObjectRefGenerator(Awaitable[R]): # type: ignore
    pass
