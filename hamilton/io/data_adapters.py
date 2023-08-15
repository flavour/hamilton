import abc
import dataclasses
import typing
from typing import Any, Collection, Dict, Tuple, Type

from hamilton.htypes import custom_subclass_check


class AdapterCommon(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def applicable_types(cls) -> Collection[Type]:
        """Returns the types that this data loader can load to.
        These will be checked against the desired type to determine
        whether this is a suitable loader for that type.

        Note that a loader can load to multiple types. This is the function to
        override if you want to add a new type to a data loader.

        Note if you have any specific requirements for loading types (generic/whatnot),
        you can override applies_to as well, but it will make it much harder to document/determine
        what is happening.

        :return:
        """
        pass

    @classmethod
    def applies_to(cls, type_: Type[Type]) -> bool:
        """Tells whether or not this data loader can load to a specific type.
        For instance, a CSV data loader might be able to load to a dataframe,
        a json, but not an integer.

        This is a classmethod as it will be easier to validate, and we have to
        construct this, delayed, with a factory.

        :param type_: Candidate type
        :return: True if this data loader can load to the type, False otherwise.
        """
        for load_to in cls.applicable_types():
            if custom_subclass_check(load_to, type_):
                return True
        return False

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """Returns the name of the data loader. This is used to register the data loader
        with the load_from decorator.

        :return: The name of the data loader.
        """
        pass

    @classmethod
    def _ensure_dataclass(cls):
        if not dataclasses.is_dataclass(cls):
            raise TypeError(
                f"DataLoader subclasses must be dataclasses. {cls.__qualname__} is not."
                f" Did you forget to add @dataclass?"
            )

    @classmethod
    def get_required_arguments(cls) -> Dict[str, Type[Type]]:
        """Gives the required arguments for the class.
        Note that this just uses the type hints from the dataclass.

        :return: The required arguments for the class.
        """
        cls._ensure_dataclass()
        type_hints = typing.get_type_hints(cls)
        return {
            field.name: type_hints.get(field.name)
            for field in dataclasses.fields(cls)
            if field.default == dataclasses.MISSING
        }

    @classmethod
    def get_optional_arguments(cls) -> Dict[str, Type[Type]]:
        """Gives the optional arguments for the class.
        Note that this just uses the type hints from the dataclass.

        :return: The optional arguments for the class.
        """
        cls._ensure_dataclass()
        type_hints = typing.get_type_hints(cls)
        return {
            field.name: type_hints.get(field.name)
            for field in dataclasses.fields(cls)
            if field.default != dataclasses.MISSING
        }

    @classmethod
    def can_load(cls) -> bool:
        """Returns whether this adapter can "load" data.
        Subclasses are meant to implement this function to
        tell the framework what to do with them.

        :return:
        """
        return False

    @classmethod
    def can_save(cls) -> bool:
        """Returns whether this adapter can "save" data.
        Subclasses are meant to implement this function to
        tell the framework what to do with them.

        :return:
        """
        return False


class DataLoader(AdapterCommon, abc.ABC):
    """Base class for data loaders. Data loaders are used to load data from a data source.
    Note that they are inherently polymorphic -- they declare what type(s) they can load to,
    and may choose to load differently depending on the type they are loading to.

    Note that this is not yet a public-facing API -- the current set of data loaders will
    be managed by the library, and the user will not be able to create their own.

    We intend to change this and provide an extensible user-facing API,
    but if you subclass this, beware! It might change.
    """

    @abc.abstractmethod
    def load_data(self, type_: Type[Type]) -> Tuple[Type, Dict[str, Any]]:
        """Loads the data from the data source.
        Note this uses the constructor parameters to determine
        how to load the data.

        :return: The type specified
        """
        pass

    @classmethod
    def can_load(cls) -> bool:
        return True


class DataSaver(AdapterCommon, abc.ABC):
    """Base class for data savers. Data savers are used to save data to a data source.
    Note that they are inherently polymorphic -- they declare what type(s) they can save from,
    and may choose to save differently depending on the type they are saving from.

    Note that this is not yet a public-facing API -- the current set of data savers will
    be managed by the library, and the user will not be able to create their own.

    We intend to change this and provide an extensible user-facing API,
    but if you subclass this, beware! It might change.
    """

    @abc.abstractmethod
    def save_data(self, data: Any) -> Dict[str, Any]:
        """Saves the data to the data source.
            Note this uses the constructor parameters to determine
            how to save the data.

        :return: Any relevant metadata. This is up the the data saver, but will likely
            include the URI, etc... This is going to be similar to the metadata returned
            by the data loader in the loading tuple.
        """
        pass

    @classmethod
    def can_save(cls) -> bool:
        return True
