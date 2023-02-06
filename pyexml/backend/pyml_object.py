
class PYMLObject():
    """Base class for all PYML objects. All PYML objects are derived from this class.
    This class is used to store the name and type of the object, as well as any other
    information that is passed to the object.
    Args:
        name (str): The name of the object
        **kwargs: Additional arguments to pass to the object
        
        Attributes:
        name (str): The name of the object
        kwargs (dict): A dictionary containing the additional arguments passed to the object
        info_dict (dict): A dictionary containing the name and type of the object, as well as any other
            information that is passed to the object.
            """
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.info_dict = {}
        self.info_dict['Name'] = self.name
        self.info_dict['Type'] = self.__class__.__name__
        self.info_dict.update(self.kwargs)
        self.id = self.name

    def info(self):
        """Returns the info_dict attribute of the object.
        Returns:
            dict: The info_dict attribute of the object."""
        return self.info_dict

    def __repr__(self):
        return str(self.info_dict)
    
    def __str__(self):
        return str(self.info_dict)

    def id(self, id=None):
        """Returns the id of the object.
        Args:
            id (int): The id of the object
            Returns:
                int: The id of the object.
                """
        if id is None:
            return self.id
        else:
            self.id = id
            return id