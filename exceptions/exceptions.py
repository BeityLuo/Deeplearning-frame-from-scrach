class ShapeNotMatchException(Exception):
    def __init__(self, obj, infor: "str"):
        """
        :param obj: 发生异常的对象
        :param infor: 发生异常的原因
        """
        super(ShapeNotMatchException, self).__init__(infor)
        print("EXCEPTION!!!! In class " + str(type(obj)) + ":  "+ infor)

