
class BaseModel():

    def __init__(self, input_shape, output_size):
        self.input_shape = input_shape
        self.output_size = output_size

        self.build_model()

        self.model.summary()

    def fit(self, x_train, y_train, **kwargs):
        return self.model.fit(x_train, y_train, **kwargs)

    def compile(self, *argv, **kwargs):
        self.model.compile(*argv, **kwargs)

    def evaluate(self, *argv, **kwargs):
        return self.model.evaluate(*argv, **kwargs)

    def save(self, path):
        self.model.save(path)