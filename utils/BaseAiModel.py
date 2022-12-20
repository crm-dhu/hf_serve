class BaseAiModel():
    def __init__(self):
        pass

    def preprocess(self, text):
        pass

    def forward(self):
        pass

    def postprocess(self):
        pass

    def __call__(self, text):
        self.preprocess(text)
        self.forward()
        return self.postprocess()