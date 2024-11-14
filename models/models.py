from .single_model import SingleModel
# from models.single_model import SingleModel


def create_model(opt):
    print(opt.model)
    model = SingleModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
