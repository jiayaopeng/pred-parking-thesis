import model.seattle

def Generator(source, target, input_dim, hidden_dim, output_dim, use_batchnorm, pixelda=False):
        return model.seattle.Feature(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, use_batchnorm=use_batchnorm)


def Classifier(source, target, hidden_dim, output_dim, class_dim, use_batchnorm):

    return model.seattle.Predictor(hidden_dim=hidden_dim, output_dim=output_dim, class_dim=class_dim, use_batchnorm=use_batchnorm)

    
#def ResnetBlock():
#    return model.svhn2mnist.ResnetBlock()
