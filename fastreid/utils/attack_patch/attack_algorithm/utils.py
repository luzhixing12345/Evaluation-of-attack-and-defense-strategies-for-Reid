
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class Attack(object):
    """
    Abstract base class for all attack classes.

    :param predict: forward pass function.
    :param loss_fn: loss function that takes .
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.

    """

    def __init__(self, predict, loss_fn, clip_min, clip_max):
        """Create an Attack instance."""
        self.predict = predict
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max
        #query(Q) set class number of Market1501(M) and DukeMTMC(D)
        self.M = 750       
        self.D = 702
    
    def get_num_classes(self):
        dict = {'M':self.M,'D':self.D}
        key = self.cfg.DATASETS.NAMES[0][0]
        return dict[key]

    def perturb(self, x, **kwargs):
        """Virtual method for generating the adversarial examples.

        :param x: the model's input tensor.
        :param **kwargs: optional parameters used by child classes.
        :return: adversarial examples.
        """
        error = "Sub-classes must implement perturb."
        raise NotImplementedError(error)
    

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)


class LabelMixin(object):
    def _get_predicted_label(self, x):
        """
        Compute predicted labels given x. Used to prevent label leaking
        during adversarial training.

        :param x: the model's input tensor.
        :return: tensor containing predicted labels.
        """
        with torch.no_grad():
            outputs = self.predict(x)
        softmax = torch.nn.Softmax(dim=1)
        outputs = softmax(outputs)
        _, y = torch.max(outputs, dim=1)
        return y

    def _verify_and_process_inputs(self, x, y):

        if y==None:             # y is none means it's a classification attack method
            if self.targeted:
                y = self._get_predicted_label(x)
                num_classes = self.get_num_classes()
                y = torch.randint(num_classes,y.shape)
            else:
                y = self._get_predicted_label(x)

            
        x = self.replicate_input(x)
        y = self.replicate_input(y)
        return x, y

    def replicate_input(self,x):
        return x.detach().clone().to(device)




