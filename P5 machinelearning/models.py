import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w


    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x,self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        dotProduct = self.run(x)
        dotProduct = nn.as_scalar(dotProduct)
        if dotProduct >= 0.0:
            return 1
        else:
             return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """

        batch_size = 1
        while True:
            noMistake = True
            for x,y in dataset.iterate_once(batch_size):
                #print(x)
                #print(y)
                #break
                if self.get_prediction(x) != nn.as_scalar(y):
                    noMistake = False
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
            if noMistake:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 1
        self.w1 = nn.Parameter(1, 72)
        self.w2 = nn.Parameter(72, 1)
        self.b1 = nn.Parameter(1, 72)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        xw_1 = nn.Linear(x, self.w1)
        predicted_y = nn.AddBias(xw_1, self.b1)
        r1d1 = nn.ReLU(predicted_y)
        xw_2 = nn.Linear(r1d1, self.w2)
        predicted_y2 =  nn.AddBias(xw_2, self.b2)
        #r2d2 = nn.ReLU(predicted_y2)




        return predicted_y2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        while True:

            for x,y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad =  nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])

                self.w1.update(grad[0], -self.learning_rate )
                self.b1.update(grad[1], -self.learning_rate )
                self.w2.update(grad[2], -self.learning_rate )
                self.b2.update(grad[3], -self.learning_rate )

            if nn.as_scalar( self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02:
                break

        return




class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 1
        self.w1 = nn.Parameter(784, 100)
        self.w2 = nn.Parameter(100, 10)
        self.b1 = nn.Parameter(1, 100)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xw_1 = nn.Linear(x, self.w1)
        predicted_y = nn.AddBias(xw_1, self.b1)
        r1d1 = nn.ReLU(predicted_y)
        xw_2 = nn.Linear(r1d1, self.w2)
        predicted_y2 =  nn.AddBias(xw_2, self.b2)
        return predicted_y2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        while True:

            for x,y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients =  nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])

                self.w1.update(gradients[0], -self.learning_rate)
                self.b1.update(gradients[1], -self.learning_rate)
                self.w2.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)

            if (dataset.get_validation_accuracy() >= 0.975):
                break

        return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 20
        self.learning_rate = .03
        self.hidden_size = 150

        self.w1 = nn.Parameter(self.num_chars, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b2 = nn.Parameter(1, self.hidden_size)
        self.w3 = nn.Parameter(self.hidden_size, 5)
        self.b3 = nn.Parameter(1, 5)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        score = 0
        for i in xs:
            if score == 0:
                score+= 1
                h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.w1), self.b1))
            else:
                h = nn.ReLU(nn.Add(nn.AddBias(nn.Linear(i, self.w1), self.b1), nn.AddBias(nn.Linear(h, self.w2), self.b2)))

        return nn.AddBias(nn.Linear(h, self.w3), self.b3)

        """
        #z = xW + hW
        h = nn.Linear(xs[0], self.W)
        z = h
        for i, x in enumerate(xs[1:]):
            z = nn.Add(nn.Linear(x, self.W), nn.Linear(z, self.W_hidden))

        return nn.Linear(z, self.final)
        """

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_ys = self.run(xs)
        return nn.SoftmaxLoss(predicted_ys, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        #while True:
        losses = []
        score = 0

        for x,y in dataset.iterate_forever(self.batch_size):
            score += 1
                #print("SCORE: ", score)
            loss = self.get_loss(x, y)
            losses += [nn.as_scalar(loss)]
            gradients =  nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
            #if score >= 5000:
                #if dataset.get_validation_accuracy() > 0.85:
                #    break

            self.w1.update(gradients[0], -self.learning_rate)
            self.b1.update(gradients[1], -self.learning_rate)
            self.w2.update(gradients[2], -self.learning_rate)
            self.b2.update(gradients[3], -self.learning_rate)
            self.w3.update(gradients[4], -self.learning_rate)
            self.b3.update(gradients[5], -self.learning_rate)

            if (dataset.get_validation_accuracy() >= 0.88):
                break

        #return
