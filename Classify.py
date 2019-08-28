import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt


class KNN:
    """Calculates the nearest neighbor."""
    __dataSet = np.array
    __inX = np.array
    __labels = np.array
    __k = 0
    __dataSetSize = 0
    __differenceMatrix = np.array
    __differenceMatrixSquared = 0
    __distancesSquared = 0
    __distancesUnsorted = 0
    __distancesSorted = 0
    __classCount = {}

    def __init__(self, dataset, inx, labels, k):
        """
        Initialize the class.
        :param dataset: list
        :param inx: data we wish to classify.
        :param labels: Classification labels.
        :param k: Number of iterations.
        """
        self.__dataSet = dataset
        self.__inX = inx
        self.__labels = labels
        self.__k = k

    def __claculate_distance(self):
        self.__dataSetSize = self.__dataSet.shape[0]
        self.__differenceMatrix = np.tile(self.__inX, (self.__dataSetSize, 1)) - self.__dataSet
        self.__differenceMatrixSquared = self.__differenceMatrix ** 2
        self.__distancesSquared = self.__differenceMatrixSquared.sum(axis=1)
        self.__distancesUnsorted = self.__distancesSquared ** 0.5
        self.__distancesSorted = self.__distancesUnsorted.argsort()

    def __calculate_nn(self):
        self.__claculate_distance()

        for i in range(self.__k):
            vote_i_label = self.__labels[self.__distancesSorted[i]]
            self.__classCount[vote_i_label] = self.__classCount.get(vote_i_label, 0) + 1

        sorted_class_count = sorted(self.__classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    def classify(self):
        """Classify the data using kNN algorithm."""
        return self.__calculate_nn()


class DecisionTree:
    """Creates a binary data tree using ID3"""
    __superSet = []
    __labels = []

    def __init__(self, dataset, labels):
        """
        Initialize the class
        :param dataset: list, the main dataset
        :param labels: Labels we wish to classify.
        """
        self.__superSet = dataset
        self.__labels = labels

    def __calculate_entropy(self, dataset):
        """Calculates the Shannon Entropy of the dataset. Entropy is the expected value of the information."""
        number_of_entries = len(dataset)
        label_counts = {}
        entropy = 0.0

        for feature_vector in dataset:
            for label in feature_vector:
                if label not in label_counts.keys():
                    label_counts[label] = 0

                label_counts[label] += 1

        for key in label_counts:
            prob = float(label_counts[key]) / number_of_entries
            entropy -= prob * math.log(prob, 2)

        return entropy

    def __split_dataset(self, dataset, axis, value):
        """Splits the given dataset given at the index of value."""
        ret_dataset = []
        for feature_vector in dataset:
            if feature_vector[axis] == value:
                reduced_feature_vector = feature_vector[:axis]
                reduced_feature_vector.extend(feature_vector[axis + 1:])
                ret_dataset.append(reduced_feature_vector)

        return ret_dataset

    def __choose_best_split_feature(self, dataset):
        """Chooses the best feature to split the dataset with"""
        number_of_features = len(dataset[0]) - 1
        base_entropy = self.__calculate_entropy(dataset)
        best_info_gain = 0.0
        best_feature = -1

        for i in range(number_of_features):
            feature_list = [example[i] for example in dataset]
            unique_values = set(feature_list)
            new_entropy = 0.0
            for value in unique_values:
                sub_dataset = self.__split_dataset(dataset, i, value)
                prob = len(sub_dataset) / float(len(dataset))
                new_entropy += prob * self.__calculate_entropy(sub_dataset)

            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

            return best_feature

    def __majority_vote(self, class_list):
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] += 1

        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count

    def __generate_tree(self, dataset, labels):
        """Generate a Decision tree."""
        class_list = [example[-1] for example in dataset]
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(self.__superSet[0]) == 1:
            return self.__majority_vote(class_list)
        best_feature = self.__choose_best_split_feature(dataset)
        print(best_feature)
        best_feature_label = labels[best_feature]
        ret_tree = {best_feature_label: {}}
        del (labels[best_feature])
        feature_values = [example[best_feature] for example in dataset]
        unique_values = set(feature_values)
        for value in unique_values:
            sub_labels = labels[:]
            ret_tree[best_feature_label][value] = self.__generate_tree(self.__split_dataset(dataset, best_feature,
                                                                                            value), sub_labels)
        return ret_tree

    def classify(self, input_tree, feature_labels, test_vector):
        """Classify and test using Decision Tree algorithm."""
        first_string = input_tree.keys()[0]
        second_dictionary = input_tree[first_string]
        feature_index = feature_labels.index(first_string)
        for key in second_dictionary.keys():
            if test_vector[feature_index] == key:
                if type(second_dictionary[key]).__name__ == 'dict':
                    class_label = self.classify(second_dictionary[key], feature_labels, test_vector)
                else:
                    class_label = second_dictionary[key]

        return class_label

    def create_tree(self):
        return self.__generate_tree(self.__superSet, self.__labels)


class NaiveBayes:
    """Calculates the Naive Bayes of the given dataset based on the categories."""
    __train_dataset = []
    __train_category = []
    __inverse_vector = []
    __verse_vector = []
    __probability = 0.0

    def __init__(self, train_set, train_category):
        """In order: training dataset, training category, testing dataset, and testing category. Note:
                        If you do not wish to test the model, don't feed anything to the test set and test cat.

                        :param train_set: A list.
                        :param train_category: A list.
                        """
        self.__train_dataset = train_set
        self.__train_category = train_category

    def train_naive_bayes(self):
        """
        Train the Naive Bayes algorithm.
        :return: The probabilities of each member in the dataset. Note: the returned dataset are np.array.
        """
        number_of_train_documents = len(self.__train_dataset)
        number_of_words = len(self.__train_dataset[0])
        probability = sum(self.__train_category) / float(number_of_train_documents)
        probability_verse_numerator = np.zeros(number_of_words);
        probability_inverse_numerator = \
            np.zeros(number_of_words)
        probability_verse_denominator = 0.0;
        probability_inverse_denominator = 0.0
        for i in range(number_of_train_documents):
            if self.__train_category[i] == 1:
                probability_inverse_numerator += self.__train_dataset[i]
                probability_inverse_denominator += sum(self.__train_dataset[i])

            else:
                probability_verse_numerator += self.__train_dataset[i]
                probability_verse_denominator += sum(self.__train_dataset[i])
        probability_verse_vector = probability_verse_numerator / probability_verse_denominator
        probability_inverse_vector = probability_inverse_numerator / probability_inverse_denominator

        self.__inverse_vector = probability_inverse_vector
        self.__verse_vector = probability_verse_vector
        self.__probability = probability

        return probability_verse_vector, probability_inverse_vector, probability

    def classify_naive_bayes(self, vector_to_classify):
        """
        Classify using the trained values.
        :param vector_to_classify: List, vector you wish to classify.
        :return: 0 or 1, based on the the training data.
        :raises ValueError if not trained.
        """
        if not self.__verse_vector.any() or not self.__inverse_vector.any():
            raise ValueError("You must train the model first!")

        probable_one = sum(vector_to_classify * self.__inverse_vector) + math.log(self.__probability)
        probable_zero = sum(vector_to_classify * self.__verse_vector) + math.log(1.0 - self.__probability)

        if probable_one > probable_zero:
            return 1
        else:
            return 0


class NaiveBayes:
    """Calculates the Naive Bayes of the given dataset based on the categories."""
    __train_dataset = []
    __train_category = []
    __inverse_vector = []
    __verse_vector = []
    __probability = 0.0

    def __init__(self, train_set, train_category):
        """In order: training dataset, training category, testing dataset, and testing category. Note:
                        If you do not wish to test the model, don't feed anything to the test set and test cat.

                        :param train_set: A list.
                        :param train_category: A list.
                        """
        self.__train_dataset = train_set
        self.__train_category = train_category

    def train_naive_bayes(self):
        """
        Train the Naive Bayes algorithm.
        :return: The probabilities of each member in the dataset. Note: the returned dataset are np.array.
        """
        number_of_train_documents = len(self.__train_dataset)
        number_of_words = len(self.__train_dataset[0])
        probability = sum(self.__train_category) / float(number_of_train_documents)
        probability_verse_numerator = np.zeros(number_of_words);
        probability_inverse_numerator = \
            np.zeros(number_of_words)
        probability_verse_denominator = 0.0;
        probability_inverse_denominator = 0.0
        for i in range(number_of_train_documents):
            if self.__train_category[i] == 1:
                probability_inverse_numerator += self.__train_dataset[i]
                probability_inverse_denominator += sum(self.__train_dataset[i])

            else:
                probability_verse_numerator += self.__train_dataset[i]
                probability_verse_denominator += sum(self.__train_dataset[i])
        probability_verse_vector = probability_verse_numerator / probability_verse_denominator
        probability_inverse_vector = probability_inverse_numerator / probability_inverse_denominator

        self.__inverse_vector = probability_inverse_vector
        self.__verse_vector = probability_verse_vector
        self.__probability = probability

        return probability_verse_vector, probability_inverse_vector, probability

    def classify_naive_bayes(self, vector_to_classify):
        """
        Classify using the trained values.
        :param vector_to_classify: List, vector you wish to classify.
        :return: 0 or 1, based on the the training data.
        :raises ValueError if not trained.
        """
        if not self.__verse_vector.any() or not self.__inverse_vector.any():
            raise ValueError("You must train the model first!")

        probable_one = sum(vector_to_classify * self.__inverse_vector) + math.log(self.__probability)
        probable_zero = sum(vector_to_classify * self.__verse_vector) + math.log(1.0 - self.__probability)

        if probable_one > probable_zero:
            return 1
        else:
            return 0


class LogisticRegression:
    """Classify and optimize using Logistic Regression."""
    __superSet = []
    __classLabels = []

    def __init__(self, dataset, labels):
        """
        The constructor. Feed it the main dataset, and the class labels.
        :param dataset: List. The main dataset.
        :param labels: List. The class labels.
        """
        self.__superSet = np.array(dataset)
        self.__classLabels = np.array(labels)

    def __sigmoid(self, in_z):
        """
        Calculates the Sigmoid function.
        :param in_z: The variable for the Sigmoid function.
        :return: The Sigmoid(z).
        """
        return 1.0 / (1 + np.exp(-in_z))

    def __cost_function(self, class_of_weight, weight):
        """
        Calculates the cost function.
        :param class_of_weight: The given class.
        :param weight: The given weight.
        :return:
        """
        if class_of_weight == 1:
            return -(np.log(self.__sigmoid(weight)))
        elif class_of_weight == 0:
            return -(1 - self.__sigmoid(weight))

    # TODO: Make a line search function.

    def gradient_slope_classify(self, ascent=True, max_cycles=500, alpha=0.001):
        """
        Calculate the gradient slope.
        :param ascent: Set to False to calculate descent instead of ascent.
        :param max_cycles: Maximum cycles you wish to calculate.
        :return: The logistic regression weights.
        """
        sign = 1
        if not ascent:
            sign = -1
        shape_of_water = self.__superSet.shape
        weights = np.ones((shape_of_water[1], 1))
        for i in range(max_cycles):
            h = self.__sigmoid(self.__superSet.T * weights)
            error = (self.__classLabels.T - h)
            weights = weights + sign * (alpha * self.__superSet.T * error)
        cost = 0
        for my_class, my_weight in zip(self.__classLabels, weights):
            cost += self.__cost_function(my_class, my_weight)

        return weights * cost
