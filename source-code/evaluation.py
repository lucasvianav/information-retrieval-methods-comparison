import math

from util import get_intersection


class Evaluation:
    """
    A class to performs various evaluation methods for a single query given to
    a IR method.

    Parameters:
        returned_set (list): set of ranked documents returned by the IR method.
        truth_set (list): set of actually relevant documents for that same
                          query.
    """

    def __init__(self, returned_set: list, truth_set: list):
        self.returned_set = returned_set
        self.truth_set = truth_set

        self.__dcg = None
        self.__idcg = None
        self.__precision = None
        self.__recall = None

    def dcg(self) -> tuple:
        """
        Performs the DCG (discounted accumulated gain) evaluation for this
        query and returns the DCG vector as well as the IDCG (the DCG's ideal
        version).

        In order to use the DCG to compare multiple IR models, you must
        perform the DCG for many queries and then use the quocient value
        between the mean DCG and the mean IDCG.

        In order to better compare the two models, you may plot this results
        with the document list as the x-axis.

        Return value:
            tuple: the first element is the DCG and the second is the IDCG.
        """

        if self.__dcg is None and self.__idcg is None:
            gain = [ 1. if doc in self.truth_set else 0. for doc in self.returned_set ]
            ideal_gain = sorted(gain, reverse=True)

            dcg = [ gain[0] ]
            for i in range(1, len(gain)):
                dcg.append(gain[i]/math.log2(i+1) + dcg[i-1])

            idcg = [ ideal_gain[0] ]
            for i in range(1, len(ideal_gain)):
                idcg.append(ideal_gain[i]/math.log2(i+1) + idcg[i-1])

            self.__dcg = dcg
            self.__idcg = idcg

        return self.__dcg, self.__idcg

    def recall(self) -> float:
        """
        Performs the recall evaluation for this query.

        Return value:
            float: this query's total recall value.
        """

        if self.__recall is None:
            intersection = get_intersection(self.returned_set, self.truth_set)
            self.__recall = len(intersection)/len(self.truth_set)

        return self.__recall

    def precision(self) -> float:
        """
        Performs the precision evaluation for this query.

        Return value:
            float: this query's total precision value.
        """

        if self.__precision is None:
            intersection = get_intersection(self.returned_set, self.truth_set)
            self.__precision = len(intersection)/len(self.returned_set)

        return self.__precision
