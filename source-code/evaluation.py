import math
from functools import reduce

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
        self.intersection = get_intersection(returned_set, truth_set)

        self.__dcg = None
        self.__idcg = None
        self.__precision = len(self.intersection)/len(returned_set)
        self.__recall = len(self.intersection)/len(truth_set)

    def getDCG(self) -> tuple:
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

    def getRecall(self) -> float:
        """
        Performs the recall evaluation for this query.

        Return value:
            float: this query's total recall value.
        """

        return self.__recall

    def getPrecision(self) -> float:
        """
        Performs the precision evaluation for this query.

        Return value:
            float: this query's total precision value.
        """

        return self.__precision

    def __precisionAtN(self, N: int) -> float:
        """
        Performs the precision evaluation for this query.

        Parameters:
            N (int): last element do be considered's index.

        Return value:
            float: parcial precision at N.
        """

        intersection = get_intersection(self.returned_set[:(N+1)], self.truth_set)
        return len(intersection)/(N+1)

    def __recallAtN(self, N: int) -> float:
        """
        Performs the recall evaluation for this query.

        Parameters:
            N (int): last element do be considered's index.

        Return value:
            float: parcial recall at N.
        """

        intersection = get_intersection(self.returned_set[:(N+1)],
                                        self.truth_set)
        return len(intersection)/len(self.truth_set)

    def getPrecisionRecallInterpol(self) -> dict:
        """
        Calculates the 11-points precision x recall interpolations values.

        Return value:
            dict: contains the 'precision' and 'recall' keys with their
                  respective points.
        """

        return_value = {
            'precision': [],
            'recall': [ i/10 for i in range(11) ]
        }

        # raw precision x recall
        # values (not interpolated)
        raw = { 'precision': [], 'recall': [] }

        for relevant_doc in self.intersection:
            i = self.returned_set.index(relevant_doc)
            raw['precision'].append(self.__precisionAtN(i))
            raw['recall'].append(self.__recallAtN(i))

        for recall in return_value['recall']:
            filtered = [ i for i, rec in enumerate(raw['recall']) if rec >=
                        recall ]
            return_value['precision'].append(max([ raw['precision'][i] for i in filtered ]))

        return return_value

    def getMAP(self) -> float:
        """
        Calculates the 11-points precision x recall interpolations values.

        Return value:
            dict: contains the 'precision' and 'recall' keys with their
            respective points.
        """

        precision = reduce(lambda acc, cur: acc + self.__precisionAtN(cur),
                           self.truth_set, 0.)

        return precision/len(self.truth_set)

