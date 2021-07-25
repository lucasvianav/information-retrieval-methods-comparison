import math
from functools import reduce
from typing import List, Tuple

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
        self.__returned_set = returned_set
        self.__truth_set    = truth_set
        self.__intersection = get_intersection(returned_set, truth_set)

        self.__dcg        = []
        self.__idcg       = []
        self.__precision  = len(self.__intersection)/len(returned_set)
        self.__recall     = len(self.__intersection)/len(truth_set)

    def getDCG(self, length: int = 15) -> Tuple[List[float], List[float]]:
        """
        Performs the DCG (discounted accumulated gain) evaluation for this
        query and returns the DCG vector as well as the IDCG (the DCG's ideal
        version).

        In order to use the DCG to compare multiple IR models, you must
        perform the DCG for many queries and then use the quocient value
        between the mean DCG and the mean IDCG.

        In order to better compare the two models, you may plot this results
        with the document list as the x-axis.

        Parameters:
            length (int, default 15): the ranking's length to consider.

        Return value:
            tuple<list<float>>>: the first element is the DCG and the second is
                                 the IDCG.
        """

        if not self.__dcg and not self.__idcg:
            gain = [
                1. if doc in self.__truth_set else 0.
                for doc in self.__returned_set[:length]
            ]
            ideal_gain = sorted(gain, reverse=True)

            dcg = [ gain[0] ]
            for i in range(1, length):
                dcg.append(gain[i]/math.log2(i+1) + dcg[i-1])

            idcg = [ ideal_gain[0] ]
            for i in range(1, length):
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

        intersection = get_intersection(self.__returned_set[:(N+1)], self.__truth_set)
        return len(intersection)/(N+1)

    def __recallAtN(self, N: int) -> float:
        """
        Performs the recall evaluation for this query.

        Parameters:
            N (int): last element do be considered's index.

        Return value:
            float: parcial recall at N.
        """

        intersection = get_intersection(self.__returned_set[:(N+1)],
                                        self.__truth_set)
        return len(intersection)/len(self.__truth_set)

    def getInterpol(self) -> Tuple[List[float], List[float]]:
        """
        Calculates the 11-points precision x recall interpolations values.

        Return value:
            tuple<list<float>>>: the first element is the precision points and
                                 the second is the recall's ones.
        """

        # first element is precision, second is recall
        return_value = ( [], [ i/10 for i in range(11) ] )

        # raw precision x recall
        # values (not interpolated)
        raw = { 'precision': [], 'recall': [] }

        for relevant_doc in self.__intersection:
            i = self.__returned_set.index(relevant_doc)
            raw['precision'].append(self.__precisionAtN(i))
            raw['recall'].append(self.__recallAtN(i))

        for recall in return_value[1]:
            filtered = [
                i for i, rec in enumerate(raw['recall'])
                if rec >= recall
            ]
            return_value[0].append(
                max([raw['precision'][i] for i in filtered])
                if filtered else 0.
            )

        return return_value

    def getMAP(self) -> float:
        """
        Calculates the 11-points precision x recall interpolations values.

        Return value:
            dict: contains the 'precision' and 'recall' keys with their
            respective points.
        """

        # reduce function
        def reduceFn(acc, cur):
            return acc + self.__precisionAtN(self.__returned_set.index(cur))

        precision = reduce(reduceFn, self.__intersection, 0.0)

        return precision/len(self.__truth_set)

