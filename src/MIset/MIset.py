import pandas as pd
import numpy as np
from joblib import Parallel,delayed
import warnings
from collections import Counter


class MIset:
    """This is the class representation of this library. Invoke the methods of this class to perform feature selection on your dataset.

    :param max_features: Choose maximum count of important features to be given by the feature selection method, defaults to 1.
    :type max_features: int

    :param variant: Choose which feature selection method must be used. The following options are available:
        
        * **'jmim'** : **'Joint Mutual Information Maximization'** method as described in this `paper <https://doi.org/10.1016/j.eswa.2015.07.007>`_.
        * **'njmim'** : **'Normalized Joint Mutual Information Maximization'** method as described in this `paper <https://doi.org/10.1016/j.eswa.2015.07.007>`_.
        * **'jomic'** : **'Joint Mutual Information with Class Relevance'** method as described in this `paper <https://doi.org/10.1016/j.jcmds.2023.100075>`_.
    :type variant: str

    :param verbose: Choose whether to print messages to show feature selection progress. A message is printed once every most relevant feature is found, parameter defaults to False.
    :type verbose: bool, optional

    :param n_jobs: The number of jobs to use while computing  the feature selection method. Passing -1 means using all processors. Parallelization is done via 'joblib'.
    :type n_jobs: int, optional

    """
    class _entropy_calc:
        """This is a private nested class under class MIset. This houses methods used for entropy and mutual information calculation.

        """
        @staticmethod        
        def marginalEntropy(arr):
            """Calculates the marginal entropy (H(X)) of a variable X

            :param arr: Values of a variable
            :type arr: pandas series

            :return: Marginal Entropy of a variable X
            :rtype: float
            """
            # Count the number of elements in the array
            total_elements=len(arr)
        
            # Make a dictionary where the key is the element and the value is its frequency
            cat_dict=dict(Counter(arr))
        
            # Calculate entropy for a single variable
            return np.round(-1*sum([(value/total_elements)*np.log2(value/total_elements) for value in cat_dict.values()]),8)
        
        
        @staticmethod
        def jointEntropy(x_arr,y_arr):
            """Calculates the joint entropy (H(X,Y)) given both variables X and Y

            :param x_arr: Values of a variable X
            :type x_arr: pandas series

            :param y_arr: Values of a variable Y
            :type y_arr: pandas series
            
            :return: Joint Entropy of a variable X and Y
            :rtype: float
            """            
            # Count the number of elements in the array
            total_elements=len(x_arr)
        
            # Since joint entropy is being calculated, both the variables must be paired first
            cat_dict=dict(Counter(list(zip(x_arr,y_arr)))) 
        
            # Calculate joint entropy
            # This is marked as H(X,Y)
            return np.round(-1*sum([(value/total_elements)*np.log2(value/total_elements) for value in cat_dict.values()]),8)
        

        @staticmethod
        def conditionalEntropy(x_arr,y_arr):
            """Calculates the joint entropy (H(Y|X)) given both variables X and Y

            :param x_arr: Values of a variable X
            :type x_arr: pandas series

            :param y_arr: Values of a variable Y
            :type y_arr: pandas series
            
            :return: Conditional Entropy of variables X and Y
            :rtype: float
            """            
            # This calculates H(Y|X)
            # So H(X|Y) = H(X,Y)- H(Y)
            return MIset._entropy_calc.jointEntropy(x_arr,y_arr)-MIset._entropy_calc.marginalEntropy(y_arr)
        

        @staticmethod
        def tripleJointEntropy(x_arr,y_arr,c_arr):
            """Calculates the joint entropy (H(X,Y,Z)) of three variables X,Y and C

            :param x_arr: Values of a variable X
            :type x_arr: pandas series

            :param y_arr: Values of a variable Y
            :type y_arr: pandas series

            :param c_arr: Values of a variable C
            :type c_arr: pandas series
            
            :return: Joint Entropy of a variables X,Y and C
            :rtype: float
            """            
            # Count the number of elements in the array
            total_elements=len(x_arr)
        
            # Since joint entropy is being calculated, both the variables must be paired first
            cat_dict=dict(Counter(list(zip(x_arr,y_arr,c_arr)))) 
        
            # Calculate triple joint entropy
            # This is marked as H(X,Y,C)
            return np.round(-1*sum([(value/total_elements)*np.log2(value/total_elements) for value in cat_dict.values()]),8)
        

        @staticmethod
        def mutualInformationScore(x_arr,y_arr):
            """Calculates the mutual information (I(X;Y)) of variables X and Y

            :param x_arr: Values of a variable X
            :type x_arr: pandas series

            :param y_arr: Values of a variable Y
            :type y_arr: pandas series
            
            :return: Mutual Infroamtion score of a variable
            :rtype: float
            """            
            # I(X;Y) = H(X) - H(X|Y)
            # OR
            # I(X;Y) = H(X) + H(Y) - H(X,Y)
            return MIset._entropy_calc.marginalEntropy(x_arr) + MIset._entropy_calc.marginalEntropy(y_arr) - MIset._entropy_calc.jointEntropy(x_arr,y_arr)
        

        @staticmethod
        def jointMutualInformationScore(x_arr,y_arr,c_arr):
            """Calculates the joint mutual information (I(X,Y;C)) of variables X, Y and C

            :param x_arr: Values of a variable X
            :type x_arr: pandas series

            :param y_arr: Values of a variable Y
            :type y_arr: pandas series

            :param c_arr: Values of a variable C
            :type c_arr: pandas series
            
            :return: Conditional Entropy of a variable
            :rtype: float
            """            
            # I(X;C|Y) = H(X|Y) - H(X|C,Y)
            #          = H(X,Y) - H(Y) - (H(X,Y,C)-H(C,Y))
            mi_score_conditional = MIset._entropy_calc.jointEntropy(x_arr,y_arr) - MIset._entropy_calc.marginalEntropy(y_arr) - (MIset._entropy_calc.tripleJointEntropy(x_arr,y_arr,c_arr) - MIset._entropy_calc.jointEntropy(c_arr,y_arr))
            
            # I(X,Y;C) = I(X;C|Y) + I(Y;C)
            return mi_score_conditional + MIset._entropy_calc.mutualInformationScore(y_arr,c_arr)
        

        @staticmethod
        def interactionInformation(x_arr,y_arr,c_arr):
            """Calculates the interaction information (I(X;Y;C)) of variables X, Y and C

            :param x_arr: Values of a variable X
            :type x_arr: pandas series

            :param y_arr: Values of a variable Y
            :type y_arr: pandas series

            :param c_arr: Values of a variable C
            :type c_arr: pandas series
                        
            :return: Interaction Information of variables X,Y and C
            :rtype: float
            """            
            # I(X;Y;C) = I(X,Y;C) - I(X;C) - I(Y;C)
            return MIset._entropy_calc.jointMutualInformationScore(x_arr,y_arr,c_arr) - MIset._entropy_calc.mutualInformationScore(x_arr,c_arr) - MIset._entropy_calc.mutualInformationScore(y_arr,c_arr)



    class _core_scores:
        """This is a private nested class under class MIset. This houses methods that calculate scoring for each feature.

        """
        @staticmethod
        def computeP1InnerLoopScores(variant,x_arr,y_arr,c_arr):
            """Calculates the 'Joint Mutual Information Maximization' score or the 'Normalized Mutual Information Maximization' score of a feature

            :param variant: Feature selection algorithm to be selected according to the variant
            :type variant: str

            :param x_arr: Values of a variable X
            :type x_arr: pandas series

            :param y_arr: Values of a variable Y
            :type y_arr: pandas series

            :param c_arr: Values of a variable C
            :type c_arr: pandas series
                        
            :return: Feature selection score
            :rtype: float
            """
            # Calculate joint MI of candidate feature with the selected feature
            if variant=='jmim':
                variant_score=MIset._entropy_calc.jointMutualInformationScore(x_arr,y_arr,c_arr)
                
            elif variant=='njmim':
                # Calculate symmetric relevence
                variant_score=np.round(MIset._entropy_calc.jointMutualInformationScore(x_arr,y_arr,c_arr)/MIset._entropy_calc.tripleJointEntropy(x_arr,y_arr,c_arr),8)
        
            else:
                # This block will never get triggered
                variant_score=-1
        
            return variant_score
        

        @staticmethod
        def computeP2InnerLoopScores(x_arr,y_arr,c_arr):
            """Calculates the 'Joint Mutual Information with Class Relevance' score of a feature

            :param x_arr: Values of a variable X
            :type x_arr: pandas series

            :param y_arr: Values of a variable Y
            :type y_arr: pandas series

            :param c_arr: Values of a variable C
            :type c_arr: pandas series
                        
            :return: Feature selection score
            :rtype: float
            """
            # Compute mutual information
            mi_score=MIset._entropy_calc.mutualInformationScore(x_arr,y_arr)
            jmi_score=MIset._entropy_calc.jointMutualInformationScore(y_arr,x_arr,c_arr)
        
            return mi_score,jmi_score


    class _misc:
        """This is a private nested class under class MIset. This houses some miscellaneous methods.

        """
        @staticmethod
        def uniqueArrayIdentifier(col_name,arr):
            return col_name if arr.nunique()==1 else None

    
    def __init__(self,max_features=1,variant='jmim',verbose=False,n_jobs=None):
        """Constructor of the class MIset

        """
        self.max_features=max_features
        self.variant=variant
        self.verbose=verbose
        self.n_jobs=n_jobs


    def _paper1FS(self,df,feature_list,class_feature_name,feature_index_dict):
        """Private method which implements the 'Joint Mutual Information Maximization' score or the 'Normalized Mutual Information Maximization' feature selection method

        :param df: Dataframe containing the data
        :type df: Pandas DataFrame

        :param feature_list: List of column names of the DataFrame on which the feature selection algorithm is to be run
        :type feature_list: list

        :param class_feature_name: The column name containing the target variable
        :type class_feature_name: str

        :param feature_index_dict: A dcitionary where the key is the column name and the value is the index number of the column in the DataFrame
        :type feature_index_dict: dict
                    
        :return: Returns None
        :rtype: void
        """
        # Create an empty list of all selected features
        selected_feature_list=[]
    
        # Create an empty dict of all selected features, and their corresponding Joint MI score
        selected_feature_score_dict={}
        
        # Get array of class variable
        c_arr=df.iloc[:,feature_index_dict[class_feature_name]]

        # First iteration dict
        first_iteration_dict={}
    
        # First iteration
        # Select the best feature with the maximum MI score
        for candidate_feature in feature_list:
            # Get array of candidate feature
            x_arr=df.iloc[:,feature_index_dict[candidate_feature]]
            
            first_iteration_dict.update({candidate_feature:MIset._entropy_calc.mutualInformationScore(x_arr,c_arr)})
    
        # Get column name with maximum mutual information
        first_iteration_max_mi_feature_name = max(first_iteration_dict, key=first_iteration_dict.get)
    
        # Append the first iteration of feature in the list
        selected_feature_list.append(first_iteration_max_mi_feature_name)

        # Update the score dictionary
        selected_feature_score_dict.update({first_iteration_max_mi_feature_name:first_iteration_dict[first_iteration_max_mi_feature_name]})

        # Remove the first feature from the candidate feature list
        feature_list.remove(first_iteration_max_mi_feature_name)

        if self.verbose:
            print(f"No.{len(selected_feature_list)} feature, '{first_iteration_max_mi_feature_name}' added.")

        # No need to run second part of the algorithm if max features is given as 1
        if self.max_features==1:
            pass
        else:
            # Select subsequent features
            # -1 is added as one feature has already been added above
            while len(selected_feature_list)<=self.max_features-1:

                # Break out of the loop if the entire candidate feature set is exhausted
                if len(feature_list)==0:
                    break
                else:
                    # Initialize dictionary to hold all features which have the minimum MI score from redundency calculation of selected dict
                    relevancy_dict={}
                    
                    for candidate_feature in feature_list:
                        # Get array of candidate feature
                        x_arr=df.iloc[:,feature_index_dict[candidate_feature]]

                        # This list will store the joint mutual information score (in the case of 'jmim' variant)
                        # Or symmetric relevence (in the case of 'njmim' variant)
                        candidate_feature_selected_feature_set_score_list=[]

                        # Implement joblib module to parallelize this process
                        candidate_feature_selected_feature_set_score_list = Parallel(n_jobs=self.n_jobs)(
                                                                                delayed(MIset._core_scores.computeP1InnerLoopScores)(
                                                                                    self.variant, 
                                                                                    x_arr, 
                                                                                    df.iloc[:,feature_index_dict[selected_feature]], # This is y_arr
                                                                                    c_arr
                                                                                )
                                                                                for selected_feature in selected_feature_list
                                                                            )
                        
                        # Store the minimum joint mutual information / symmetric relevence score of a candidate feature against the entire selected feature set
                        relevancy_dict.update({candidate_feature:min(candidate_feature_selected_feature_set_score_list)})
                            
                    # Take the max MI score in relevancy dictionary
                    # Implement the maximum of the minimum approach
                    max_relevancy_feature_name=max(relevancy_dict, key=relevancy_dict.get)
        
                    # Append the feature into the selected feature list
                    selected_feature_list.append(max_relevancy_feature_name)
        
                    # Update the score dictionary
                    selected_feature_score_dict.update({max_relevancy_feature_name:relevancy_dict[max_relevancy_feature_name]})
        
                    # Remove the selected feature from the candidate feature base
                    feature_list.remove(max_relevancy_feature_name)
        
                    # Use verbosity parameter
                    if self.verbose:
                        print(f"No.{len(selected_feature_list)} feature, '{max_relevancy_feature_name}' added.")
    
        # Initialize instance variables
        self.selected_feature_list=selected_feature_list
        self.selected_feature_score_dict=selected_feature_score_dict

        return


    def _paper2FS(self,df,feature_list,class_feature_name,feature_index_dict):
        """Private method which implements the 'Joint Mutual Information with Class Relevance' feature selection method

        :param df: DataFrame containing the data
        :type df: Pandas DataFrame

        :param feature_list: List of column names of the DataFrame on which the feature selection algorithm is to be run
        :type feature_list: list

        :param class_feature_name: The column name containing the target variable
        :type class_feature_name: str

        :param feature_index_dict: A dcitionary where the key is the column name and the value is the index number of the column in the DataFrame
        :type feature_index_dict: dict
                    
        :return: Returns None
        :rtype: void
        """
        # Create an empty list of all selected features
        selected_feature_list=[]
    
        # Create an empty dict of all selected features, and their corresponding relevancy score
        selected_feature_score_dict={}

        # Get array of class variable
        c_arr=df.iloc[:,feature_index_dict[class_feature_name]]

        # First iteration dict
        first_iteration_dict={}
    
        # First iteration
        for candidate_feature in feature_list:
            # Get array of candidate feature
            x_arr=df.iloc[:,feature_index_dict[candidate_feature]]
            
            first_iteration_dict.update({candidate_feature:MIset._entropy_calc.mutualInformationScore(x_arr,c_arr)})
    
        # Get column name with maximum mutual information
        first_iteration_max_mi_feature_name = max(first_iteration_dict, key=first_iteration_dict.get)
    
        # Append the first iteration of feature in the list
        selected_feature_list.append(first_iteration_max_mi_feature_name)

        # Update the score dictionary
        selected_feature_score_dict.update({first_iteration_max_mi_feature_name:first_iteration_dict[first_iteration_max_mi_feature_name]})

        # Remove the first feature from the candidate feature list
        feature_list.remove(first_iteration_max_mi_feature_name)

        if self.verbose:
            print(f"No.{len(selected_feature_list)} feature, '{first_iteration_max_mi_feature_name}' added.")
        
        # No need to run second part of the algorithm if max features is given as 1
        if self.max_features==1:
            pass
        else:
            # Select subsequent features
            # -1 is added as one feature has already been added above
            while len(selected_feature_list)<=(self.max_features-1):
                # Break out of the loop if the entire candidate feature set is exhausted
                if len(feature_list)==0:
                    break
                else:
                    # Initialize dictionary to hold all features which have the minimum MI score from redundency calculation of selected dict
            
                    relevant_score_dict={}
                    
                    for candidate_feature in feature_list:
                        # Get array of candidate feature
                        x_arr=df.iloc[:,feature_index_dict[candidate_feature]]

                        # Initialize variables
                        sum_jmi=0
                        sum_mi=0

                        # Implement joblib module to parallelize this process
                        # Output of this operation will be a list of tuples, with each tuple containing two elements
                        # The first element of the tuple is the mutual information score
                        # The second element of the tuple is the joint mutual information score
                        score_results = Parallel(n_jobs=self.n_jobs)(
                            delayed(MIset._core_scores.computeP2InnerLoopScores)(
                                x_arr,
                                df.iloc[:,feature_index_dict[selected_feature]], # This is y_arr
                                c_arr
                            )
                            for selected_feature in selected_feature_list
                        )
                        
                        sum_mi = sum(score_tuple[0] for score_tuple in score_results)
                        sum_jmi = sum(score_tuple[1] for score_tuple in score_results)
                            
                        # Compute relevant score
                        # relevant_score = average(jmi) - average(mi)
                        relevant_score=(sum_jmi/len(selected_feature_list))-(sum_mi/len(selected_feature_list))
        
                        # Update the relevant score dictionary
                        relevant_score_dict.update({candidate_feature:relevant_score})
        
                    # Take the max relevancy score in dictionary
                    max_relevancy_feature_name=max(relevant_score_dict, key=relevant_score_dict.get)
        
                    # Append the feature into the selected feature list
                    selected_feature_list.append(max_relevancy_feature_name)
        
                    # Update the score dictionary
                    selected_feature_score_dict.update({max_relevancy_feature_name:relevant_score_dict[max_relevancy_feature_name]})
        
                    # Remove the selected feature from the candidate feature base
                    feature_list.remove(max_relevancy_feature_name)
        
                    # Use verbosity parameter
                    if self.verbose:
                        print(f"No.{len(selected_feature_list)} feature, '{max_relevancy_feature_name}' added.")

        # Initialize instance variables
        self.selected_feature_list=selected_feature_list
        self.selected_feature_score_dict=selected_feature_score_dict
               
        return
        
    
    def fit(self,df,feature_list,class_feature_name):
        """Fit the feature selection algorithm on your dataset.
        
        :param df: Pandas DataFrame
        :type df: Pandas DataFrame
        
        :param feature_list: List of column names of the DataFrame on which feature selection is to be performed.
        :type feature_list: list
        
        :param class_feature_name: Name of the column containing your target variable.
        :type class_feature_name: str

        :return: Returns None
        :rtype: None

        """
        if not isinstance(self.max_features,int):
            raise ValueError("Invalid value for parameter 'max_features'. This parameter only accepts a value of 'integer' datatype.")

        if self.max_features<1:
            raise ValueError("Invalid value for parameter 'max_features'. Value must be >= 1.")

        if not isinstance(self.variant,str):
            raise ValueError("Invalid value for parameter 'variant'. This parameter only accepts a value of 'string' datatype.")

        if self.variant not in ['jmim','njmim','jomic']:
            raise ValueError("Invalid value for parameter 'variant'. Supported parameters are 'jmim','njmim','jomic'")

        if not isinstance(self.verbose,bool):
            raise ValueError("Invalid value for parameter 'verbose'. This parameter only accepts a value of 'boolean' datatype.")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Invalid value for parameter 'df'. This parameter only accepts a value of 'Pandas DataFrame' datatype.")
        
        if not isinstance(feature_list,list):
            raise ValueError("Invalid value for parameter 'feature_list'. This parameter only accepts a value of 'list' datatype.")
            
        if not isinstance(class_feature_name,str):
            raise ValueError("Invalid value for parameter 'class_feature_name'. This parameter only accepts a value of 'string' datatype.")

        if len(feature_list)==0:
            raise ValueError("Parameter 'feature_list' is empty. At least one feature name must be present.")

        if class_feature_name not in list(df.columns):
            raise ValueError("Class variable is absent in the DataFrame.")

        if class_feature_name in feature_list:
            raise ValueError("Class feature name must not be present in the candidate feature list.")

        if df[class_feature_name].nunique()!=2:
            raise ValueError("Class variable does not have two unique classes.")

        if df[feature_list+[class_feature_name]].isnull().values.any():
            raise ValueError("At least one null value is present in the DataFrame.")
        
        # Create copies of input parameters
        df=df.copy()
        feature_list=list(feature_list)
        class_feature_name=str(class_feature_name)


        # Create a mapping of column name against DataFrame index
        feature_index_dict = {col: df.columns.get_loc(col) for col in (feature_list+[class_feature_name])}

        # Identify features with only one unique value throughout
        redundent_feature_list = Parallel(n_jobs=self.n_jobs)(delayed(MIset._misc.uniqueArrayIdentifier)(col,df.iloc[:,feature_index_dict[col]]) for col in feature_list)

        # Remove elements with None type value
        redundent_feature_list = list(filter(lambda x: x is not None, redundent_feature_list))

        if len(redundent_feature_list)>0:
            warnings.warn(f"Features identified having only one unique value throughout. These features are removed from candidate feature list. These features are : {','.join(redundent_feature_list)}", UserWarning)

            feature_list=[col for col in feature_list if col not in redundent_feature_list]

            if len(feature_list)==0:
                raise ValueError("Parameter 'feature_list' is empty after removing all features with only one unique value. Features with more than one unique value must be present in 'feature_list'.")

        # Determine which feature selection method to choose
        match self.variant:
            case 'jmim':
                self._paper1FS(df,feature_list,class_feature_name,feature_index_dict)

            case 'njmim':
                self._paper1FS(df,feature_list,class_feature_name,feature_index_dict)

            case 'jomic':
                self._paper2FS(df,feature_list,class_feature_name,feature_index_dict)

            case _:
                # Will never get triggered
                pass
            
        return


    def top_features(self):
        """Get a list of feature names deemed the most important by the feature selection algorithm.

        Each entry in the list represents the most important feature selected during that iteration. For example, the first index of the list is the most important feature in the first iteration, the second index of the list is the most important feature in the second iteration and so on.

        :return: Returns the list of most important features.
        :rtype: list[str]

        """
        return self.selected_feature_list

    
    def feature_scores(self):
        """Get a dictionary where the key is the feature name and the value is its feature importance score as computed by your selected algorithm.

        :return: Returns a dictionary of feature scores.
        :rtype: dict

        """
        return self.selected_feature_score_dict

    
    def feature_selection_order(self):
        """Get a dictionary which provides information on which feature was deemed as most important at each iteration.
        
        The key of the dictionary is the iteration number while the value of the dictionary is the most important feature according to the feature selection algorithm in that iteration.

        :return: Returns a dictionary of most important feature at each iteration order.
        :rtype: dict

        """
        order_dict={}

        counter=1
        # Key is the order in which the feature was encountered, value is feature name
        for key,value in self.selected_feature_score_dict.items():
            order_dict.update({counter:key})
            counter+=1

        return order_dict
