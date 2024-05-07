import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split


def is_invertible(matrix: np.ndarray, threshold: float = 1e-10) -> bool:
    """
    Checks whether a given matrix is invertible.
    Args:
    - matrix: The input matrix to be checked for invertibility.
    - threshold: Threshold value for considering the determinant of the matrix to be zero.
    Returns:
    - Boolean: True if the matrix is invertible, False otherwise.
    """
    return abs(scipy.linalg.det(matrix)) > threshold


class LinearRegression:

    def __init__(self):
        """
        Initializes the LinearRegression object with attributes for weights and fitting progress.
        """
        self._weights = None
        self.fit_progress_completed = False
        self.X_train = None
        self.y_train = None

    def fit(self, X, y) -> None:
        """
        Fits the linear regression model to the given training data.
        Args:
        - X: Input features.
        - y: Target variable.
        Raises:
        - RuntimeError: If XtX is a singular matrix.
        """
        self.X_train = pd.DataFrame(X)
        self.y_train = pd.Series(y)
        y_true_labels = np.array(y)
        feature_matrix = np.array(X)
        # Add 1 to every feature vector
        feature_matrix = np.concatenate((feature_matrix, np.ones((feature_matrix.shape[0], 1))), axis=1)
        self._weights = np.zeros(len(feature_matrix[0]))

        feature_matrix_transpose = feature_matrix.transpose()
        feature_matrix_squared = np.matmul(feature_matrix_transpose, feature_matrix)
        if is_invertible(feature_matrix_squared):
            feature_matrix_squared_inverse = np.linalg.inv(feature_matrix_squared)
            squared_inverse_mul_transpose = np.matmul(feature_matrix_squared_inverse, feature_matrix_transpose)
            self._weights = np.matmul(squared_inverse_mul_transpose, y_true_labels)
            self.fit_progress_completed = True
        else:
            self.fit_progress_completed = False
            raise RuntimeError("XtX is a singular matrix. a unique minimal (optimal) solution does not exist")

    def predict(self, X) -> np.ndarray:
        """
        Predicts the target variable for the given input features.
        Args:
        - X: Input features.
        Returns:
        - numpy array: Predicted target variable.
        Raises:
        - RuntimeError: If model has not been fitted yet.
        """
        if self.fit_progress_completed:
            feature_matrix = pd.DataFrame(X)
            feature_matrix["constant"] = 1
            predictions = feature_matrix.apply(lambda feature_vector: np.inner(list(feature_vector), self._weights),
                                               axis=1)
            return predictions
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def score(self, X, y) -> float:
        """
        Computes the R^2 score of the model.
        Args:
        - X: Input features.
        - y: True labels.
        Returns:
        - float: R^2 score.
        """
        if self.fit_progress_completed:
            sum_of_squares_error = self.calculate_sum_of_squared_errors(X, y)
            sum_of_squares_total = self.calculate_sum_of_squares_deviation_from_true_mean(y)

            return 1 - (sum_of_squares_error / sum_of_squares_total)
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def calculate_sum_of_squared_errors(self, X, y) -> float:
        """
        Calculates the sum of squared errors for the model predictions.
        Args:
        - X: Input features.
        - y: True labels.
        Returns:
        - float: Sum of squared errors.
        """
        y_true_labels = np.array(y)
        y_predictions = np.array(self.predict(X))

        return ((y_true_labels - y_predictions) ** 2).sum()

    def calculate_sum_of_squares_deviation_from_true_mean(self, y) -> float:
        """
        Calculates the total sum of squares deviation from the true mean of the target variable.
        Args:
        - y: True labels.
        Returns:
        - float: Total sum of squares deviation.
        """
        true_labels = np.array(y)
        return ((true_labels - true_labels.mean()) ** 2).sum()

    def fit_using_train_and_print_test_score(self, feature_matrix_train, feature_matrix_test,
                                             true_labels_train, true_labels_test) -> None:
        """
        Fits the model using training data, calculates the R^2 score using test data, and prints the results.
        Args:
            feature_matrix_train (array-like): Feature matrix for training the model.
            feature_matrix_test (array-like): Feature matrix for testing the model.
            true_labels_train (array-like): True labels corresponding to the training data.
             true_labels_test (array-like): True labels corresponding to the test data.
        Returns:
             None
        """
        self.fit(feature_matrix_train, true_labels_train)
        r_squared = self.score(feature_matrix_test, true_labels_test)
        print(f"The model achieved a R^2 of {r_squared} "
              f"and found weights vector of {self._weights}")

    def summary(self, X_test = None, y_test = None):
        """
        Prints the summary of the fitted linear regression model.
        Args:
            - model: Fitted LinearRegression object.
            - X_column_names: List of column names of the input features.
        """
        self.X_train = pd.DataFrame(self.X_train)
        self.y_train = pd.Series(self.y_train)
        X_test = pd.DataFrame(X_test)
        y_test = pd.Series(y_test)

        print("==============================================================================")
        print(f"Dep. Variable:                {self.y_train.name}")
        print("Loss Function:                MSE")
        if X_test is not None and y_test is not None:
            print(f"R-squared:                    {self.score(X_test, y_test):.3f}")
        else:
            print(f"R-squared:                    {self.score(self.X_train, self.y_train):.3f}")
        # print(f"Adj. R-squared:               {model.score:.3f}")
        print(f"No. Observations for train:            {len(self.X_train)}")
        if X_test is not None and y_test is not None:
            print(f"No. Observations for test:             {len(X_test)}")
        print("==============================================================================")
        print("Weights Used For Training: ")
        print("{:<15} {:>10}".format(" ", "coef"))
        X_column_names = self.X_train.columns.tolist()
        for i in range(len(self._weights)):
            print("{:<15} {:10.4f}".format(
                X_column_names[i] if i < len(X_column_names) else "const",
                self._weights[i]
            ))
        print("==============================================================================")


def main3() -> None:
    """
    Answer for question 3
    """
    data_set = pd.read_csv("simple_regression.csv")
    true_labels = data_set["y"]
    feature_matrix = data_set.drop(columns=["y"])

    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(feature_matrix, true_labels, test_size=0.2, shuffle=True))

    linear_regression_model = LinearRegression()
    linear_regression_model.fit(feature_matrix_train, true_labels_train)
    linear_regression_model.summary(feature_matrix_test, true_labels_test)
    # linear_regression_model.fit_using_train_and_print_test_score(feature_matrix_train, feature_matrix_test,
                                                                   # true_labels_train, true_labels_test)


def main4() -> None:
    """
    Answer for question 4
    """
    feature_matrix, true_labels = fetch_california_housing(return_X_y=True)

    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(feature_matrix, true_labels, test_size=0.2, shuffle=True))

    linear_regression_model = LinearRegression()
    linear_regression_model.fit_using_train_and_print_test_score(feature_matrix_train, feature_matrix_test,
                                                                   true_labels_train, true_labels_test)


def get_feature_matrix_with_all_polynomial_combinations(feature_matrix,
                                                        polynomial_degree: int) -> np.ndarray:
    """
    Generates polynomial features up to a specified degree for the given feature matrix.
    Args:
    - feature_matrix: Input feature matrix.
    - polynomial_degree: Degree of the polynomial features to be generated.
    Returns:
    - numpy array: Polynomial feature matrix.
    """
    poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
    return poly.fit_transform(feature_matrix)


def get_polynomial_degrees_errors(feature_matrix, true_labels, lowest_degree: int = 1,
                                  highest_degree: int = 8) -> tuple[list, list]:
    """
    Computes the sum of squared errors for polynomial regression models of different degrees.
    Args:
    - feature_matrix: Input features.
    - true_labels: True labels.
    - lowest_degree: The lowest polynomial degree to consider.
    - highest_degree: The highest polynomial degree to consider.
    Returns:
    - tuple: Arrays of polynomial degrees and corresponding errors.
    """
    polynomial_regression_model = LinearRegression()
    polynomial_degrees_errors = list()

    polynomial_degrees_array = np.arange(lowest_degree, highest_degree + 1, 1)
    for current_polynomial_degree in polynomial_degrees_array:
        # fit a new model using the current polynomial degree
        polynomial_feature_matrix = get_feature_matrix_with_all_polynomial_combinations(feature_matrix,
                                                                                        current_polynomial_degree)
        # Split to train and test
        polynomial_matrix_train, polynomial_matrix_test, true_labels_train, true_labels_test = (
            train_test_split(polynomial_feature_matrix, true_labels, test_size=0.2, shuffle=True))

        polynomial_regression_model.fit(polynomial_matrix_train, true_labels_train)
        current_polynomial_degree_error = 1 - polynomial_regression_model.score(polynomial_matrix_test, true_labels_test)
        polynomial_degrees_errors.append(current_polynomial_degree_error)

    return list(polynomial_degrees_array), list(polynomial_degrees_errors)


def print_polynomial_degrees_errors(polynomial_degrees_array, polynomial_degrees_errors) -> None:
    """
    Visualizes the relationship between polynomial degrees and their errors in a scatter plot.
    Args:
    - polynomial_degrees_array: Array of polynomial degrees.
    - polynomial_degrees_errors: Array of corresponding sum of squared errors.
    """
    plt.scatter(polynomial_degrees_array, polynomial_degrees_errors)
    # Set x-axis ticks to show only integers in jumps of 1
    plt.xticks(np.arange(min(polynomial_degrees_array), max(polynomial_degrees_array) + 1, 1))
    # Add text for each point
    for (polynomial_degree, error) in zip(polynomial_degrees_array, polynomial_degrees_errors):
        plt.text(polynomial_degree + 0.1, error, f'{error:.4f}', fontsize=7, verticalalignment='center')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('1 - R^2')
    plt.title('Polynomial Degree vs Model Error')
    plt.show()


def get_degree_with_lowest_error(polynomial_degrees_array, polynomial_degrees_errors) -> int:
    """
    Finds the polynomial degree with the lowest error from the provided array of degrees and corresponding errors.
    Args:
    - polynomial_degrees_array: Array of polynomial degrees.
    - polynomial_degrees_errors: Array of corresponding sum of squared errors.
    Returns:
    - int: Polynomial degree with the lowest error.
    """
    return min(zip(polynomial_degrees_array, polynomial_degrees_errors),
               key=lambda degree_and_error_tuple: degree_and_error_tuple[1])[0]


def main5() -> None:
    """
    Answer for question 5
    """
    data_set = pd.read_csv("Students_on_Mars.csv")
    true_labels = data_set["y"]
    feature_matrix = data_set.drop(columns=["y"])

    polynomial_degrees_array, polynomial_degrees_errors = get_polynomial_degrees_errors(feature_matrix, true_labels)
    # Get the degree that achieves the lowest error value and fit a model using it
    best_polynomial_degree = get_degree_with_lowest_error(polynomial_degrees_array, polynomial_degrees_errors)
    polynomial_feature_matrix = get_feature_matrix_with_all_polynomial_combinations(feature_matrix,
                                                                                    best_polynomial_degree)
    print(f"The selected polynomial degree is {best_polynomial_degree}.")
    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(polynomial_feature_matrix, true_labels, test_size=0.2, shuffle=True))

    linear_regression_model = LinearRegression()
    linear_regression_model.fit_using_train_and_print_test_score(feature_matrix_train, feature_matrix_test,
                                                                 true_labels_train, true_labels_test)

    # Print all the polynomial degrees and their errors in a scatter plot
    print_polynomial_degrees_errors(polynomial_degrees_array, polynomial_degrees_errors)


if __name__ == '__main__':
    print("\n---------- This is the main for Question 3 ----------\n")
    main3()
    # print("\n---------- This is the main for Question 4 ----------\n")
    # main4()
    # print("\n---------- This is the main for Question 5 ----------\n")
    # main5()
