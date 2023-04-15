"""

Author's Name          : Esha Esha
Registration Number    : 041041612
Course Number and Name : CST 2102 - Business Intelligence Programming

Program Name           : Esha Esha - Main.py
Date Written           : 10 Dec 2022
Purpose/Description    : Program is designed to perform data analytics. Program has two menu and performs below mentioned functions.
                       : Load data set from sklearn.
                       : Explore and split the data set.
                       : Train the chosen model.
                       : Test the chosen model.
                       : Visualization and graph representation.
                       : Performs regression.

Copyright © 2022 Esha Vig. All rights reserved.

"""



# Import list of all libraries
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


# Function declaration/definition
# load_dataset : Function to load the dataset
def load_data(dataset_name):
    # try loading the dataset
    try:
        # if dataset is california
        if dataset_name == "California":
            model = fetch_california_housing()
        # if dataset is diabetes
        else:
            model = load_diabetes()
        print(model)
    # if error occurs
    except 'Load Datset Error':
        print('\nError in performing Load dataset')
    # if load successful
    else:
        print("\nData has been loaded")
        return model


# explore_data : Function to explore the dataset
def explore_data(model):
    # try exploring data
    try:
        df = pd.DataFrame(model.data, columns=model.feature_names)
        # display the dataset
        print(df.head())
    # data extraction error
    except 'Explore Data Error':
        print('\nError in explore_data')
    # if data extraction is successful
    else:
        print("\nData exploration has been done")


# split_data : Function to split the dataset
def split_data(model):
    # try spliting data
    try:
        X_train, X_test, y_train, y_test = train_test_split(model.data, model.target, random_state=11)
    # if error
    except 'Splitting Data Error' :
        print("\nError in splitting the data")
    # sent successful message
    else:
        print("Data splitting has been done")
        return X_train, X_test, y_train, y_test


# train_model : Function to train the model
def train_model(model, X_train, y_train):
    # try training model
    try:
        linear_regression = LinearRegression()
        linear_regression.fit(X=X_train, y=y_train)
        LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None)
        for i, name in enumerate(model.feature_names):
            print(f'{name:>10}: {linear_regression.coef_[i]}')
    # if error
    except 'Training Data Error':
        print('\nError in training dataset')
    # send successful message
    else:
def test_model(linear_regression, X_test, y_test):
        print("\nTraining model has been done")
        return linear_regression


# test_model : Function to test the model

# visualize  Function to perform visualization
def visualize(predicted, expected):
    # try visualization
    try:
        print(expected,'\n',predicted)
        df = pd.DataFrame()
        df['Expected'] = pd.Series(expected)
        df['Predicted'] = pd.Series(predicted)
        figure = plt.figure(figsize=(9, 9))
    # try testing the model
    try:
        predicted = linear_regression.predict(X_test)
        expected = y_test
    # Test model error
    except 'Test Model Error':
        print('\nError in testing model/')
    # if successful
    else:
        print("\nModel testing has been done")
        return predicted, expected


# regression : Function to perform the regression
def regression(predicted,expected):
    # try regression
    try:
        print('\nCoefficient of Determination : ', metrics.r2_score(expected, predicted))
        print('\nMean Squared Error : ', metrics.mean_squared_error(expected, predicted))
    # if error
    except 'Regression Error':
        print("\nError in regression.")
    # if successful - print message
    else:
        print("\nRegression modelling has been done")

        axes = sns.scatterplot(data=df, x='Expected', y='Predicted', hue='Predicted', palette='cool', legend=False)
        start = min(expected.min(), predicted.min())
        end = max(expected.max(), predicted.max())
        axes.set_xlim(start, end)
        axes.set_ylim(start, end)
        line = plt.plot([start, end], [start, end], 'k--')
        plt.show()
    # if error occurs
    except 'Visualization Error':
        print("\nError in visualization.")
    # if successful - print message
    else:
        print("\nVisualization has been completed")



# Main function : call all other functions from here
if __name__ == "__main__":

    # Variables - declaration and initialization
    is_exit = False
    dataset_name = ''
    menu = '1'
    choice = ''

    # User Menu(s)
    while not is_exit:

        # Error handling for main menu
        try:
            # First option Menu/Main menu
            while menu == '1':

                # variable initialization/Indicators
                is_load = False
                is_explore = False
                is_split = False
                is_train = False
                is_test = False
                is_visualize = False
                is_regress = False

                # Get user choice
                choice = input("\nYou have following options\n1.Calofornia Dataset\n2.Diabtetes data set\n3.Exit\nPlease enter your choice : ")

                # If dataset is california housing
                if choice == '1':
                    dataset_name = "California"
                    menu = '2'

                # if dataset is diabetes
                elif choice == '2':
                    dataset_name = "Diabetes"
                    menu = '2'

                # if user want to exit
                elif choice == '3':
                    is_exit = True
                    break

                # if choice in invalid
                else:
                    print("\nPlease enter valid choice")
        # Main menu error
        except 'Main Menu Error':
            print('Error in main menu')
            exit()

        # Print second main menu
        try:
            # Menu - 2(load,explore,split,train,test,test,visualize,regress)
            while menu == '2':
                choice = input('\nPlease choose from following \n1. Load Dataset \n2.Explore Dataset \n3.Split Dataset \n4.Train Dataset \n5.Test Model \n6.Visualize Dataset \n7.Regression \n8.Goto to previous Menu\nPlease enter your choice : ')

                # if choice is 1
                if choice == '1':
                    # Load dataset
                    model = load_data(dataset_name)
                    # set indicator to true
                    is_load = True

                # if choice is 2
                elif choice == '2':
                    # check if previous step is run
                    if not is_load:
                        print('\nPerform Loading before proceeding')
                        continue
                    # Explore dataset
                    explore_data(model)
                    # set indicator to true
                    is_explore = True

                # if choice is 3
                elif choice == '3':
                    # check if previous step is run
                    if not is_load:
                        print('\nPerform Loading before proceeding.')
                        continue
                    # Split dataset
                    X_train, X_test, y_train, y_test = split_data(model)
                    # set indicator true
                    is_split = True

                # if choice is 4
                elif choice == '4':
                    # check if previous step is run
                    if not is_split:
                        print('\nPerform splitting before proceeding')
                        continue
                    # Train model
                    linear_regression = train_model(model, X_train, y_train)
                    # set indicator to true
                    is_train = True

                # if choice is 5
                elif choice == '5':
                    # check if previous step is performed
                    if not is_train:
                        print('\nPerform Training before proceeding')
                        continue
                    # Test model
                    predicted, expected = test_model(linear_regression, X_test, y_test)
                    # set indicator to true
                    is_test = True

                # if choice is 6
                elif choice == '6':
                    # check if previous step is performed
                    if not is_test:
                        print('\nPerform Testing before proceeding')
                        continue
                    # Visualize
                    visualize(predicted, expected)
                    # set indicator to true
                    is_visualize = True

                # if choice is 7
                elif choice == '7':
                    # check if previous step is performed
                    if not is_test:
                        print('\nPerform testing before proceeding')
                        continue
                    # Regression
                    regression(predicted, expected)
                    # set indicator to true
                    is_regress = True

                # if choice is 8
                elif choice == '8':
                    # set previous menu on
                    menu = '1'
                    # Exiting to main menu
                    break

                # Invalid choice
                else:
                    print('\nPlease select a valid choice')

        # if error in menu 2
        except 'Error in Menu 2':
            print('Error in Menu 2. Exiting to Main Menu')
            # set menu 1 to true
            menu ='1'
            # goto menu 1
            break

# End of the program

