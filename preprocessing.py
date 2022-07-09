import opendatasets as od
import pandas as pd
import numpy as np
import os

def download_kaggle_data():
    # Import data for kaggle spaceship titanic from https://www.kaggle.com/competitions/spaceship-titanic/data
    od.download('https://www.kaggle.com/competitions/spaceship-titanic/data')

def process_data(path_to_csv):
    """
    Read content of csv data and create new csv data with columns adjusted for machine learning.
    """
    # Read data from csv file as pandas dataframe
    # Collumns are: PassengerId,HomePlanet,CryoSleep,Cabin,Destination,Age,VIP,RoomService,FoodCourt,ShoppingMall,Spa,VRDeck,Name,Transported
    # First data row: 0001_01,Europa,False,B/0/P,TRAPPIST-1e,39.0,False,0.0,0.0,0.0,0.0,0.0,Maham Ofracculy,False
    input_data = pd.read_csv(path_to_csv)

    #Prepare new, empty dataframe for output
    output_data = pd.DataFrame()

    # Add PassangerId to output data
    output_data['PassengerId'] = input_data['PassengerId']

    # Add one-hot encoded columns for each option (Earth, Mars, Europa) in the HomePlanet column in input data.
    # Missing data is ignored.
    output_data['HomePlanet_Earth'] = input_data['HomePlanet'].apply(lambda x: 1 if x == 'Earth' else 0)
    output_data['HomePlanet_Mars'] = input_data['HomePlanet'].apply(lambda x: 1 if x == 'Mars' else 0)
    output_data['HomePlanet_Europa'] = input_data['HomePlanet'].apply(lambda x: 1 if x == 'Europa' else 0)

    # Add one-hot encoded column CryptoSleep based on the value in the CryoSleep column in input data.
    # If the value is 'True', the column is 1, otherwise 0. missing values are treated as 0.
    output_data['CryptoSleep'] = input_data['CryoSleep'].apply(lambda x: 1 if x == True else 0)

    # Create new dataframe for cabin data
    cabin_data = pd.DataFrame()
    # Add columns Cabin_p1, Cabin_p2 and Cabin_p3 to cabin_data based on the value in the Cabin column in input data.
    # The value of input data is a string, so we need to split it into three parts with "/" as separator.
    # The first part is the Cabin_p1, the second part is Cabin_p2 and the third part is Cabin_p3.
    # If the input data is empty, put None in the columns.
    cabin_data['Cabin_p1'] = input_data['Cabin'].apply(lambda x: str(x).split('/')[0] if len(str(x).split('/')) > 0 else None)
    cabin_data['Cabin_p2'] = input_data['Cabin'].apply(lambda x: str(x).split('/')[1] if len(str(x).split('/')) > 1 else None)
    cabin_data['Cabin_p3'] = input_data['Cabin'].apply(lambda x: str(x).split('/')[2] if len(str(x).split('/')) > 2 else None)

    # For cabin parts 1 from 'A' to G' add one-hot encoded columns to output data for each option (A,B,C,D,E,F,G)
    # Example: Cabin_p1_A = 1 if the value in Cabin_p1 is 'A', otherwise 0
    for letter in range(ord('A'), ord('G') + 1):
        output_data['Cabin_' + chr(letter)] = cabin_data['Cabin_p1'].apply(lambda x: 1 if x == chr(letter) else 0)

    # For cabin parts 2 add column Cabin_Number to output data. Value is the number in the second part of the cabin divided by 2000.
    # Example: Cabin_Number = 0 if the value in Cabin_p2 is empty, otherwise the value in Cabin_p2 divided by 2000.
    output_data['Cabin_Number'] = cabin_data['Cabin_p2'].apply(lambda x: 0 if x == None or str(x) == 'nan' else int(x) / 2000)

    # For cabin parts 3 add column to output data telling if the cabin is a type P cabin.
    # Example: Cabin_p3 = 1 if the value in Cabin_p3 is 'P', otherwise 0
    output_data['Cabin_P'] = cabin_data['Cabin_p3'].apply(lambda x: 1 if x == 'P' else 0)

    # Clean cabin data from memory
    del cabin_data

    # Add one-hot encoded columns for each option in the Destination column in input data.
    # Missing data is ignored.
    for destination in input_data['Destination'].unique():
        if str(destination) == None or str(destination) == 'nan':
            continue
        output_data['Destination_' + str(destination).replace(' ', '_')] = input_data['Destination'].apply(
            lambda x: 1 if str(x) == str(destination) else 0)

    # Add column Age to output data. Value is the age from input data divided by 100.
    # Round numbers to 4 decimal places.
    # Example: Age = 0 if the value in Age is empty, otherwise the value in Age divided by 100.
    output_data['Age'] = input_data['Age'].apply(lambda x: 0 if x == None or str(x) == 'nan' else int(x) / 100)

    # Add column VIP to output data. Value is 1 if the value in VIP is 'True', otherwise 0.
    # Example: VIP = 1 if the value in VIP is 'True', otherwise 0.
    output_data['VIP'] = input_data['VIP'].apply(lambda x: 1 if x == True else 0)

    # Add column RoomService to output data. Value is input value divided by 15000.
    # Round numbers to 4 decimal places.
    # Example: RoomService = 0 if the value in RoomService is empty, otherwise the value in RoomService divided by 15000.
    output_data['RoomService'] = input_data['RoomService'].apply(lambda x: 0 if x == None or str(x) == 'nan' else (int(x) / 15000).__round__(4))

    # Add column FoodCourt to output data. Value is input value divided by 30000.
    # Round numbers to 4 decimal places.
    # Example: FoodCourt = 0 if the value in FoodCourt is empty, otherwise the value in FoodCourt divided by 30000.
    output_data['FoodCourt'] = input_data['FoodCourt'].apply(lambda x: 0 if x == None or str(x) == 'nan' else (int(x) / 30000).__round__(4))

    # Add column ShoppingMall to output data. Value is input value divided by 30000.
    # Round numbers to 4 decimal places.
    # Example: ShoppingMall = 0 if the value in ShoppingMall is empty, otherwise the value in ShoppingMall divided by 30000.
    output_data['ShoppingMall'] = input_data['ShoppingMall'].apply(lambda x: 0 if x == None or str(x) == 'nan' else (int(x) / 30000).__round__(4))

    # Add column Spa to output data. Value is input value divided by 30000.
    # Round numbers to 4 decimal places.
    # Example: Spa = 0 if the value in Spa is empty, otherwise the value in Spa divided by 30000.
    output_data['Spa'] = input_data['Spa'].apply(lambda x: 0 if x == None or str(x) == 'nan' else (int(x) / 30000).__round__(4))

    # Add column VRDeck to output data. Value is input value divided by 30000.
    # Round numbers to 4 decimal places.
    # Example: VRDeck = 0 if the value in VRDeck is empty, otherwise the value in VRDeck divided by 30000.
    output_data['VRDeck'] = input_data['VRDeck'].apply(lambda x: 0 if x == None or str(x) == 'nan' else (int(x) / 30000).__round__(4))

    # If there is a Transported column in input data, add column Transported to output data.
    # Add column Transported to output data. Value is 1 if the value in Transported is 'True', otherwise 0.
    # Example: Transported = 1 if the value in Transported is 'True', otherwise 0.
    if 'Transported' in input_data.columns:
        output_data['Transported'] = input_data['Transported'].apply(lambda x: 1 if x == True else 0)

    return output_data

def generate_processed_csv(input_csv, output_csv):
    # Process data from input csv ans dave it as output csv
    output_data = process_data(input_csv)
    output_data.to_csv(output_csv, index=False)

def generate_train_npy_files(input_csv, output_npy_dir):
    """
    Open CSV file containing processed data located in input_csv
    Create numpy arrays from the data and save it to the output_npy_dir
    Arrays are:
    - train_x.npy: containing all the data (except the PassengerId and Transported column) from all rows except every 5th row
    - train_y.npy: containing only Transported columns from all rows except every 5th row
    - train_id.npy: containing only PassengerId column from all rows except every 5th row
    - valid_x.npy: containing data from every 5th row without the PassengerId and Transported column
    - valid_y.npy: containing only Transported column from every 5th row
    - valid_id.npy: containing only PassengerId column from every 5th row
    """
    # Load data from input csv
    data = pd.read_csv(input_csv)

    # Split data into train and valid data
    valid_data = data.iloc[::5, :]
    train_data = data.drop(data.index[::5])

    # Create train arrays
    train_x = train_data.drop(['PassengerId', 'Transported'], axis=1).values
    train_y = train_data['Transported'].values
    train_id = train_data['PassengerId'].values

    # Create valid arrays
    valid_x = valid_data.drop(['PassengerId', 'Transported'], axis=1).values
    valid_y = valid_data['Transported'].values
    valid_id = valid_data['PassengerId'].values

    # Save arrays to npy files
    np.save(os.path.join(output_npy_dir, 'train_x.npy'), train_x)
    np.save(os.path.join(output_npy_dir, 'train_y.npy'), train_y)
    np.save(os.path.join(output_npy_dir, 'train_id.npy'), train_id)
    np.save(os.path.join(output_npy_dir, 'valid_x.npy'), valid_x)
    np.save(os.path.join(output_npy_dir, 'valid_y.npy'), valid_y)
    np.save(os.path.join(output_npy_dir, 'valid_id.npy'), valid_id)

def generate_test_npy_files(input_csv, output_npy_dir):
    """
    Open CSV file containing processed data located in input_csv
    Create numpy arrays from the data and save it to the output_npy_dir
    Arrays are:
    - test_x.npy: containing all the data (except the PassengerId column) from all rows
    - test_id.npy: containing only PassengerId column from all rows
    """
    # Load data from input csv
    data = pd.read_csv(input_csv)

    # Create test arrays
    test_x = data.drop(['PassengerId'], axis=1).values
    test_id = data['PassengerId'].values

    # Save arrays to npy files
    np.save(os.path.join(output_npy_dir, 'test_x.npy'), test_x)
    np.save(os.path.join(output_npy_dir, 'test_id.npy'), test_id)

def convert_kaggle_csvs_to_npy_files():
    """
    Generate npy files for train and test from kaggle csvs
    train data is in <path-to-py>/spaceship-titanic/train.py
    test data is in <path-to-py>/spaceship-titanic/test.py
    """
    # set workdir to path where py script is located
    workdir = os.path.dirname(os.path.abspath(__file__))

    # Process train data
    generate_processed_csv(os.path.join(workdir,'spaceship-titanic', 'train.csv'), os.path.join(workdir,'spaceship-titanic', 'train_processed.csv'))
    # Process test data
    generate_processed_csv(os.path.join(workdir,'spaceship-titanic', 'test.csv'), os.path.join(workdir,'spaceship-titanic', 'test_processed.csv'))

    #Generate files
    generate_train_npy_files(os.path.join(workdir, 'spaceship-titanic', 'train_processed.csv'), os.path.join(workdir, 'spaceship-titanic'))
    generate_test_npy_files(os.path.join(workdir, 'spaceship-titanic', 'test_processed.csv'), os.path.join(workdir, 'spaceship-titanic'))

def get_train_data():
    """
    Load train_id.npy, train_x.npy, train_y.npy and return them as 3 variables
    """
    # set workdir to path where py script is located
    workdir = os.path.dirname(os.path.abspath(__file__))

    # Load arrays
    train_id = np.load(os.path.join(workdir, 'spaceship-titanic', 'train_id.npy'), allow_pickle=True)
    train_x = np.load(os.path.join(workdir, 'spaceship-titanic', 'train_x.npy'), allow_pickle=True)
    train_y = np.load(os.path.join(workdir, 'spaceship-titanic', 'train_y.npy'), allow_pickle=True)

    return train_id, train_x, train_y

def get_valid_data():
    """
    Load valid_id.npy, valid_x.npy, valid_y.npy and return them as 3 variables
    """
    # set workdir to path where py script is located
    workdir = os.path.dirname(os.path.abspath(__file__))

    # Load arrays
    valid_id = np.load(os.path.join(workdir, 'spaceship-titanic', 'valid_id.npy'), allow_pickle=True)
    valid_x = np.load(os.path.join(workdir, 'spaceship-titanic', 'valid_x.npy'), allow_pickle=True)
    valid_y = np.load(os.path.join(workdir, 'spaceship-titanic', 'valid_y.npy'), allow_pickle=True)

    return valid_id, valid_x, valid_y

def get_test_data():
    """
    Load test_id.npy and test_x.npy and return them as 2 variables
    """
    # set workdir to path where py script is located
    workdir = os.path.dirname(os.path.abspath(__file__))

    # Load arrays
    test_id = np.load(os.path.join(workdir, 'spaceship-titanic', 'test_id.npy'), allow_pickle=True)
    test_x = np.load(os.path.join(workdir, 'spaceship-titanic', 'test_x.npy'), allow_pickle=True)

    return test_id, test_x

def create_submission_csv(predictions, output_csv):
    """
    Create a csv file with the predictions
    Example:
        PassengerId,Transported
        0013_01,False
        0018_01,False
        0019_01,False
        0021_01,False
    """
    workdir = os.path.dirname(os.path.abspath(__file__))

    # Load test PassengerIds
    test_id = np.load(os.path.join(workdir, 'spaceship-titanic', 'test_id.npy')).astype(str)

    # Convert predictions to boolean
    predictions = predictions.astype(bool)

    # Create dataframe where PassengerId column is test_id and Transported column is predictions
    df = pd.DataFrame({'PassengerId': test_id, 'Transported': predictions})

    # Write dataframe to csv
    df.to_csv(output_csv, index=False)

def describe_data():
    train_id, train_x, train_y = get_train_data()
    valid_id, valid_x, valid_y = get_valid_data()
    test_id, test_x = get_test_data()

    print('Train data:')
    print('  - id:', train_id.shape)
    print('preview:', train_id[:5])
    print('  - x:', train_x.shape)
    print('preview:', train_x[:5])
    print('  - y:', train_y.shape)
    print('preview:', train_y[:5])

    print('Valid data:')
    print('  - id:', valid_id.shape)
    print('preview:', valid_id[:5])
    print('  - x:', valid_x.shape)
    print('preview:', valid_x[:5])
    print('  - y:', valid_y.shape)
    print('preview:', valid_y[:5])

    print('Test data:')
    print('  - id:', test_id.shape)
    print('preview:', test_id[:5])
    print('  - x:', test_x.shape)
    print('preview:', test_x[:5])
