from lstm_model import LSTMModel
from opencv import VideoPoseEstimation, SupportFunctions


if __name__ == "__main__" :
    support = SupportFunctions()
    estimator = VideoPoseEstimation()
    lstm = LSTMModel()
    
    # Creating dir for keypoints
    # support.create_dir_for_keypoints()
    
    # Collecting training data
    print("Collecting training data")
    # estimator.collecting_training_data()
    
    # Preprocessing and creating feature sets
    print("Preprocessing data")
    # X_train, X_test, y_train, y_test = lstm.preprocess_data_for_model()
    
    # Train model
    print("Training model")
    # lstm.train_model(X_train, y_train, "./action2.h5")
    
    # Load saved model
    # model = lstm.load_lstm_model("./action.h5")
    
    # Perform prediction on video
    estimator.predict_from_video("./vids/drowning/5.mp4")