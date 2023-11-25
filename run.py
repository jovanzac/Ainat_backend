import warnings
warnings.filterwarnings("ignore")
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from Utils.opencv import VideoPoseEstimation

if __name__ == "__main__" :
    estimator = VideoPoseEstimation()
    estimator.predict_from_video("./vids/drowning/5.mp4", emit_func=True)