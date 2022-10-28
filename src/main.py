# from src.hand_classifier import HandModelKNN
# from src.skeletize import skeletize
from hand_classifier import HandModelKNN
from skeletize import skeletize
import argparse as ap

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

def main():
    parser = ap.ArgumentParser(
        prog='ASL hands gestures alphabet classifier (KNN)',
        description="Hand classifier to detect the 'SAW' word in ASL hand gestures",
        epilog='The code is almost not commented, sorry about that'
        )
    parser.add_argument("choice", choices=['skeletize', "classify", "load"])
    result = parser.parse_args()
    if result.choice == 'classify':
        print("Launching model")
        model = HandModelKNN(1, URL)
        model.read_cam()
        print("End of session")
    elif result.choice == 'load':
        model = HandModelKNN(1, URL, loading=True)
    else:
        skeletize()
    return 0

if __name__ == "__main__":
    main()