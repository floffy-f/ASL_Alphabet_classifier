# from src.hand_classifier import HandModelKNN
# from src.skeletize import skeletize
from hand_classifier import HandModelKNN
from pin import ask_pin
from skeletize import skeletize
import argparse as ap

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

def main():
    parser = ap.ArgumentParser(
        prog='ASL hands gestures alphabet classifier (KNN)',
        description="Hand classifier to detect the 'SAW' word in ASL hand gestures",
        epilog='The code is almost not commented, sorry about that'
        )
    parser.add_argument("choice",
                        choices=["skeletize", "classify", "load"],
                        nargs="*",
                        default="classify")
    result = parser.parse_args()
    if len(result.choice) == 0:
        result.choice.append("classify")
    if 'skeletize' in result.choice:
        skeletize()
    if 'load' in result.choice:
        model = HandModelKNN(1, URL, loading=True)
    if 'classify' in result.choice:
        ask_pin()
        print("Launching model")
        model = HandModelKNN(1, URL)
        model.read_cam()
        print("End of session")
    return 0

if __name__ == "__main__":
    main()