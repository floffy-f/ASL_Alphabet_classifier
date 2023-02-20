# ASL_Alphabet_classifier
Classifier for the gestures from the ASL alphabet

Please download the database useful to this project at [this link](http://www.massey.ac.nz/~albarcza/gesture_dataset2012.html),
and cite the original author when using this project :

```
@article{Barczak11,
  author = {Barczak, Andre and Reyes, Napoleon and Abastillas, M and Piccio, A and Susnjak, Teo},
  year = {2011},
  month = {01},
  pages = {},
  title = {A New 2D Static Hand Gesture Colour Image Dataset for ASL Gestures},
  volume = {15},
  journal = {Res Lett Inf Math Sci}
}
```

Use the relevant LICENSE file as well.

## Requirements

`deprecated`, `moviepy`, `webbrowser`, `mediapipe`, `OpenCV`, `sklearn`, `numpy`, `getpass`.

## How to use

Put the database in a directory called `train_set/images` at the root of this project, then launch the skeletization program after creating an empty `train_set/skeleton/` directory:
```bash
$ python3 src/main.py skeletize load
```
Then launch the inference :
```bash
$ python3 src/main.py classify
```
You can now perform the gestures and verify that the training went well, quit with the `q` or `<esc>` keys.
