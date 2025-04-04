# APSC 258 Design Project Report

We needed to design a model that effectively allows for the PiCar to follow a green line **in real time**. This creates the following design requirements:

1. The model must output adequate turn angles for all possible track turns (within 3-5 degrees all the time).
2. The model must retain consistent outputs to avoid turning jitter, i.e. not output turning angles of 93 and 87 immediately after when following a straight line.
3. The model must by lightweight enough to run in real time, being able to work alongside video streaming and processing from the PiCar and continuous transmission of turning commands from a laptop.

When evaluating our models' performance, we used the following criteria:
- How well does the model keep the PiCar close to the center of the track?
- How volatile is the angle output during evaluation?
- How big is the model (how many parameters, size of model file)?

To speed up our model evaluation and architecture design, we opted to run tensorflow locally on laptops with dedicated GPUs.

## Data Collection

The data is collected from 3 maps based on the anticipated conditions for our IRL PiCar:

1. A modified default map with windy sections, improving turn variety. 16000 x 2 images.
2. A map with right-angle turns to mimic tape-laid track turns. 8000 x 2 images.
3. A map where tracks are placed side by side, and the center line of both are visible to the PiCar simultaneously. This hopefully makes the model more resilient to noise. 8000 x 2 images.

For all maps, the car was recorded driving on the track in both directions (clockwise and counter clockwise) to insure uniform coverage of all turning angles. This wisdom comes from methods in the 2023W2 class, where teams created flipped-duplicates of training images since the turn angles were skewed to right turns. The result is a total of 64000 images.

## Preprocessing

All images go through a color-channel reduction to the green centerline of the tracks. Green pixels are filtered out from the images, creating a grayscale view, and only the center of the track being illuminated.