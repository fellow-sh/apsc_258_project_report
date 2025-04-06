# Objectives

We needed to design a model that effectively allows for the PiCar to follow a green line **in real time**. This creates the following design requirements:

1. The model must output adequate turn angles for all possible track turns (within 3-5 degrees all the time).

The PiCar must navigate a track with potentially sharp or subtle turns. If the model outputs inaccurate turn angles, the car may veer off the track or fail to follow the green line. A precision of 3-5 degrees ensures the car can handle a wide range of track configurations while maintaining smooth movement.

2. The model must retain consistent outputs to avoid turning jitter, i.e. not output turning angles of 93 and 87 immediately after when following a straight line. 

Jitter refers to rapid, erratic changes in the turning angle, which can cause the car to wobble or behave unpredictably. This is especially problematic on straight paths, where the car should maintain a steady course. Consistent outputs ensure stability and smooth operation, improving the user experience and reducing wear on the car's components.

3. The model must by lightweight enough to run in real time, being able to work alongside video streaming and processing from the PiCar and continuous transmission of turning commands from a laptop.

While most of the system processing is handled by a laptop where it must process video input, detect the green line, calculate turn angles, and transmit commands—all in real time. A lightweight model ensures that these tasks can run simultaneously without lag, enabling the car to respond quickly to changes in the track. This is critical for maintaining real-time performance and avoiding delays that could lead to errors in navigation.

These objectives collectively aim to balance accuracy, stability, and efficiency, ensuring the PiCar can reliably follow the green line in a dynamic, real-world environment.