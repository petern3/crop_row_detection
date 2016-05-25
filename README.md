crop_row_detection
==================

A useful tool for agricultural robot developers to have is the ability for the
robot to be able to detect crop rows. There have been many methods developed
for this purpose, but this repository contains a real-time algorithm that works
in a variety of conditions.

Benefits
--------
Although thas algorithm is far from perfect (it only detects about 50%
of the rows, I've put ways to improve below), there are several good
things about the algorithm:

1. It took an average of 32.1ms to process each image (max of 70ms) on
   my laptop (2.5Ghz intel i5, 2Gb ram). This means it can be run in
   real-time
2. Because of the grayscale transform, which does 2*g-r-b, the green
   values come out more, and it also has some lighting invarience. This
   hasn't been tested though.
3. The algorithm iteratively increases the Hough threshold parameter,
   which means it is more robust to different kinds of crops at
   different stages
4. Even on curved crop rows, it prioritizes the angle of the rows
   closer to the camera simply because closer rows have more pixels
   (perspective). This means that a robot would detect the angle of the
   rows closer rather than further away.

Things to Improve
-----------------
Should you want to develop this algorithm further, these are the most
important aspects I've picked out:

### Finding a Variable Number of Rows

At the moment the algorithm seems to be the most effective at
finding precisely three crop rows. Firstly, it should be modified
such that any number of rows works roughly evenly, and then some
criteria made that means that the number of rows to find does not
have to be manually set.

### Iterative Optimization of Skeleton Threshold

One of the first steps of the skeletonization process is to apply a
threshold. In order to get the most effective skeleton, the
threshold could be repeated until some criteria achieved that gives
the best output for the Hough transform. This would give better
outputs for both more sparse crops (image 5 in CRBD) and more dense
crops.

### Filter for a 'Vanishing Point'

Instead of removing faulty rows based purely on angle, they can be
removed if it doesn't go to a similar vanishing point to the rest of
the rows. Ideally it wouldn't just be a plain threshold if the top
point of the line is in the centre - the vanishing point may be to
one side if the camera is not looking down the rows or if the rows
are curved.

### Testing of Other Colour Spaces

The first step in this and many other algorithms is to find the
green values in the image in one way or another. This works for many
of the images in the CRBD, but there are three cases that this
method has been untested and may not function:

1. The crop is not green
2. The plants are too small
3. The plants are too large and there are no gaps between rows

In these cases, it could be worth investigating other colour spaces
for whichever consistently has the most contrast between where the
crop rows are and aren't. Such colour spaces that could be compared
are YUV and HSV, with the RGB that the proposed method uses. If none
of these methods are able to detect rows when there are no gaps, it
would also be worth investigating if the robot is able to drive
between them at all without damaging the crops. It could be that no
detected rows is a beneficial feature.

### Detection of Curved Crop Rows

Many crop rows are not linear due to circular irrigators or various
landscapes. In these cases, a simple set of lines may not be enough.
There are two ways one could go about this: Either use a Hough
transform to find circles treating rows as circles of various radii,
or aligning the lines to the bottom part of the crop rows, as that
is the important part for a mobile robot.