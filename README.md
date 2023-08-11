# f1-predictor-gui
*FORMULA ONE WINNER PREDICTION * 
this is the gui version of the formula one predictor where predictions are made based on the driver and constructor perfomance combined.
The prdictions are made based on the voting classifier using an ensemble model which contains all the models listed in my previous formula one predictor repository in the same profile.
the gui is made using the tkinter library.
the driver confidence and the constructor reliability is fetched from a dictionary because they are all predetermined values present in the dataset.
the gui contains four option menu input columns 1) driver 2) constructor 3)grand prix name 4)the starting grid.
and then there is a text label which shows the result whether the driver will finish in the (top 3) podium or points finish (pos 4 to 10) or a no points finish (pos 11 to 20)
an exit button is added to get out of the user interface
an f1 image background is added which can be changed mentioning the image path.
