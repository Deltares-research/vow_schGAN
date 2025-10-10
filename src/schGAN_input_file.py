# This is a script that reads the coordiantes of the CPTs and organizes them from west to east (left to right).
# Then it calculates the euclidean distance in two ways: first the distance between the first cpt and all the rest
# and then the distance between each cpt and the next one.
# Once we know all the distances, we can create the input file for schemaGAN. This uses a csv file of size 512x32
# with 512 columns (representing the distance) and 32 rows (representing the depth. In each position a value is assigned
# to represent the soil type at that depth and distance. To do this we first create a 512x32 matrix filled with 0s
# and then we find the closest cpt to each column and assign the soil type of that cpt to all the rows of that column.
# Finally we save the matrix as a csv file. Key here is to keep track of the distance scale. It is not necesarry
# to have a 1:1 scale. First lets find the max distance, then we can divide that by 512 see how many sections of
# that size fit in the max distance. We want to fit around 6 CPTs per 512 columns, so we can adjust the scale accordingly.
