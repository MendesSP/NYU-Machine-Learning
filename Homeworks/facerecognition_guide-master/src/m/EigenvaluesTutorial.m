%Eigenvalues Tutorial

%define the path for the folder with the images
path_fn = '/Users/andremendes/facerec/data/att_faces';

%List the files using the function list_files
L = list_files(path_fn);

%Read each image in the folders
[X y width height] = read_images(path_fn);

%Perform PCA in the image
[W, mu] = pca(X, y, k);


