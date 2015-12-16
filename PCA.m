%%Aplying PCA using the tutorial http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf

%Author: Andre Mendes


%% Initialization
clear ; close all; clc

%% Step 1 - Get the Data
fprintf('Step 1 - Get Data\n');
x= [2.5 0.5 2.2 1.9 3.1 2.3 2 1 1.5 1.1]';
y=[2.4 .7 2.9 2.2 3 2.7 1.6 1.1 1.6 .9]';

fprintf('Ploting Data\n');
figure(1)
plot(x,y,'+');
axis([-1 4 -1 4]);
legend('Original Data');
xlabel('x');
ylabel('y');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% Step 2 - Subtract the mean
xMean = mean(x);
yMean = mean(y);
dataAjust = [x-xMean y-yMean];

fprintf('Step 2 - Subtract the mean\n');
figure(2)
plot(dataAjust(:,1),dataAjust(:,2),'+');
axis([-2 2 -2 2]);
xlabel('x');
ylabel('y');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% Step 3 - Calculate the covariance matrix
fprintf('Step 3 - Calculate the covariance matrix\n');

covMatrix = cov(dataAjust(:,1),dataAjust(:,2));

fprintf('Program paused. Press enter to continue.\n');
pause;
%% Step4 - Calculate eigenvalues and eigenvectors of the covariance matrix
fprintf('Step4 - Calculate eigenvalues and eigenvectors of the covariance matrix\n');

[eigenvectors, eigenvalues] = eig(covMatrix);

hold on
plot(dataAjust(:,1),(eigenvectors(1,1)/eigenvectors(2,1))*dataAjust(:,1),'r--');
plot(dataAjust(:,2),(eigenvectors(1,2)/eigenvectors(2,2))*dataAjust(:,2),'m--');
legend('Ajusted Data','Eigenvector 1','EigenVector 2');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% Step 5 - Choosing Components and forming a feature vector
%Step 5.1 - Dimension Reduction
fprintf('Step 5 - Choosing Components and forming new feature vector \n');
fprintf('Step 5.1 - Dimension Reduction\n');

featureVector1 = [eigenvectors(:,2)]; %choose the eigenvector corresponding to the highest eigenvalue
featureVector2 = [eigenvectors(:,2) eigenvectors(:,1)];

fprintf('Program paused. Press enter to continue.\n');
pause;
%% Step 5.2 - Get the new Data set
fprintf('Step 5.2 - Get the new Data set\n');

rowFeatureVector1 = featureVector1';
rowDataAjust = dataAjust';
finalData1 = rowFeatureVector1*rowDataAjust;

rowFeatureVector2 = featureVector2';
rowDataAjust = dataAjust';
finalData2 = rowFeatureVector2*rowDataAjust;

figure(3)
plot(finalData2(1,:),finalData2(2,:),'+');
axis([-2 2 -2 2]);
xlabel('Eigenvector 1');
ylabel('Eigenvector 2');
legend('Transformed Data with two eigenvectors');

figure(4)
plot(dataAjust(:,2),finalData1(:),'+');
xlabel('y');
ylabel('Eigenvector 2');
legend('Transformed Data with one eigenvector');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% Get the data back
fprintf('Get the data back');
recoverRowDataAjust1 = (rowFeatureVector1'*finalData1)+yMean;

figure(5)
plot(recoverRowDataAjust1(1,:),recoverRowDataAjust1(2,:),'+');
axis([-1 4 -1 4]);
xlabel('x');
ylabel('y');
legend('Recovered Data with one eigenvector');

recoverRowDataAjust2 = (rowFeatureVector2'*finalData2)';
recoverRowDataAjust2 = [ recoverRowDataAjust2(:,1)+xMean  recoverRowDataAjust2(:,2)+yMean];

figure(6)
plot(recoverRowDataAjust2(:,1),recoverRowDataAjust2(:,2),'+');
axis([-1 4 -1 4]);
xlabel('x');
ylabel('y');
legend('Recovered Data with two eigenvectors');





