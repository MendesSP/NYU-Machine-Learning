clear all
clc
close all

addpath('/Users/andremendes/OneDrive/LibSVM');

load('cs6923Project.mat');

seed=1;
rng(seed);

dataset = [train train_label];
dataset = dataset(randperm(size(dataset,1)),:);
train_cv = dataset(1:35000,:);
test_set = dataset(35001:50000,:);

data = train_cv(:,1:77);
labels = train_cv(:,78);

opts = '-s 0 -t 2 -c 4 -g 4';    %# libsvm training options
nfold = 10;

acc = svmtrain(labels, data, sprintf('%s -v %d -q',opts,nfold));
model = svmtrain(labels, data, strcat(opts,' -q'));
fprintf('Cross Validation Accuracy = %.4f%%\n', 100*mean(acc));






