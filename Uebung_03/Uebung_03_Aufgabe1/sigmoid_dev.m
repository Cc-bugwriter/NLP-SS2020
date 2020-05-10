function [a] = sigmoid_dev(z)
%SIGMOID_DEV 此处显示有关此函数的摘要
%   此处显示详细说明
a = sigmoid(z).*(1-sigmoid(z));
end

