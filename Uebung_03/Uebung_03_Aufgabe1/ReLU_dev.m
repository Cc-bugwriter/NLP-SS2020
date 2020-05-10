function [a] = ReLU_dev(z)
%RELU_DEV 此处显示有关此函数的摘要
%   此处显示详细说明
a = zeros(size(z));
a((find(z>0))) = 1;
end

