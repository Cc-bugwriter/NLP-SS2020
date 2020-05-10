function [a] = ReLU(z)
%RELU 此处显示有关此函数的摘要
%   此处显示详细说明
a = zeros(size(z));
a((find(z>0))) = z((find(z>0)));
end

